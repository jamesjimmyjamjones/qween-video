import os, io, base64, time, uuid, threading, json, hashlib
# NOTE: torch is imported LAZILY inside get_pipe(), not here. Importing
# torch takes a minute+ on cold disk, and startup probes fail a container
# whose port isn't open yet. Binding the port fast and answering
# 200-while-loading keeps slow boots alive (proven on the image stack).
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, Header, HTTPException, Response, Request
from pydantic import BaseModel
from typing import List, Optional

# Persistent scratch disk (survives scale-to-zero): weights download once,
# every later boot reuses them.
os.environ.setdefault("HF_HUB_CACHE", "/data/hf")

# Config via deployment env vars (set in the console, not baked in).
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.1-I2V-14B-720P")
# NSFW text encoder swap (the load-bearing piece for NSFW vocabulary):
# umT5-XXL is SHARED across all Wan 2.1 sizes/tasks, so one file covers
# the 14B I2V stack. Loaded over the stock encoder after from_pretrained.
TEXT_ENCODER_REPO = os.environ.get("TEXT_ENCODER_REPO", "NSFW-API/NSFW-Wan-UMT5-XXL")
TEXT_ENCODER_FILE = os.environ.get("TEXT_ENCODER_FILE", "nsfw_wan_umt5-xxl_bf16.safetensors")
TEXT_ENCODER_ENABLE = os.environ.get("TEXT_ENCODER_ENABLE", "1") == "1"
# Optional DiT-side LoRA (renderer tuning; encoder stays the priority).
# Empty = no LoRA, exactly today's behavior.
LORA_REPO = os.environ.get("LORA_REPO", "")
LORA_FILE = os.environ.get("LORA_FILE", "")
LORA_SCALE = float(os.environ.get("LORA_SCALE", "1.0"))
API_KEY = os.environ.get("API_KEY", "").strip()
# Placement: OFFLOAD=1 enables sequential CPU offload (escape hatch if a
# config ever exceeds VRAM). Default 0 = whole pipeline on GPU.
OFFLOAD = os.environ.get("OFFLOAD", "0") == "1"
# Fixed frame geometry (server-side = predictable VRAM; the app does not
# get a knob that can OOM the box). 81 frames @16fps ≈ 5s.
VIDEO_H = int(os.environ.get("VIDEO_H", "720"))
VIDEO_W = int(os.environ.get("VIDEO_W", "1280"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "81"))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "16"))

pipe = None
loading_error = None
encoder_note = "stock"

def _background_load():
    # Holds the pipe lock so a job arriving mid-load waits instead of
    # starting a second full load next to this one (two loads = OOM).
    global loading_error
    try:
        with _pipe_lock:
            get_pipe()
    except Exception as e:
        loading_error = str(e)[:500]

def get_pipe():
    import torch  # lazy: see note at top of file
    global pipe, encoder_note
    if pipe is None:
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        if OFFLOAD:
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        try:
            # Video attention is the VRAM hog; slicing trades some speed
            # for a much lower peak. Best-effort: not all builds expose it.
            pipe.enable_attention_slicing()
        except Exception:
            pass
        if TEXT_ENCODER_ENABLE and TEXT_ENCODER_REPO:
            # Swap the stock umT5 for the NSFW-trained one (same arch, so
            # a state-dict load; strict=False tolerates harmless key drift,
            # and the counts are reported in /status for verification).
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            enc_path = hf_hub_download(TEXT_ENCODER_REPO, TEXT_ENCODER_FILE,
                                       cache_dir="/data/hf")
            state = load_file(enc_path)
            missing, unexpected = pipe.text_encoder.load_state_dict(state, strict=False)
            encoder_note = "%s/%s missing=%d unexpected=%d" % (
                TEXT_ENCODER_REPO, TEXT_ENCODER_FILE, len(missing), len(unexpected))
        if LORA_REPO:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE or None,
                                   adapter_name="nsfw")
            pipe.set_adapters(["nsfw"], adapter_weights=[LORA_SCALE])
        pipe.set_progress_bar_config(disable=True)
    return pipe

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load in the background, answer HTTP immediately (see note at top).
    threading.Thread(target=_background_load, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def _cors(request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": request.headers.get(
                "access-control-request-headers", "content-type, x-api-key, authorization"),
            "Access-Control-Max-Age": "600",
        })
    # Real responses need the header too (preflight passes, then the
    # browser blocks the reply without it).
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# Payload privacy (E2E through any relay/proxy): IDENTICAL envelope to the
# image stack (AES-256-GCM, PBKDF2-SHA256 200k, 16B salt, 12B iv, no AAD),
# so the app reuses its seal code byte-for-byte. Empty = plaintext mode.
EDIT_SECRET = os.environ.get("EDIT_SECRET", "")

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()

def seal_envelope(obj: dict) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt = os.urandom(16); iv = os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", EDIT_SECRET.encode(), salt, 200_000, dklen=32)
    ct = AESGCM(key).encrypt(iv, json.dumps(obj).encode(), None)
    return {"v": 1, "salt": _b64e(salt), "iv": _b64e(iv), "ct": _b64e(ct)}

def open_envelope(enc: dict) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not EDIT_SECRET:
        raise ValueError("no secret configured")
    salt = base64.b64decode(enc["salt"]); iv = base64.b64decode(enc["iv"])
    ct = base64.b64decode(enc["ct"])
    key = hashlib.pbkdf2_hmac("sha256", EDIT_SECRET.encode(), salt, 200_000, dklen=32)
    pt = AESGCM(key).decrypt(iv, ct, None)
    return json.loads(pt.decode())

class VideoReq(BaseModel):
    prompt: str
    images: Optional[List[str]] = None  # dataURL or raw base64; FIRST frame is the anchor
    negative_prompt: str = " "
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    num_frames: int = NUM_FRAMES
    seed: Optional[int] = None
    key: str = ""  # app-level API key when sent inside a sealed body

def decode_img(s: str) -> Image.Image:
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")

def encode_mp4(frames) -> str:
    # imageio-ffmpeg bundles a static ffmpeg: no system packages needed.
    import imageio.v2 as imageio
    import numpy as np
    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=VIDEO_FPS,
                                codec="libx264", quality=8)
    for f in frames:
        writer.append_data(np.asarray(f.convert("RGB")))
    writer.close()
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()

@app.get("/health")
def health():
    # ALWAYS 200; the BODY carries the truth (loading/ready/error).
    if pipe is None:
        body = {"status": "error" if loading_error else "loading"}
        if loading_error:
            body["error"] = loading_error
        return body
    return {"status": "ready", "model": MODEL_ID,
            "encoder": encoder_note,
            "lora": LORA_REPO or None,
            "offload": OFFLOAD,
            "sealed": bool(EDIT_SECRET)}

@app.get("/status")
def status():
    return {"ok": True, "loaded": pipe is not None, "model": MODEL_ID,
            "encoder": encoder_note,
            "lora": LORA_REPO or None,
            "offload": OFFLOAD,
            "sealed": bool(EDIT_SECRET)}

# Async jobs: a clip takes minutes, far longer than any proxy/gateway will
# hold one HTTP connection open. So /render ENQUEUES and returns a job id
# instantly; the app polls /job/{id} with tiny fast requests until the
# video is ready. One worker (the pipe is not thread-safe).
_jobs = {}
_jobs_lock = threading.Lock()
_pipe_lock = threading.Lock()

def _run_job(job_id, kwargs):
    import torch  # function-level: torch itself loads lazily at boot
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["state"] = "running"
    try:
        with _pipe_lock:
            with torch.inference_mode():
                out = get_pipe()(**kwargs).frames[0]
        vid = encode_mp4(out)
        if EDIT_SECRET:
            vid = {"enc": seal_envelope({"video": vid})}
        with _jobs_lock:
            _jobs[job_id].update(state="done", video=vid)
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(state="error", error=str(e)[:300])
    with _jobs_lock:  # cap memory: forget old jobs
        while len(_jobs) > 20:
            _jobs.pop(next(iter(_jobs)))

@app.post("/render")
async def render(request: Request, x_api_key: str = Header(default="")):
    import torch  # already imported by get_pipe; cheap no-op afterwards
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "bad json")
    # Sealed clients send {"enc": {...}} (keys inside the ciphertext).
    # Plaintext still works exactly as before (header key).
    if isinstance(body, dict) and isinstance(body.get("enc"), dict):
        if not EDIT_SECRET:
            raise HTTPException(400, "sealed body but no server secret")
        try:
            body = open_envelope(body["enc"])
        except Exception:
            # Deliberately vague: wrong secret vs corrupt body look identical.
            raise HTTPException(403, "bad secret (decrypt failed)")
    try:
        req = VideoReq(**body)
    except Exception:
        raise HTTPException(400, "bad request shape")
    k = (req.key or x_api_key or "").strip()
    if API_KEY and k != API_KEY:
        raise HTTPException(401, f"bad api key (want {len(API_KEY)}, have {len(k)})")
    if not req.images:
        raise HTTPException(400, "at least one input image is required (first frame)")
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j["state"] in ("queued", "running"))
        if active >= 2:
            raise HTTPException(429, "busy, try again shortly")
    imgs = [decode_img(s) for s in req.images]
    gen = torch.Generator(device="cpu").manual_seed(req.seed) if req.seed is not None else None
    kwargs = dict(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=max(4, min(50, req.num_inference_steps)),
        guidance_scale=req.guidance_scale,
        num_images_per_prompt=1,
    )
    if gen is not None:
        kwargs["generator"] = gen
    kwargs["image"] = imgs[0]
    # Clamp frames server-side: the app gets no knob that can OOM the box.
    kwargs["num_frames"] = max(9, min(121, req.num_frames or NUM_FRAMES))
    kwargs["height"] = VIDEO_H
    kwargs["width"] = VIDEO_W
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"state": "queued", "queued": time.time()}
    threading.Thread(target=_run_job, args=(job_id, kwargs), daemon=True).start()
    return {"job": job_id}

@app.get("/job/{job_id}")
def job(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job (server may have restarted)")
        return dict(j)
