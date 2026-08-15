# dd-ltx25-motion SERVERLESS handler (2026-08-15). The i2v pipeline construction and
# generate are COPIED from lib/pipeline/host/ltx-worker/server25.py (the locked 2026-08-09
# i2v config: DISTILLED_SIGMAS 8 steps, cfg 2.5 negative-active, audio neutral, 24fps) —
# provenance matters: if server25's i2v config changes, change it here too.
# Worker contract: input {image_b64, prompt, negative_prompt, width, height, num_frames,
# seed, job_key} -> output {video_b64, cached, gen_s} or {error}. The volume cache at
# $VOL/cache/<job_key>.mp4 makes retries and duplicate submissions FREE and durable across
# workers — better than the pod-local result cache it replaces.
import base64
import glob
import os
import tempfile
import time

VOL = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/workspace"
# DUAL LAYOUT (2026-08-15): once the 2.3 HOST stack joins the volume, the 2.5 motion stack
# moves to $VOL/ltx25/ and the top level belongs to boot.sh's 2.3 world. Prefer the subdir,
# fall back to the original top-level layout so validation order never matters.
BASE = f"{VOL}/ltx25" if os.path.isdir(f"{VOL}/ltx25/models") else VOL
MDL = os.environ.get("LTX_MODELS_DIR", f"{BASE}/models")
CACHE = f"{BASE}/cache"
os.makedirs(CACHE, exist_ok=True)

import torch  # noqa: E402
from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number  # noqa: E402
from ltx_pipelines.utils.model_paths import ModelPaths  # noqa: E402
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMAS  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402
from ltx_pipelines.utils.quantization_factory import QuantizationKind  # noqa: E402

TRANSFORMER = f"{MDL}/transformer.safetensors"
for _p in (TRANSFORMER, f"{MDL}/gemma4-te.safetensors", f"{MDL}/video-vae-conv.safetensors", f"{MDL}/audio-vae.safetensors", f"{MDL}/distilled-lora.safetensors"):
    if not os.path.exists(_p):
        raise RuntimeError(f"weights missing on volume: {_p}")
MP = ModelPaths.from_split(
    transformer_path=TRANSFORMER,
    text_encoder_path=f"{MDL}/gemma4-te.safetensors",
    video_vae_path=f"{MDL}/video-vae-conv.safetensors",
    audio_vae_path=f"{MDL}/audio-vae.safetensors",
)
QUANT = os.environ.get("LTX_QUANT", "fp8-cast").strip()
_qp = QuantizationKind(QUANT).to_policy(checkpoint_path=TRANSFORMER) if QUANT else None

_t0 = time.time()
print(f"[i2v-serverless] building TI2VidOneStagePipeline quant={QUANT or 'bf16'} ...", flush=True)
PIPELINE = TI2VidOneStagePipeline(
    model_paths=MP,
    loras=(LoraPathStrengthAndSDOps(f"{MDL}/distilled-lora.safetensors", 1.0, LTXV_LORA_COMFY_RENAMING_MAP),),
    quantization=_qp,
)
print(f"[i2v-serverless] pipeline constructed in {time.time() - _t0:.1f}s", flush=True)


def _generate_i2v(img_path, prompt, negative, width, height, num_frames, seed):
    with torch.inference_mode():
        video, _audio, tiling = PIPELINE(
            prompt=prompt,
            negative_prompt=negative or DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=24.0,
            num_inference_steps=8,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=2.5, stg_scale=0.0, rescale_scale=0.0,
                modality_scale=1.0, skip_step=0, stg_blocks=[28],
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
                modality_scale=1.0, skip_step=0, stg_blocks=[],
            ),
            images=[ImageConditioningInput(img_path, 0, 1.0)],
            sigmas=DISTILLED_SIGMAS,
            tiling_config=AUTO_TILING,
        )
        out = tempfile.mktemp(suffix=".mp4")
        encode_video(video=video, fps=24.0, audio=None, output_path=out,
                     video_chunks_number=get_video_chunks_number(num_frames, tiling))
        return out


def handler(event):
    inp = event.get("input") or {}
    job_key = str(inp.get("job_key") or "").strip()
    try:
        if job_key:
            cached = os.path.join(CACHE, f"{job_key}.mp4")
            if os.path.isfile(cached) and os.path.getsize(cached) > 0:
                with open(cached, "rb") as f:
                    return {"video_b64": base64.b64encode(f.read()).decode(), "cached": True, "gen_s": 0}
        img_b64 = inp.get("image_b64") or ""
        if not img_b64:
            return {"error": "image_b64 missing"}
        img = tempfile.mktemp(suffix=".png")
        with open(img, "wb") as f:
            f.write(base64.b64decode(img_b64))
        t0 = time.time()
        out = _generate_i2v(
            img,
            str(inp.get("prompt") or ""),
            str(inp.get("negative_prompt") or ""),
            int(inp.get("width") or 1024),
            int(inp.get("height") or 576),
            int(inp.get("num_frames") or 121),
            int(inp.get("seed") or 10),
        )
        gen_s = round(time.time() - t0, 1)
        with open(out, "rb") as f:
            data = f.read()
        if job_key:
            tmp = os.path.join(CACHE, f".{job_key}.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(CACHE, f"{job_key}.mp4"))
        for p in (img, out):
            try:
                os.remove(p)
            except Exception:
                pass
        return {"video_b64": base64.b64encode(data).decode(), "cached": False, "gen_s": gen_s}
    except Exception as e:  # the client fails the beat to its still — never crash the worker loop
        import traceback
        traceback.print_exc()
        return {"error": str(e)[:400]}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
