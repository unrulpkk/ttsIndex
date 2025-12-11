import os
import base64
import tempfile
import traceback
import runpod
from huggingface_hub import snapshot_download

INDEX_TTS_VERSION = os.environ.get("INDEX_TTS_VERSION", "1.5")
MODEL_ID = os.environ.get("MODEL_ID", "IndexTeam/IndexTTS-1.5")
MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
USE_CACHED_MODEL = os.environ.get("USE_CACHED_MODEL", "0")

MODEL = None

def ensure_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    if USE_CACHED_MODEL == "1":
        print("[runpod] Using cached model, expecting checkpoints in", MODEL_DIR)
    else:
        print("[runpod] Downloading model snapshot to", MODEL_DIR, "from", MODEL_ID)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR, token=HF_TOKEN)

    try:
        if str(INDEX_TTS_VERSION).startswith("2"):
            from indextts.infer_v2 import IndexTTS2
            MODEL = IndexTTS2(cfg_path=os.path.join(MODEL_DIR, "config.yaml"),
                              model_dir=MODEL_DIR,
                              use_fp16=True,
                              use_cuda_kernel=True,
                              use_deepspeed=False)
        else:
            from indextts.infer import IndexTTS
            MODEL = IndexTTS(cfg_path=os.path.join(MODEL_DIR, "config.yaml"),
                             model_dir=MODEL_DIR,
                             use_fp16=True,
                             use_cuda_kernel=True,
                             use_deepspeed=False)
    except Exception as e:
        print("[runpod] Failed to import/init model:", e)
        traceback.print_exc()
        raise
    print("[runpod] Model loaded")
    return MODEL

def handler(job):
    try:
        inp = job.get("input", {})
        text = inp.get("text")
        if not text:
            return {"error": "missing 'text' in input"}

        model = ensure_model()

        spk_path = None
        emo_path = None

        def b64_to_file(b64s, suffix=".wav"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(base64.b64decode(b64s))
            tmp.flush()
            tmp.close()
            return tmp.name

        if inp.get("spk_audio_b64"):
            spk_path = b64_to_file(inp["spk_audio_b64"])
        if inp.get("emo_audio_b64"):
            emo_path = b64_to_file(inp["emo_audio_b64"])

        out_path = os.path.join("/tmp", "gen.wav")

        if str(INDEX_TTS_VERSION).startswith("2"):
            model.infer(spk_audio_prompt=spk_path if spk_path else None,
                        text=text,
                        output_path=out_path,
                        emo_audio_prompt=emo_path if emo_path else None,
                        emo_alpha=inp.get("emo_alpha", 1.0),
                        verbose=False)
        else:
            model.infer(spk_audio_prompt=spk_path if spk_path else None,
                        text=text,
                        output_path=out_path,
                        emo_audio_prompt=emo_path if emo_path else None,
                        emo_alpha=inp.get("emo_alpha", 1.0),
                        verbose=False)

        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"gen_wav_b64": b64, "out_path": out_path, "text": text}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def _health(job):
    return {"status": "ok"}

runpod.serverless.start({"handler": handler})
