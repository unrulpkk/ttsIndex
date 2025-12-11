import sys
import os
import base64
import io
import torch
import soundfile as sf
import runpod

# 将 repo 路径加入 python 路径，以便能 import 里面的模块
sys.path.append('/app/repo')

# --- ⚠️ 这里需要根据 index-tts 的实际代码修改 ---
# 假设 repo 里有一个 infer 模块或者 class
# from inference import IndexTTS 
# 或者类似于:
# from models import SynthesizerTrn
# from text import text_to_sequence
# -----------------------------------------------

# 模拟的全局变量，用于缓存模型
models = {
    "1.5": None,
    "2.0": None
}

def load_model(version):
    """
    加载指定版本的模型到 GPU
    """
    if models.get(version) is not None:
        return models[version]
    
    print(f"Loading model version {version}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- ⚠️ 实际加载逻辑 ---
    # 示例伪代码：
    # config_path = f"/app/repo/checkpoints/{version}/config.json"
    # model_path = f"/app/repo/checkpoints/{version}/model.pth"
    # net = IndexTTS(config_path, model_path, device=device)
    # -----------------------
    
    # 这里放一个占位符，请替换为真实加载代码
    net = "PLACEHOLDER_MODEL_OBJECT" 
    
    models[version] = net
    print(f"Model {version} loaded.")
    return net

def run_inference(model, text, speaker_id, speed):
    """
    执行推理
    """
    # --- ⚠️ 实际推理逻辑 ---
    # audio_numpy = model.infer(text, speaker=speaker_id, speed=speed)
    # sampling_rate = model.sampling_rate
    # -----------------------
    
    # 伪造一个静音音频用于测试 (请删除)
    import numpy as np
    sampling_rate = 24000
    audio_numpy = np.zeros(int(sampling_rate * 2)) 
    
    return audio_numpy, sampling_rate

async def handler(job):
    """
    RunPod 处理函数
    """
    job_input = job["input"]
    
    # 1. 解析参数
    text = job_input.get("text")
    version = str(job_input.get("version", "2.0")) # 默认 2.0
    speaker = job_input.get("speaker", 0)
    speed = job_input.get("speed", 1.0)

    if not text:
        return {"error": "Missing 'text' parameter"}

    if version not in ["1.5", "2.0"]:
        return {"error": "Version must be '1.5' or '2.0'"}

    try:
        # 2. 获取/加载模型 (懒加载策略，或者在 handler 外面预加载)
        model = load_model(version)
        
        # 3. 推理
        audio_data, sr = run_inference(model, text, speaker, speed)
        
        # 4. 将音频转换为 Base64 返回
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)
        base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "audio_base64": base64_audio,
            "sampling_rate": sr,
            "version": version
        }

    except Exception as e:
        return {"error": str(e)}

# 启动 Serverless 监听
runpod.serverless.start({"handler": handler})
