import sys
import os
import oss2
import base64
import io
import torch
import soundfile as sf
import runpod
from indextts.infer_v2 import IndexTTS2

# 将 repo 路径加入 python 路径，以便能 import 里面的模块
sys.path.append('/app/repo')

# --- ⚠️ 这里需要根据 index-tts 的实际代码修改 ---
# 假设 repo 里有一个 infer 模块或者 class
# from inference import IndexTTS 
# 或者类似于:
# from models import SynthesizerTrn
# from text import text_to_sequence
# -----------------------------------------------

def down_file(url, save_folder, filename=None, chunk_size=1024*4):
    """
    下载文件(流式)
    """
    # 提取文件名
    if filename == None:
        filename = urllib.parse.unquote(
            urllib.parse.urlparse(url).path.split("/")[-1])

    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # 下载文件
    save_filepath = os.path.join(save_folder, filename)
    with requests.get(url, stream=True) as req:
        raw_file_size = int(req.headers['Content-Length'])

        with open(save_filepath, 'wb') as f:
            for chunk in req.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    file_len = os.path.getsize(save_filepath)
    if file_len != raw_file_size:
        raise requests.exceptions.ConnectionError('下载文件连接中断，文件不完整')
    return save_filepath
    
def upload_to_aliyun(local_file_path, oss_file_path):
    """
    上传文件到阿里云OSS
    :param local_file_path: 本地文件路径
    :param job_id: job的ID，用于生成OSS目标文件路径
    :return: 上传成功返回文件的完整OSS路径，失败返回None
    """
    try:
        # 获取文件后缀
        # 生成OSS目标文件路径
        start_time = time.time()  # 记录开始时间
        # 上传文件
        result = bucket.put_object_from_file(oss_file_path, local_file_path)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        if result.status == 200:
            print(f"成功上传到阿里云，耗时{elapsed_time}")
            # 返回https的完整OSS路径
            return f"https://{ALIYUN_BUCKET_NAME}.{ALIYUN_ENDPOINT}/{oss_file_path}"
        else:
            return None
    except Exception as e:
        print(f"上传文件到阿里云OSS失败: {e}")
        raise Exception(f"上传文件到阿里云OSS失败: {e}")
        
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
    mp3_url = job_input["mp3_url"]
    mp3_path = down_file(mp3_url, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))    
    prompt_value=job_input["prompt_value"]
    
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
