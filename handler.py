import sys
import os
import oss2
import base64
import io
import torch
import soundfile as sf
import runpod
from index-tts.indextts.infer_v2 import IndexTTS2

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
def handler(job):
    """
    RunPod 处理函数
    """
    job_input = job["input"]
    mp3_url = job_input["mp3_url"]
    mp3_path = down_file(mp3_url, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))    
    prompt_value=job_input["prompt_value"]
    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
    text = "Translate for me, what is a surprise!"
    tts.infer(spk_audio_prompt=mp3_path, text=prompt_value, output_path="gen.wav", verbose=True)


# 启动 Serverless 监听
runpod.serverless.start({"handler": handler})
