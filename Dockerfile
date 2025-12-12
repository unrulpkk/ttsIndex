# 使用 RunPod 官方更先进的镜像: PyTorch 2.4, Python 3.11, CUDA 12.4.1
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# 1. 系统依赖 (IndexTTS 需要 espeak-ng, 编译 pynini 需要 openfst 等)
# 增加 git-lfs 以防下载大文件需要
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    espeak-ng \
    cmake \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

# 2. 克隆仓库
RUN git clone https://github.com/index-tts/index-tts.git /app/repo

# 3. 安装依赖
WORKDIR /app/repo

# 这一步非常关键：安装仓库自带依赖
# IndexTTS 经常需要 pynini，这在 Linux 下 pip 安装通常没问题，但需要 build-essential (上面已装)
RUN pip install --no-cache-dir -r requirements.txt

# 4. 安装 RunPod SDK 和其他必须库
RUN pip install --no-cache-dir runpod scipy librosa soundfile

# 5. 下载模型 (执行下载脚本)
WORKDIR /app
#RUN mkdir -p /app/repo/checkpoints/1.5
#RUN mkdir -p /app/repo/checkpoints/2.0
#RUN git clone https://huggingface.co/IndexTeam/IndexTTS-2 /app/repo/checkpoints/2.0
#RUN huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir /app/repo/checkpoints/1.5--local-dir-use-symlinks False
#RUN huggingface-cli download IndexTeam/IndexTTS-2 --local-dir /app/repo/checkpoints/2.0--local-dir-use-symlinks False
# 6. 复制处理脚本
COPY handler.py /app/handler.py

# 7. 环境变量设置 (防止 Python 输出缓冲)
ENV PYTHONUNBUFFERED=1
# 增加 PYTHONPATH 确保能导入 repo 里的模块
ENV PYTHONPATH=/app/repo

CMD [ "python", "-u", "/app/handler.py" ]
