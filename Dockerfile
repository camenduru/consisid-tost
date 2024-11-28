FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install transformers diffusers accelerate peft opencv-python protobuf sentencepiece optimum-quanto moviepy insightface onnxruntime onnxruntime-gpu facexlib spandrel scikit-video timm ftfy SentencePiece && \
    git clone https://github.com/PKU-YuanGroup/ConsisID /content/ConsisID && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/parsing_parsenet.pth -d /content/ConsisID-preview/face_encoder -o parsing_parsenet.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/parsing_bisenet.pth -d /content/ConsisID-preview/face_encoder -o parsing_bisenet.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/detection_Resnet50_Final.pth -d /content/ConsisID-preview/face_encoder -o detection_Resnet50_Final.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/EVA02_CLIP_L_336_psz14_s6B.pt -d /content/ConsisID-preview/face_encoder -o EVA02_CLIP_L_336_psz14_s6B.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/antelopev2/1k3d68.onnx -d /content/ConsisID-preview/face_encoder/models/antelopev2 -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/antelopev2/2d106det.onnx -d /content/ConsisID-preview/face_encoder/models/antelopev2 -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/antelopev2/genderage.onnx -d /content/ConsisID-preview/face_encoder/models/antelopev2 -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/antelopev2/glintr100.onnx -d /content/ConsisID-preview/face_encoder/models/antelopev2 -o glintr100.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/antelopev2/scrfd_10g_bnkps.onnx -d /content/ConsisID-preview/face_encoder/models/antelopev2 -o scrfd_10g_bnkps.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/buffalo_l/1k3d68.onnx -d /content/ConsisID-preview/face_encoder/models/buffalo_l -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/buffalo_l/2d106det.onnx -d /content/ConsisID-preview/face_encoder/models/buffalo_l -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/buffalo_l/det_10g.onnx -d /content/ConsisID-preview/face_encoder/models/buffalo_l -o det_10g.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/buffalo_l/genderage.onnx -d /content/ConsisID-preview/face_encoder/models/buffalo_l -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/models/buffalo_l/w600k_r50.onnx -d /content/ConsisID-preview/face_encoder/models/buffalo_l -o w600k_r50.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/scheduler/scheduler_config.json -d /content/ConsisID-preview/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/text_encoder/config.json -d /content/ConsisID-preview/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/text_encoder/model-00001-of-00002.safetensors -d /content/ConsisID-preview/text_encoder -o model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/text_encoder/model-00002-of-00002.safetensors -d /content/ConsisID-preview/text_encoder -o model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/text_encoder/model.safetensors.index.json -d /content/ConsisID-preview/text_encoder -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/tokenizer/added_tokens.json -d /content/ConsisID-preview/tokenizer -o added_tokens.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/tokenizer/special_tokens_map.json -d /content/ConsisID-preview/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/tokenizer/spiece.model -d /content/ConsisID-preview/tokenizer -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/tokenizer/tokenizer_config.json -d /content/ConsisID-preview/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/transformer/config.json -d /content/ConsisID-preview/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/transformer/diffusion_pytorch_model-00001-of-00002.safetensors -d /content/ConsisID-preview/transformer -o diffusion_pytorch_model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/transformer/diffusion_pytorch_model-00002-of-00002.safetensors -d /content/ConsisID-preview/transformer -o diffusion_pytorch_model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/transformer/diffusion_pytorch_model.safetensors.index.json -d /content/ConsisID-preview/transformer -o diffusion_pytorch_model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/vae/config.json -d /content/ConsisID-preview/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/ConsisID-preview/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/configuration.json -d /content/ConsisID-preview -o configuration.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/BestWishYsh/ConsisID-preview/raw/main/model_index.json -d /content/ConsisID-preview -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth -d /content/model_real_esran -o RealESRGAN_x4.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/AlexWortega/RIFE/resolve/main/flownet.pkl -d /content/model_rife -o flownet.pkl

COPY ./worker_runpod.py /content/ConsisID/worker_runpod.py
WORKDIR /content/ConsisID
CMD python worker_runpod.py