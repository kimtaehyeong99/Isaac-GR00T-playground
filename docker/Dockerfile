FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# ─── System dependencies (same as upstream) ───
RUN apt-get update && apt-get install -y \
    build-essential yasm cmake libtool git pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo tmux ffmpeg libegl1

# ─── Python base setup ───
RUN pip install --upgrade pip setuptools && pip install uv
RUN sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt && \
    sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt

# ─── Python dependencies (upstream) ───
RUN pip install imageio h5py boto3 transformers[torch] deepspeed timm peft \
    diffusers wandb tianshou dm_tree openai albumentations==1.4.18 \
    decord==0.6.0 torchcodec==0.4.0

# ─── pytorch3d (upstream) ───
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd /opt/pytorch3d && pip install .

# ─── Additional dependencies (baked in, no runtime pip install needed) ───
RUN pip install --no-deps \
    tyro==0.9.17 \
    omegaconf==2.3.0 \
    lmdb==1.7.5 \
    msgpack-numpy==0.4.8 \
    termcolor==3.2.0 \
    gymnasium==1.2.2 \
    av==15.0.0
RUN pip install docstring-parser shtab "antlr4-python3-runtime==4.9.3"
RUN pip install "transformers==4.51.3"
RUN pip install mpi4py

# ─── 1. Copy upstream source (submodule, unmodified) ───
COPY Isaac-GR00T/ /workspace/gr00t/

# ─── 2. Copy overlay files (new files not in upstream) ───
COPY custom/overlay/ /workspace/gr00t/

# ─── 3. Apply patches (modifications to upstream files) ───
COPY custom/patches/ /tmp/patches/
RUN cd /workspace/gr00t && \
    for p in /tmp/patches/*.patch; do \
        echo "Applying patch: $p" && \
        git apply "$p" || { echo "git apply failed, trying patch -p1..." && patch -p1 < "$p"; } \
    done && \
    rm -rf /tmp/patches

# ─── 4. Install gr00t package ───
RUN cd /workspace/gr00t && uv sync && uv pip install -e .

# ─── EGL/Vulkan setup (same as upstream) ───
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF

WORKDIR /workspace/gr00t
