# NInfer V100 (sm_70) runtime image.
#
# Prerequisite: put the prebuilt server binary at ./bin/ninfer-serve before
# building (download it from the GitHub Releases page of this repository and
# unpack the tarball). The binary is compiled for compute capability 7.0
# (Tesla V100) only and links CUDA 12.8 user-space libraries.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 \
        libavformat60 \
        libavcodec60 \
        libavutil58 \
        libswscale7 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY bin/ninfer-serve /opt/ninfer/ninfer-serve
COPY entrypoint.sh /opt/ninfer/entrypoint.sh
RUN chmod +x /opt/ninfer/ninfer-serve /opt/ninfer/entrypoint.sh

WORKDIR /opt/ninfer
ENTRYPOINT ["/opt/ninfer/entrypoint.sh"]
