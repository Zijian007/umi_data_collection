# X3 轨迹工具（umi-traj）的容器版：与本机安装用同一个 install.sh、同一份 conda lock，结果一致。
#
#   docker build -t umi-traj .                                   # 一次，15-40 分钟
#   docker run --rm -v /path/to/videos:/data umi-traj /data      # 处理 /path/to/videos 里的全部 .insv
#     结果在 /path/to/videos/umi_traj_out/<录像名>/
#   Linux 上加 --user "$(id -u):$(id -g)"，结果文件就归你自己而不是 root。
#
# 编译默认 -march=native（与验证用的 lab0 一致）：镜像在哪台机器 build，就在那台（或同代 CPU）上跑。
# 要把镜像拷到别的机器用，build 时加 --build-arg ORB3_MARCH=x86-64-v2。
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tar bzip2 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/umi
COPY . /opt/umi
ARG JOBS=8
ARG ORB3_MARCH=native
ARG UMI_COMMIT=unknown
ENV UMI_DEPS=/opt/umi-deps UMI_COMMIT=${UMI_COMMIT}
RUN JOBS=${JOBS} ORB3_MARCH=${ORB3_MARCH} bash install.sh \
 && rm -rf /opt/umi-deps/mamba/pkgs /opt/umi-deps/src/Pangolin /opt/umi-deps/src/ORB_SLAM3/build \
 && chmod -R a+rX /opt/umi /opt/umi-deps
ENTRYPOINT ["bash", "/opt/umi/umi-traj"]
CMD ["--help"]
