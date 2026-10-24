#!/usr/bin/env bash
# install.sh -- 安装 umi-calib（重新标定相机）需要的额外组件。约 1-3 分钟，占 ~400 MB。
#
#   bash software/calibration/setup/install.sh
#
# 先要装好 umi-traj（仓库根目录 bash install.sh）。不需要 sudo，全部放进同一个 <deps> 目录，重跑安全。
#   1) 解压仓库自带的 Basalt 标定程序：官方 0.1.7 + 本仓库的补丁（固定 IMU 尺度、第二阶段只优化相机时间差），
#      预编译 x86_64，自带动态加载器和全部运行库，不依赖系统 glibc 版本（来源见 basalt/PROVENANCE.md）。
#   2) 建 <deps>/calib_env：叠在 umi-traj 的 Python 环境上的 venv，另装 OpenCV（contrib）和 rosbags。
#      需要能访问 pypi.org（或自行设置 PIP_INDEX_URL 镜像）。
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
DEPS=${UMI_DEPS:-$ROOT/.deps}
fail() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$DEPS/env.sh" ] || fail "还没装 umi-traj：先运行  bash $ROOT/install.sh"
# shellcheck disable=SC1091
source "$DEPS/env.sh"
[ "$(uname -m)" = x86_64 ] || fail "自带的 Basalt 是 x86_64 Linux 版，当前机器是 $(uname -m)"
mkdir -p "$DEPS/logs"

TARBALL=$HERE/basalt/umi-basalt-0.1.7-fsoffset-linux-x86_64.tar.xz
TARBALL_SHA=c531b065fde3d51b4a188168eb3867cef3ec17338ae11f9397d0a5236c5aa4c2
B=$DEPS/basalt/umi-basalt-0.1.7-fsoffset-linux-x86_64
echo "==== 1/3 Basalt 标定程序 ===="
if [ ! -x "$B/bin/basalt_calibrate_imu" ]; then
  echo "$TARBALL_SHA  $TARBALL" | sha256sum -c --quiet - || fail "Basalt 压缩包校验和不对（文件损坏；重新 git pull / git checkout 该文件）"
  mkdir -p "$DEPS/basalt" && tar xJf "$TARBALL" -C "$DEPS/basalt"
fi
( cd "$B" && sha256sum -c --quiet SHA256SUMS ) || fail "Basalt 程序校验和不对：删掉 $DEPS/basalt 后重跑"
"$B/lib/ld-linux-x86-64.so.2" --library-path "$B/lib" "$B/bin/basalt_calibrate_imu" --help 2>&1 | grep -q -- "--fixed-scale-opt-offset" \
  || fail "Basalt 程序在这台机器上跑不起来（把 '$B/lib/ld-linux-x86-64.so.2 --library-path $B/lib $B/bin/basalt_calibrate_imu --help' 的输出发给维护者）"
echo "  OK $B"

echo "==== 2/3 Python 环境（OpenCV contrib、rosbags）===="
ENV=$DEPS/calib_env
[ -x "$ENV/bin/python" ] || "$KIWI_PY" -m venv --system-site-packages "$ENV"
if ! "$ENV/bin/python" -c "import cv2, rosbags, lz4; assert hasattr(cv2, 'aruco')" 2>/dev/null; then
  "$ENV/bin/python" -m pip install --no-cache-dir -r "$HERE/requirements.txt" > "$DEPS/logs/calib_pip.log" 2>&1 \
    || fail "pip 安装失败，日志在 $DEPS/logs/calib_pip.log（网络问题可设置 PIP_INDEX_URL 换镜像后重跑）"
fi
echo "  OK $ENV"

echo "==== 3/3 自检 ===="
PYTHONPATH="$ROOT" "$ENV/bin/python" -m software.calibration check
