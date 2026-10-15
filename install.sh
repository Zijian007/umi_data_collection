#!/usr/bin/env bash
# install.sh -- 一次性安装 X3 轨迹工具（umi-traj）的全部依赖。
#
#   bash install.sh                        # 装到 ./.deps（约 5 GB，首次 15-40 分钟，取决于网速和 CPU）
#   UMI_DEPS=/some/dir bash install.sh     # 装到别处
#   JOBS=8 bash install.sh                 # 限制编译并行数（默认 = CPU 核数）
#
# 不需要 sudo，不改系统：conda 包管理器（micromamba）、Python 3.11 + numpy/scipy/gtsam、FFmpeg、OpenCV、编译器，
# 以及从源码编译的 Pangolin + ORB-SLAM3，全部放在 <deps> 一个目录里。删掉这个目录就是卸载。
# 需要：Linux x86_64（aarch64 能装但未实测）、bash、curl、tar、能访问 github.com / conda.anaconda.org / pypi.org。
# 中途失败可以直接重跑，已完成的步骤会跳过。每一步的完整日志在 <deps>/logs/。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=$ROOT/software/trajectory
DEPS=${UMI_DEPS:-$ROOT/.deps}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
MAMBA_VERSION=2.9.0-0
mkdir -p "$DEPS/logs" "$DEPS/bin"
DEPS=$(cd "$DEPS" && pwd)
ENV=$DEPS/env
export MAMBA_ROOT_PREFIX=$DEPS/mamba
MM=$DEPS/bin/micromamba

step() { echo; echo "==== $* ===="; }
fail() { echo; echo "安装失败：$1"; [ -n "${2:-}" ] && [ -f "$2" ] && { echo "---- $2 的最后 30 行 ----"; tail -30 "$2"; }; exit 1; }
run_logged() {  # <log> <cmd...>
  local log=$1; shift
  if ! "$@" > "$log" 2>&1; then fail "命令失败：$*" "$log"; fi
}

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  PLAT=linux-64;      MAMBA_SHA=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3;;
  Linux-aarch64) PLAT=linux-aarch64; MAMBA_SHA=9f93b974adcb4d166996af969b6cd371287d1a3e52733704727884d9b74cb7a7;;
  *) fail "只支持 Linux（x86_64 / aarch64）。Windows / macOS 请用 Docker（见 README）。";;
esac
for c in curl tar; do command -v $c > /dev/null || fail "缺少系统命令 $c（Ubuntu: sudo apt install $c）"; done

echo "安装到 $DEPS（平台 $PLAT，编译并行 $JOBS）"

step "1/5 micromamba $MAMBA_VERSION"
if [ ! -x "$MM" ] || ! echo "$MAMBA_SHA  $MM" | sha256sum -c --quiet - 2>/dev/null; then
  curl -fsSL --retry 3 -o "$MM.part" "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-$PLAT" \
    || fail "下载 micromamba 失败（需要能访问 github.com）"
  echo "$MAMBA_SHA  $MM.part" | sha256sum -c --quiet - || fail "micromamba 校验和不对（下载损坏或被替换）"
  mv "$MM.part" "$MM"; chmod +x "$MM"
fi
"$MM" --version

step "2/5 conda 环境（Python、FFmpeg、OpenCV、编译器等）"
if [ ! -f "$ENV/.umi_env_ok" ]; then
  rm -rf "$ENV"
  if [ "$PLAT" = linux-64 ]; then
    run_logged "$DEPS/logs/2_conda.log" "$MM" create -y -p "$ENV" --file "$T/env/linux-64.lock"
  else
    run_logged "$DEPS/logs/2_conda.log" "$MM" create -y -p "$ENV" -f "$T/env/environment.yml"
  fi
  touch "$ENV/.umi_env_ok"
fi
"$ENV/bin/python" --version

step "3/5 pip：GTSAM"
if ! "$ENV/bin/python" -c "import gtsam" 2>/dev/null; then
  run_logged "$DEPS/logs/3_pip.log" "$ENV/bin/python" -m pip install --no-cache-dir -r "$T/env/requirements.txt"
fi
"$ENV/bin/python" -c "import gtsam, numpy, scipy, av, matplotlib; print('gtsam ok, numpy', numpy.__version__)"

step "4/5 编译 Pangolin + ORB-SLAM3（10-30 分钟）"
O=$DEPS/src/ORB_SLAM3
if [ ! -x "$O/Examples/Monocular-Inertial/x3_mono_slam" ] || [ ! -f "$O/Vocabulary/ORBvoc.txt" ]; then
  echo "日志：$DEPS/logs/4_build.log"
  JOBS=$JOBS run_logged "$DEPS/logs/4_build.log" "$MM" run -p "$ENV" bash "$T/orbslam3/build_orbslam3.sh" "$DEPS"
fi
tail -4 "$DEPS/logs/4_build.log" 2>/dev/null || true

step "5/5 写环境文件并自检"
cat > "$DEPS/env.sh" <<EOF
# 由 install.sh 生成；umi-traj 会自动 source。
export UMI_DEPS="$DEPS"
export PATH="$ENV/bin:\$PATH"
export KIWI_PY="$ENV/bin/python"
export ORB3_DIR="$O"
export ORB3_VOCAB="$O/Vocabulary/ORBvoc.txt"
export ORB3_EXTRA_LD="$ENV/lib:$DEPS/pangolin/lib"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS:-4}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-\${TMPDIR:-/tmp}/umi-traj-mpl}"
EOF
{
  echo "installed $(date -u +%FT%TZ)"
  echo "commit $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${UMI_COMMIT:-unknown}")"
  echo "platform $PLAT micromamba $MAMBA_VERSION"
  sha256sum "$O/lib/libORB_SLAM3.so" "$O/Examples/Monocular-Inertial/x3_mono_slam" | sed 's|  .*/|  |'
} > "$DEPS/BUILD_INFO.txt"
bash "$ROOT/umi-traj" check || fail "自检没通过（见上面 FAIL 的那一行）"
echo
echo "安装完成。用法："
echo "  $ROOT/umi-traj <录像.insv 或 放录像的文件夹>"
