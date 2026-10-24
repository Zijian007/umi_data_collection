#!/usr/bin/env bash
# build_orbslam3.sh <deps_dir>
#
# 编译 X3 轨迹工具用的 ORB-SLAM3（由 install.sh 在已激活的 conda 环境里调用，也可以单独重跑）：
#   1. Pangolin v0.6（ORB-SLAM3 链接需要；运行时不开窗口）  -> <deps>/pangolin
#   2. cheng-chi/ORB_SLAM3（UMI 用的 fork）@ b741dca3          -> <deps>/src/ORB_SLAM3
#   源码先从组织下的 fork 拉（Torbjorn-s-Apple-Americano/Pangolin、/ORB_SLAM3，2026-10-07 建，防止原仓库被删），
#   拉不到再退回原作者仓库；两种来源都按固定的提交号校验，所以编出来的是同一份代码。
#      + opencv47_aruco_api.patch（OpenCV >= 4.7 的 ArUco 接口）
#      + orbslam3_map_select.patch（环境变量 ORB_MAP_INDEX 选定位用的地图）
#      + x3_mono_slam.cc（纯视觉单目驱动，逐帧写位姿）
#   产物：<deps>/src/ORB_SLAM3/lib/libORB_SLAM3.so、Examples/Monocular-Inertial/x3_mono_slam、Vocabulary/ORBvoc.txt
#
# 环境变量：JOBS（并行编译数，默认 nproc）、ORB3_MARCH（默认 native = 按本机 CPU；
#   要在别的 CPU 上复用同一份编译结果时设成 x86-64-v2 之类）。
# 编译器、OpenCV、Eigen、Boost、OpenSSL、OpenGL 全部来自 conda 环境（$CONDA_PREFIX），不碰系统。
set -euo pipefail
DEPS=$(mkdir -p "${1:?deps dir}" && cd "$1" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV=${CONDA_PREFIX:?run inside the activated conda env (install.sh does this)}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
MARCH=${ORB3_MARCH:-native}
SRC=$DEPS/src
mkdir -p "$SRC"

PANGOLIN_TAG=v0.6
PANGOLIN_COMMIT=dd801d244db3a8e27b7fe8020cd751404aa818fd
PANGOLIN_URLS=(https://github.com/Torbjorn-s-Apple-Americano/Pangolin.git https://github.com/stevenlovegrove/Pangolin.git)
ORB3_COMMIT=b741dca39015330ef4bcc3a85f89493503ade04b
ORB3_URLS=(https://github.com/Torbjorn-s-Apple-Americano/ORB_SLAM3.git https://github.com/cheng-chi/ORB_SLAM3.git)

clone_first() {  # <dest> <git clone args...> -- <url...>: try each url in turn
  local dest=$1; shift; local args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  for u in "$@"; do
    rm -rf "$dest"
    if git -c advice.detachedHead=false clone -q "${args[@]}" "$u" "$dest"; then echo "  source: $u"; return 0; fi
    echo "  clone failed: $u -- trying the next source"
  done
  echo "could not clone from any source: $*"; return 1
}

# Compiler flags: everything (Pangolin, DBoW2, g2o, ORB-SLAM3, x3_mono_slam) is compiled as C++14
# (system gcc 9, whose default is C++14). GCC >= 11 and conda's activation flags default to C++17; mixing C++17 parts
# (DBoW2 / g2o / Pangolin) with the C++14 ORB-SLAM3 library breaks Eigen's aligned allocation across the boundary and the
# mapper dies with "double free or corruption" right after map initialisation. conda's other defaults (-march=nocona,
# -O2, _FORTIFY_SOURCE, ...) are dropped as well; only the env's header / library paths are kept (the fork adds -O3 -march=native).
export CFLAGS="-isystem $ENV/include"
export CXXFLAGS="-std=c++14 -isystem $ENV/include"
export CPPFLAGS=""
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib -Wl,-rpath-link,$ENV/lib"
CMAKE_COMMON=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ENV" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX")

# ---------- 1. Pangolin ----------
if [ ! -f "$DEPS/pangolin/lib/cmake/Pangolin/PangolinConfig.cmake" ]; then
  echo "[pangolin] clone $PANGOLIN_TAG"
  clone_first "$SRC/Pangolin" --depth 1 --branch "$PANGOLIN_TAG" -- "${PANGOLIN_URLS[@]}"
  got=$(git -C "$SRC/Pangolin" rev-parse HEAD)
  [ "$got" = "$PANGOLIN_COMMIT" ] || { echo "Pangolin $PANGOLIN_TAG is $got, expected $PANGOLIN_COMMIT"; exit 1; }
  # GCC >= 11 不再间接包含 <limits>；v0.6 的 colour.h 用到了 std::numeric_limits
  grep -q '#include <limits>' "$SRC/Pangolin/include/pangolin/gl/colour.h" || \
    sed -i 's|#include <stdexcept>|#include <stdexcept>\n#include <limits>|' "$SRC/Pangolin/include/pangolin/gl/colour.h"
  # 不要视频/图像格式/Python/示例，只要 GL 显示部分
  cmake -S "$SRC/Pangolin" -B "$SRC/Pangolin/build" "${CMAKE_COMMON[@]}" -DCMAKE_INSTALL_PREFIX="$DEPS/pangolin" \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PYPANGOLIN_MODULE=OFF \
    -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBPNG=OFF -DBUILD_PANGOLIN_LIBJPEG=OFF -DBUILD_PANGOLIN_LIBTIFF=OFF \
    -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_ZSTD=OFF -DBUILD_PANGOLIN_LZ4=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF \
    -DBUILD_PANGOLIN_LIBUVC=OFF -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_LIBREALSENSE2=OFF -DBUILD_PANGOLIN_OPENNI=OFF \
    -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_TELICAM=OFF -DBUILD_PANGOLIN_PLEORA=OFF \
    -DBUILD_PANGOLIN_TOON=OFF -DBUILD_PANGOLIN_V4L=OFF -DBUILD_PANGOLIN_VIDEO=OFF
  cmake --build "$SRC/Pangolin/build" -j "$JOBS"
  cmake --install "$SRC/Pangolin/build"
else
  echo "[pangolin] already built"
fi

# ---------- 2. ORB-SLAM3 (UMI fork) ----------
O=$SRC/ORB_SLAM3
if [ ! -f "$O/.umi_patched" ]; then
  echo "[orbslam3] clone $ORB3_COMMIT"
  clone_first "$O" -- "${ORB3_URLS[@]}"
  git -C "$O" -c advice.detachedHead=false checkout -q "$ORB3_COMMIT"
  # git apply (git comes with the conda env) instead of patch(1): a bare ubuntu:22.04 image has no `patch`
  git -C "$O" apply "$HERE/opencv47_aruco_api.patch"
  git -C "$O" apply "$HERE/orbslam3_map_select.patch"
  mkdir -p "$O/Examples/Monocular"
  cp "$HERE/x3_mono_slam.cc" "$O/Examples/Monocular/x3_mono_slam.cc"
  # 可执行文件落在 Examples/Monocular-Inertial（fork 的 CMakeLists 末尾的 RUNTIME 目录）
  printf '\nadd_executable(x3_mono_slam\nExamples/Monocular/x3_mono_slam.cc)\ntarget_link_libraries(x3_mono_slam ${PROJECT_NAME})\n' >> "$O/CMakeLists.txt"
  if [ "$MARCH" != native ]; then
    sed -i "s/-march=native/-march=$MARCH/g" "$O/CMakeLists.txt" "$O/Thirdparty/DBoW2/CMakeLists.txt" "$O/Thirdparty/g2o/CMakeLists.txt"
  fi
  touch "$O/.umi_patched"
fi

if [ ! -f "$O/Thirdparty/DBoW2/lib/libDBoW2.so" ]; then
  echo "[orbslam3] DBoW2"
  cmake -S "$O/Thirdparty/DBoW2" -B "$O/Thirdparty/DBoW2/build" "${CMAKE_COMMON[@]}" -DOpenCV_DIR="$ENV/lib/cmake/opencv4"
  cmake --build "$O/Thirdparty/DBoW2/build" -j "$JOBS"
fi
if [ ! -f "$O/Thirdparty/g2o/lib/libg2o.so" ]; then
  echo "[orbslam3] g2o"
  cmake -S "$O/Thirdparty/g2o" -B "$O/Thirdparty/g2o/build" "${CMAKE_COMMON[@]}" -DEigen3_DIR="$ENV/share/eigen3/cmake"
  cmake --build "$O/Thirdparty/g2o/build" -j "$JOBS"
fi
echo "[orbslam3] library + x3_mono_slam"
cmake -S "$O" -B "$O/build" "${CMAKE_COMMON[@]}" -DOpenCV_DIR="$ENV/lib/cmake/opencv4" \
  -DEigen3_DIR="$ENV/share/eigen3/cmake" -DPangolin_DIR="$DEPS/pangolin/lib/cmake/Pangolin"
cmake --build "$O/build" -j "$JOBS" --target ORB_SLAM3 x3_mono_slam

if [ ! -f "$O/Vocabulary/ORBvoc.txt" ]; then
  echo "[orbslam3] vocabulary"
  tar -xzf "$O/Vocabulary/ORBvoc.txt.tar.gz" -C "$O/Vocabulary"
fi
sha256sum "$O/lib/libORB_SLAM3.so" "$O/Examples/Monocular-Inertial/x3_mono_slam" "$O/Vocabulary/ORBvoc.txt" | sed 's|  .*/|  |'
echo BUILD_ORBSLAM3_DONE
