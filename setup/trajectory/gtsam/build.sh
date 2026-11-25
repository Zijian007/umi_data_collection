#!/usr/bin/env bash
# Build against the shared library in the pinned PyPI wheel. GTSAM itself is not rebuilt.
set -euo pipefail
deps=$(cd "${1:?dependency directory required}" && pwd)
env_dir="$deps/env"
python="$env_dir/bin/python"
src="$deps/src/gtsam-4.3.0"
build="$deps/build/gtsam-factor-headers"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
site=$("$python" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
suffix=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
version=$("$python" -c 'from importlib.metadata import version, packages_distributions; print(next((version(d) for d in packages_distributions().get("gtsam", ())), ""))')
shopt -s nullglob
libraries=("$site"/gtsam.libs/libgtsam-*.so.4.3.0)
if [[ $(uname -m) != x86_64 || $version != 4.3.0 || ${#libraries[@]} != 1 ]]; then
    echo "当前 GTSAM 环境使用 Python 视觉因子。"
    exit 0
fi
target="$site/_umi_gtsam_factors$suffix"
if [[ $target -nt $here/build.sh && $target -nt $here/natural_factor.cpp &&
      $target -nt $here/poster_factors.cpp && $target -nt $here/kb4.h && $target -nt ${libraries[0]} ]] &&
    "$python" -c 'import gtsam; from _umi_gtsam_factors import NaturalTrackFactor, PosterCornerFactor, PosterDepthFactor'; then
    exit 0
fi
mkdir -p "$deps/src" "$build"
if [[ ! -d $src ]]; then
    git clone --depth 1 --branch 4.3.0 https://github.com/borglab/gtsam.git "$src"
fi
if [[ $(git -C "$src" rev-parse HEAD) != 71a25ca36c084cbad1f872e812d6d97fbadfdb05 ]]; then
    echo "GTSAM 头文件版本与 4.3.0 wheel 不符：$src" >&2
    exit 1
fi
"$env_dir/bin/cmake" -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$env_dir" -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
    -DGTSAM_ALLOW_DEPRECATED_SINCE_V43=OFF -DGTSAM_BUILD_PYTHON=OFF
include=$("$python" -c 'import sysconfig; print(sysconfig.get_path("include"))')
temporary=$(mktemp -d "$site/.umi-gtsam-XXXXXX")
trap 'rm -rf -- "$temporary"' EXIT
"$env_dir/bin/g++" -std=c++17 -O3 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -shared -fvisibility=hidden \
    -I"$src/wrap/pybind11/include" -I"$include" -I"$env_dir/include" -I"$src" -I"$build" \
    -I"$src/gtsam/3rdparty/Eigen" "$here/natural_factor.cpp" "$here/poster_factors.cpp" "${libraries[0]}" \
    -Wl,-rpath,'$ORIGIN/gtsam.libs' -o "$temporary/_umi_gtsam_factors$suffix"
PYTHONPATH="$temporary" "$python" -c 'import gtsam; from _umi_gtsam_factors import NaturalTrackFactor, PosterCornerFactor, PosterDepthFactor'
mv "$temporary/_umi_gtsam_factors$suffix" "$target"
echo "已安装 KB4 自然特征和海报因子：$target"
