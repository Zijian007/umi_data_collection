#!/usr/bin/env bash
# 统一测试与质量检查入口
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEPS_DIR="${UMI_DEPS:-$ROOT_DIR/.deps}"
TEST_PY=python3
for candidate in "$DEPS_DIR/calib_env/bin/python" "$DEPS_DIR/env/bin/python"; do
    if [ -x "$candidate" ]; then
        TEST_PY="$candidate"
        break
    fi
done

RUN_CHECKS=0
RUN_UNIT=1
RUN_JS=1
TEST_TARGETS=()

usage() {
    echo "用法: $0 [选项] | $0 --unit [测试模块或用例 ...]"
    echo ""
    echo "选项:"
    echo "  --unit       运行 Python 测试；可指定 test_rear_assets 或 模块.类.方法"
    echo "  --js         仅运行查看器前端测试 (node --test)"
    echo "  --checks     仅运行 CLI 运行时健康检查 (umi-calib check, umi-scene check, KB4 C++ 回归)"
    echo "  --all        运行全部（单元测试 + 前端测试 + CLI 检查 + KB4 C++ 回归）"
    echo "  -h, --help   显示帮助信息"
    echo ""
    echo "示例: $0 --unit test_rear_assets test_calibration_inventory"
}

if [ $# -gt 1 ] && [ "$1" != "--unit" ]; then
    usage >&2
    exit 2
fi

if [ $# -gt 0 ]; then
    case "$1" in
        --unit) RUN_JS=0; shift; TEST_TARGETS=("$@") ;;
        --js) RUN_UNIT=0 ;;
        --checks) RUN_UNIT=0; RUN_JS=0; RUN_CHECKS=1 ;;
        --all) RUN_CHECKS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知选项: $1" >&2; usage >&2; exit 2 ;;
    esac
fi

for test_target in "${TEST_TARGETS[@]}"; do
    if [[ "$test_target" == -* ]]; then
        echo "--unit 后请填写测试模块或用例，不接受其他选项：$test_target" >&2
        exit 2
    fi
done

echo "=================================================="
echo "          UMI Data Collection 统一测试"
echo "=================================================="

# 1. 主单元测试
if [ "$RUN_UNIT" -eq 1 ]; then
    echo ""
    echo "[单元] 运行主单元测试套件 (tests/)..."
    if [ ${#TEST_TARGETS[@]} -gt 0 ]; then
        PYTHONPATH="$ROOT_DIR/tests${PYTHONPATH:+:$PYTHONPATH}" \
            "$TEST_PY" -m unittest -q "${TEST_TARGETS[@]}"
    else
        "$TEST_PY" -m unittest discover -s tests -q
    fi
    echo "✓ 主单元测试通过"
fi

# 2. 前端查看器测试
if [ "$RUN_JS" -eq 1 ]; then
    echo ""
    echo "[前端] 运行三维查看器 JS 测试..."
    if command -v node >/dev/null 2>&1; then
        node --test tests/js/viewer.test.mjs
        echo "✓ 查看器前端测试通过"
    else
        echo "[-] 未安装 node，跳过查看器前端测试"
    fi
fi

# 3. CLI 运行时健康检查（可选）
if [ "$RUN_CHECKS" -eq 1 ]; then
    echo ""
    echo "[自检] 运行 CLI 运行时自检..."
    ./umi-calib check
    ./umi-scene check
    if command -v c++ >/dev/null 2>&1; then
        KB4_TEST=$(mktemp)
        c++ -std=c++14 -DNDEBUG -I software/trajectory/setup/orbslam3 tests/cpp/test_kb4_bearing.cpp -o "$KB4_TEST"
        "$KB4_TEST"
        rm -f "$KB4_TEST"
    else
        echo "[-] 未安装 c++，跳过 KB4 射线几何测试"
    fi
    echo "✓ CLI 自检通过"
fi

echo ""
echo "=================================================="
echo "               已执行的测试与检查通过！"
echo "=================================================="
