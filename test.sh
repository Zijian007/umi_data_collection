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
RUN_LEGACY=1
RUN_JS=1

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --unit       仅运行主测试套件 (tests/ 下 unittest)"
    echo "  --legacy     仅运行夹爪开合历史测试 (software/gripper_width/legacy/tests/)"
    echo "  --js         仅运行查看器前端测试 (node --test)"
    echo "  --checks     仅运行 CLI 运行时健康检查 (umi-calib check, umi-scene check)"
    echo "  --all        运行全部（单元测试 + 历史测试 + 前端测试 + CLI 检查）"
    echo "  -h, --help   显示帮助信息"
}

if [ $# -gt 1 ]; then
    usage >&2
    exit 2
fi

if [ $# -gt 0 ]; then
    case "$1" in
        --unit)
            RUN_LEGACY=0; RUN_JS=0 ;;
        --legacy)
            RUN_UNIT=0; RUN_JS=0 ;;
        --js)
            RUN_UNIT=0; RUN_LEGACY=0 ;;
        --checks)
            RUN_UNIT=0; RUN_LEGACY=0; RUN_JS=0; RUN_CHECKS=1 ;;
        --all)
            RUN_CHECKS=1 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "未知选项: $1" >&2; usage >&2; exit 2 ;;
    esac
fi

echo "=================================================="
echo "          UMI Data Collection 统一测试"
echo "=================================================="

# 1. 主单元测试
if [ "$RUN_UNIT" -eq 1 ]; then
    echo ""
    echo "[1/3] 运行主单元测试套件 (tests/)..."
    "$TEST_PY" -m unittest discover -s tests -q
    echo "✓ 主单元测试通过"
fi

# 2. 夹爪开合历史测试
if [ "$RUN_LEGACY" -eq 1 ]; then
    echo ""
    echo "[2/3] 运行夹爪开合历史测试 (legacy/tests/)..."
    # 本套测试不使用外部 pytest 插件，避免加载系统 ROS 等无关依赖。
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$TEST_PY" -m pytest -q -p no:cacheprovider software/gripper_width/legacy/tests
    echo "✓ 夹爪开合测试通过"
fi

# 3. 前端查看器测试
if [ "$RUN_JS" -eq 1 ]; then
    echo ""
    echo "[3/3] 运行三维查看器 JS 测试..."
    if command -v node >/dev/null 2>&1; then
        node --test tests/js/viewer.test.mjs
        echo "✓ 查看器前端测试通过"
    else
        echo "[-] 未安装 node，跳过查看器前端测试"
    fi
fi

# 4. CLI 运行时健康检查（可选）
if [ "$RUN_CHECKS" -eq 1 ]; then
    echo ""
    echo "[4/4] 运行 CLI 运行时自检..."
    ./umi-calib check
    ./umi-scene check
    echo "✓ CLI 自检通过"
fi

echo ""
echo "=================================================="
echo "               已执行的测试与检查通过！"
echo "=================================================="
