#!/usr/bin/env bash
# 统一测试与质量检查入口
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CALIB_PY="$ROOT_DIR/.deps/calib_env/bin/python"
ENV_PY="$ROOT_DIR/.deps/env/bin/python"

RUN_ALL=0
RUN_UNIT=1
RUN_LEGACY=1
RUN_JS=1

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --unit       仅运行主测试套件 (tests/ 下 293+ 项 unittest)"
    echo "  --legacy     仅运行夹爪开合历史测试 (software/gripper_width/legacy/tests/)"
    echo "  --js         仅运行查看器前端测试 (node --test)"
    echo "  --checks     额外运行 CLI 运行时健康检查 (umi-calib check, umi-traj check)"
    echo "  --all        运行全部（单元测试 + 历史测试 + 前端测试 + CLI 检查）"
    echo "  -h, --help   显示帮助信息"
    exit 0
}

if [ $# -gt 0 ]; then
    case "$1" in
        --unit)
            RUN_LEGACY=0; RUN_JS=0 ;;
        --legacy)
            RUN_UNIT=0; RUN_JS=0 ;;
        --js)
            RUN_UNIT=0; RUN_LEGACY=0 ;;
        --checks)
            RUN_UNIT=0; RUN_LEGACY=0; RUN_JS=0; RUN_ALL=1 ;;
        --all)
            RUN_ALL=1 ;;
        -h|--help)
            usage ;;
        *)
            echo "未知选项: $1"; usage ;;
    esac
fi

echo "=================================================="
echo "          UMI Data Collection 统一测试"
echo "=================================================="

# 1. 主单元测试
if [ "$RUN_UNIT" -eq 1 ]; then
    echo ""
    echo "[1/3] 运行主单元测试套件 (tests/)..."
    if [ -x "$CALIB_PY" ]; then
        "$CALIB_PY" -m unittest discover -s tests -q
    elif [ -x "$ENV_PY" ]; then
        echo "[提示] 未检测到 calib_env，回退至 env 环境..."
        "$ENV_PY" -m unittest discover -s tests -q
    else
        python3 -m unittest discover -s tests -q
    fi
    echo "✓ 主单元测试通过"
fi

# 2. 夹爪开合历史测试
if [ "$RUN_LEGACY" -eq 1 ]; then
    echo ""
    echo "[2/3] 运行夹爪开合历史测试 (legacy/tests/)..."
    if [ -x "$ENV_PY" ]; then
        "$ENV_PY" -m pytest -q -p no:cacheprovider software/gripper_width/legacy/tests
    elif [ -x "$CALIB_PY" ]; then
        "$CALIB_PY" -m pytest -q -p no:cacheprovider software/gripper_width/legacy/tests
    else
        pytest -q -p no:cacheprovider software/gripper_width/legacy/tests
    fi
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
if [ "$RUN_ALL" -eq 1 ]; then
    echo ""
    echo "[4/4] 运行 CLI 运行时自检..."
    ./umi-calib check
    ./umi-traj check
    echo "✓ CLI 自检通过"
fi

echo ""
echo "=================================================="
echo "               全部测试与检查通过！"
echo "=================================================="
