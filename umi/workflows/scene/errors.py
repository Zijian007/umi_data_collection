"""场景处理的失败边界。"""

from __future__ import annotations

from ruamel.yaml.error import YAMLError


class MapUnavailableError(ValueError):
    """地图资产缺失、损坏或声明无效；允许 auto 切换整个任务。"""

    category = 'map_unavailable'


class MapCompatibilityError(MapUnavailableError):
    """地图可读取，但与当前场景、引擎、标定或视角不兼容。"""

    category = 'map_incompatible'


class PosterVioQualityError(ValueError):
    """poster_vio 输入有效，但观测或求解质量不足以交付轨迹。"""

    category = 'poster_vio_quality'


def fallback_record(error: Exception, *, category: str | None = None) -> dict:
    """稳定的路由类别与原始异常类型；完整调用栈仍由异常链保留。"""
    cause = error.__cause__ or error
    return dict(category=category or error.category, reason=str(error),
                error_type=type(cause).__name__)


# 单个批次或任务内的这些异常只把该项标记为失败，不中断同一命令里的其他项。
ITEM_FAILURES = (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError)
