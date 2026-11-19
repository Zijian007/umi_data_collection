"""训练采样质量策略的公共取值与入口校验；无数值求解依赖。"""

QUALITY_POLICIES = ('all', 'position_supported')
UNKNOWN_SUPPORT_POLICIES = ('drop', 'error')


def validate_quality_policy(quality_policy: str, unknown_support: str) -> None:
    if quality_policy not in QUALITY_POLICIES:
        raise ValueError(f'未知训练质量策略：{quality_policy}')
    if unknown_support not in UNKNOWN_SUPPORT_POLICIES:
        raise ValueError(f'未知位置支持处理策略：{unknown_support}')


