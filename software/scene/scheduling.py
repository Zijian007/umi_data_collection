"""单任务的双手并发预算；多任务调度时不再嵌套并行。"""

from __future__ import annotations

from software import settings


def arm_cores(arms: tuple[str, ...], cores: str, *, parallel: bool) -> dict[str, str] | None:
    if not parallel or len(arms) < 2:
        return None
    available = settings.available_cpus(cores)
    # ORB 每手约需四核；核心不足时沿用串行。
    if len(available) < settings.values('runtime')['arm_min_cpus'] * len(arms):
        return None
    ordered = sorted(available)
    return {arm: ','.join(map(str, ordered[index::len(arms)])) for index, arm in enumerate(arms)}
