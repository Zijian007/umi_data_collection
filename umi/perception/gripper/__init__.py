"""夹爪标签检测、夹爪开口映射与三维中心间距。"""

from umi.core.frame import validate_roi as validate_roi


def default_tag_roi() -> tuple[int, int, int, int]:
    from umi.config import operational_settings

    return operational_settings.values('gripper_tag_detection')['roi_px']
