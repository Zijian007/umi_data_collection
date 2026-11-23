"""夹爪标签检测、夹爪开口映射与三维中心间距。"""

from umi.core.frame import SRC_H


def validate_roi(roi, shape: tuple[int, int] = (SRC_H, SRC_H)) -> tuple[int, int, int, int]:
    if not isinstance(roi, (tuple, list)) or len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError('ROI 需要 LEFT TOP RIGHT BOTTOM 四个原始像素坐标')
    left, top, right, bottom = roi
    if not (0 <= left < right <= shape[1] and 0 <= top < bottom <= shape[0]):
        raise ValueError(f'ROI 必须位于 {shape[1]}×{shape[0]} 工作镜头半幅内，且 RIGHT > LEFT、BOTTOM > TOP')
    return left, top, right, bottom


def default_tag_roi() -> tuple[int, int, int, int]:
    from umi.config import operational_settings

    return operational_settings.values('gripper_tag_detection')['roi_px']
