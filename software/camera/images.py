"""镜头图像资产准备，不选择设备或运行融合。"""
from pathlib import Path

def prepare_mask(source: Path, output: Path, size_px: int) -> Path:
    """按同一视野缩放遮手图；最近邻保留遮挡类别，原始设备资产只读。"""
    import cv2

    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[0] != mask.shape[1]:
        raise ValueError(f'遮手图必须是可读取的正方形图像：{source}')
    if mask.shape == (size_px, size_px):
        return source
    resized = cv2.resize(mask, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(output), resized):
        raise ValueError(f'无法写入本次运行的遮手图：{output}')
    return output

