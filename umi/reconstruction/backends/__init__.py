"""场景任务的三个重建后端。"""


class PosterVioQualityError(ValueError):
    """poster_vio 输入有效，但观测或求解质量不足以交付轨迹。"""

    category = 'poster_vio_quality'


class PosterIndependentQualityError(ValueError):
    """独立海报重建未通过可回退的质量门槛。"""

    category = 'poster_independent_quality'
