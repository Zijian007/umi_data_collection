# 环境海报

场景处理使用环境镜头 `rear` 观测这张 ArUco 海报，根据已知角点几何估计海报相对相机的位姿。在批次的 `collection.yaml` 中通过 `environment_poster.layout_path` 选择布局。 [场景处理说明](../../docs/03-场景处理.md)

海报采用 `DICT_4X4_100` 字典和 1 格黑边。7 列 × 11 行，ID 从左到右、从上到下为 12–88，标签朝向一致。PDF 纸张为 1000 × 1500 mm，黑色外框边长 100 mm，中心间距 130 mm，左右页边距为 60 mm，上下为 50 mm。 [打印 PDF](environment_poster.pdf) · [布局 JSON](environment_poster.json)

布局保存每个 ID 的四个米制角点。原点是纸张左上角，X 向右、Y 向下、Z 指入纸面。角点顺序与 OpenCV 解码后的 TL/TR/BR/BL 一致。布局记录来源 PDF 的 SHA-256。已经核对全部 77 个 PDF 矢量外框，并通过渲染和解码验证标签的 ID、位置和朝向。

当前 `geometry_status: pdf_design` 表示已确认设计尺寸。打印时使用 100% 实际尺寸，并实测黑框边长及横纵跨标签距离。打印非等比缩放、折皱或单个标签重新粘贴会使旧几何失效。海报整体可在不同录像之间移动，各段位姿重新求解。同一段录像内保持静止、平整。

独立检测入口见[视觉模块说明](../../umi/perception/README.md)。
