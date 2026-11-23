# 场景批次目录

每次采集使用 `scenes/<场景>/<批次>/`。从[批次模板](example_scene/collection.example.yaml)复制一份 `collection.yaml`，再按[批次准备指南](../docs/01-安装与日常使用.md#4-准备本次配置和视频)放置原片。

- `mapping/` 保存地图后端的建图原片。
- `take_*/left/` 和 `take_*/right/` 保存已分组任务的原片，`pending/` 保存待配对原片。
- `take_*/results/` 是成功结果入口，`.pipeline/` 由程序管理。

运行 `./umi-scene run scenes/<场景>/<批次>`。后端、地图复用、补片和排障见[场景处理](../docs/03-场景处理.md)。
