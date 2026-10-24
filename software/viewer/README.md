# 查看器代码阅读指南

运行和操作见 [三维查看指南](../../docs/查看.md)，数据格式见 [轨迹结果](../../docs/轨迹/结果.md)。

## 阅读顺序

`server.main()` 创建 `RecordingCatalog` 并启动 HTTP 服务；`data.py` 通过轨迹模块的统一入口读取结果、视频和摘要。浏览器的 `app.mjs` 管理选择与播放状态，三维、视频、曲线共用轨迹时间轴。

## 文件职责

```text
software/viewer/
  data.py            结果发现、TUM / CSV 读取和时间对齐
  server.py          HTTP API、本地静态资源与视频 Range 流
  media.py           兼容模式的限并发 FFmpeg 解码
  static/
    app.mjs          录像选择与统一播放状态
    scene.mjs        Z-up 透视投影、轨迹和姿态轴、鼠标操作
    chart.mjs        速度曲线与点击定位
    media.mjs        兼容画面请求合并、解码及与位姿同步
    math.mjs         时间定位与四元数运算
    viewer.css       自适应布局
```

浏览器侧自行计算三维透视投影，Canvas 显示轨迹；不从 CDN 加载绘图库。大轨迹只在绘制轮廓时降采样，逐帧读取和时间定位保留原始索引。对录像目录外的软链接不暴露文件内容，视频支持 HEAD 和单段 HTTP Range，便于拖动跳转。

测试与依赖环境见 [代码维护](../../docs/dev/README.md#验证)。Node 仅用于前端测试。
