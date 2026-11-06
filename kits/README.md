# 设备套件

在 `kits/<套件>/kit.yaml` 保存左右相机、相机标定和夹爪几何版本，可跨场景复用。例如下面的单右手配置，将序列号替换为真实设备，并选择已登记的版本：

```yaml
arms:
  right:
    serial: CAMERA_SERIAL
    calibration: '001'
    gripper: handheld_v1
    gripper_version: '003'
```

也可用 `./umi-scene kit kit_001 --left LEFT_SERIAL --right RIGHT_SERIAL` 根据设备当前默认值生成；手写和命令生成二选一。

套件 ID 来自目录，不重复写入 YAML。测量文件保存在 `registry/`，无需复制。场景批次在 `batch.yaml` 中声明 `kit: kit_001`。每次 `run` 都读取当前套件；修改标定或几何编号后，重跑当前批次即可生效。历史运行保留原来使用的版本，无需编辑 `.umi/manifest.yaml`。

`.example.yaml` 不作为真实套件读取；版本号写成 `'001'`、`'003'` 等字符串，保留引号。
