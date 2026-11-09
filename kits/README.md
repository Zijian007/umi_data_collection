# 设备套件

每个套件的 `kit.yaml` 指定左右相机序列号、相机标定版本和当前安装的夹爪文件，可跨场景复用。仅需手动维护，不需要安装或登记夹爪。

```yaml
arms:
  left:
    serial: LEFT_CAMERA_SERIAL
    calibration: '002'
    gripper:
      tcp: left_tcp_camera.json
      width: left_width.json
  right:
    serial: RIGHT_CAMERA_SERIAL
    calibration: '002'
    gripper:
      tcp: right_tcp_camera.json
      width: right_width.json
```

只保留实际使用的手。相机序列号和标定版本必须已在 registry 中登记，版本号保留引号。夹爪文件路径相对 `kit.yaml`，需位于套件目录内：

| 字段 | 要求 |
|---|---|
| `gripper.tcp` | 必填；相机与 TCP 的安装变换 JSON，平移单位为米 |
| `gripper.width` | 可选；当前安装的像素距离与实际开口毫米数的实测对应关系 JSON |
| `gripper.mask` | 可选；自定义遮手 PNG，省略时使用通用遮手图 |
| `rear_calibration` | 每手可选的相机字段；dual 模式填写已登记的后镜头版本 |

左右安装几何相同时，可以共同引用一个 `tcp_camera.json`；不同时分别填写。没有实测开口关系时省略 `width`，不要填写示例数值。TCP 格式与方向见[设备与坐标](../docs/设备与坐标.md#几何资产)，宽度格式与测量方法见[训练数据指南](../docs/训练数据.md#夹爪宽度标定与接入)。

```text
kits/kit_001/
├── kit.yaml
├── tcp_camera.json
├── left_width.json       # 有实测数据时创建
└── right_width.json      # 有实测数据时创建
```

批次在 `batch.yaml` 中声明 `kit: kit_001`。每次 `run` 读取当前套件，在开始计算前解析测量并复制到运行目录 `devices/<序列号>/`，结果记录实际文件的哈希。修改文件后重跑即可，历史运行保留自己的副本。配置变更适用于下一次运行，无需编辑 `.umi/manifest.yaml`。

`registry/` 仅保存相机档案和版本化标定。套件 ID 来自目录，不重复写入 YAML；套件与夹爪不再另设版本目录或默认指针。`.example.yaml` 不作为真实套件读取。
