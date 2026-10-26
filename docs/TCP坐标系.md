# TCP 坐标系

当前默认采用与 Genie omnipicker 机器人一致的机械轴向约定，标识为 `genie_omnipicker_axes_v1`。左右手使用相同的工具自身轴向定义。

## TCP 原点和三轴

原点仍是**夹爪闭合时，两指尖接触位置的中点**。本次只改轴向，保留原相机到指尖的平移。

| 轴 | 正方向 |
|---|---|
| +X | 从外侧指指向内侧指，用户确认内侧指位于前镜头画面左侧 |
| +Z | 从夹爪本体指向指尖 |
| +Y | 按右手规则确定，即 `Y = Z × X` |

三轴固定在夹爪上并随它运动。+Z 使用本项目既有 CAD 中的本体到指尖方向，与前镜头光轴约有 21.5° 倾角，不再与镜头朝向重合。位置和倾角仍是 CAD 名义值；轴向约定的统一不构成新的物理标定。

## 机器人依据与相机坐标

轴向依据 `genie_robot_description/urdf/G2_t2_crs/G2_t2_crs_omnipicker.urdf` 的 `gripper_l_center_link`、`gripper_r_center_link`：二者相对夹爪基座均绕 Z 转 −90°，因此 +X 沿基座 −Y、+Y 沿基座 +X、+Z 沿基座 +Z。基座中内侧指在 −Y，外侧指在 +Y。

URDF 中的 `camera_link` 是机械 link，其三轴与夹爪基座平行，并非本项目 X3 的光学坐标系。这里使用本项目的相机—指尖 CAD 外参和用户确认的手指方向计算旋转；没有将 URDF 的 `Rz(-90°)` 直接当作光学相机到 TCP 的旋转。

## 与 wholebody VR 数采的区别

wholebody 当前记录的是 `arm_l_end_link` / `arm_r_end_link` 的**末端法兰位姿**，尚未自动换算到 `gripper_*_center_link`。本项目 `traj_tcp.txt` 则已经通过相机到指尖的几何换算为**夹爪 TCP 位姿**。工具轴向约定相同，不代表两套输出记录了同一个参考点。

机器人实测与目标都需要先使用实际安装的 `T_end_tcp` 换算法兰到 TCP；随后再统一父坐标系或转换为一致的相对位姿表示。wholebody 新录制已将 VR 目标转换到 `base_link`；修复前的 `arm_base_link` 目标数据需单独核对和转换，详见其[数采字段说明](../../wholebody/docs/user/collect/schema.md#末端法兰与夹爪-tcp)。

对于 `G2_t2_crs_omnipicker.urdf`，左右 `T_end_tcp` 的平移均为法兰系下 `[0, 0, 0.14918]` m；左侧旋转为单位阵，右侧旋转为 `diag(-1, -1, 1)`，即绕法兰 Z 转 180°。这里比较的是各侧法兰与该侧夹爪 TCP，左右夹爪自身仍采用相同的机械轴向定义。完整矩阵、固定关节来源、四元数顺序及正反换算见[法兰到 TCP 的固定变换](../../wholebody/docs/user/collect/schema.md#omnipicker-法兰到-tcp-的固定变换)。

## 程序怎样计算

采用 `T_a_b` 将 b 系坐标变换到 a 系的约定。当前相机到 TCP 外参约为：

```text
R_camera_tcp ≈
[ -1   0           0          ]
[  0  -0.93041757 -0.36650123 ]
[  0  -0.36650123  0.93041757 ]

t_camera_tcp = [-0.0001815, 0.0369877, 0.1650461] m

T_scene_tcp = T_scene_camera × T_camera_tcp
```

旋转从原机械 CAD 定义导出：保留 +Z，将原 CAD 的 X、Y 同时反向以匹配“内侧指在画面左侧”的确认。相机中的指尖位置保持原值；正反外参和四元数随新资产保存。

## 与场景坐标系的关系

场景坐标系固定在环境中，+Z 朝上；TCP 坐标系随夹爪运动。

- `traj_tcp.txt` 的 `x y z` 是 TCP 原点在场景中的位置，单位米。
- `qx qy qz qw` 是 TCP 自身三轴相对于场景三轴的朝向。
- 双手使用同一地图版本时共享场景坐标。独立 `umi-traj` 的世界坐标每段分别建立。

本次统一工具轴向不提供场景到机器人基座的外参，也不改变机器人控制器的 TCP 配置。部署仍需确认控制器实际使用的工具坐标系。

## 旧结果与新结果

旧资产不可覆盖，各版本按自己的定义读取：

| 版本或约定 | TCP 轴向 |
|---|---|
| `b29b429c0fcc` | 原机械 CAD 轴，开合方向正负未与机器人对应 |
| `b90726d8a0ed` / `front_optical_parallel_v1` | 与前镜头平行，+X 图像右、+Y 图像下、+Z 镜头前方 |
| `genie_omnipicker_axes_v1` | 本文定义的机器人机械轴，+X 外侧指向内侧指、+Z 本体指向指尖 |

这些版本的 TCP 原点位置相同，姿态约定不同。默认版本改变后，独立 `umi-traj` 的新运行读取新默认；已有场景套件和批次仍固定原版本。已有结果不会改写。

已有套件需要采用新约定时，用实际序列号更新套件，再创建新批次：

```bash
./umi-scene kit scenes/shelf kit_001 --left LEFT_SERIAL --right RIGHT_SERIAL
./umi-scene session scenes/shelf --name 20261026_003 --kit kit_001
```

新批次继续复用场景地图，无需因工具换轴重新建图。新运行的实际 `tcp_frame` 写入轨迹文件头、各手的 `details/summary.json` 和 `details/run_settings.json`；场景 `episode.json` 的 `tcp_frames` 记录各手约定，并随导出保留。

## front.mp4 的来源

程序从 X3 原始 `.insv` 中提取前镜头画面并生成 `front.mp4`，用于查看和核对手指方向。双手场景结果分别位于 `<任务>/results/left/front.mp4` 和 `<任务>/results/right/front.mp4`；独立轨迹结果位于 `<输出目录>/front.mp4`。原片无需另行拍摄或上传这个 MP4。

## 资产位置

源定义为 [t_tcp_camera_nominal_cad.json](../software/trajectory/assets/calib/t_tcp_camera_nominal_cad.json)。运行时读取 `registry/grippers/handheld_v1/current.json` 指向的版本，已有批次使用自己的固定版本。新资产记录了原 CAD 来源、机器人 URDF 的 SHA256 和用户确认的轴向依据。

[双手采集操作手册](操作手册.md) · [设备与目录](设备与目录.md)
