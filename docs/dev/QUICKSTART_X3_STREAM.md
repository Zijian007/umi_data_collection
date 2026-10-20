# 五步：插上 Insta360 X3，在浏览器里看到画面

> 预览程序还没放进这个仓库。下面的步骤对应以后要搬来的 `software/tools/live_monitor/`。算轨迹从仓库根目录的 `README.md` 开始。

写给**第一次接触实时预览的人**。目标是一台全新的 Windows 机器，从 `git clone` 到看见画面
**15 分钟以内**，除了「用 Zadig 绑一次驱动」和「放一个 SDK 包」之外不需要你做任何判断。

> **只支持 Windows。** 相机驱动（WinUSB）和 Insta360 的 CameraSDK 都是这样。
> Linux/macOS 上这条链路跑不起来，别试。

---

## 第 1 步：装 Python 和 Git

- **Python 3.10 – 3.12**：<https://www.python.org/downloads/windows/>，
  安装时**务必勾上 "Add python.exe to PATH"**。
  - 最低要求其实是 **3.9**（实验室那台机器就是 3.9.7），但 `requirements-x3.txt` 钉死的
    包版本在 3.10–3.12 wheel 最齐；更新的 Python 上要现编译，慢且容易失败。
  - **别用微软商店里的那个"应用执行别名"**：`C:\...\WindowsApps\python.exe` 是 0 字节占位，
    运行它只会弹商店页面。脚本能认出这种情况并告诉你怎么关掉它。
- **Git**：<https://git-scm.com/download/win>，一路下一步即可。

验证（两条都要有输出）：

```bat
python --version
git --version
```

## 第 2 步：clone 仓库

预览程序还不在这个仓库里。下面的目录名和脚本路径是预览程序搬来之后的用法，现在按它操作不会找到 `software\tools\live_monitor`。

```bat
cd D:\
git clone https://github.com/Zijian007/umi_data_collection.git
cd umi_data_collection
```

### 如果 clone 要求登录

命令失败并报 `could not read Username for 'https://github.com'` 或 `Failed to enumerate credentials` 时，两条路二选一：

**A. 用 GitHub CLI（推荐）**

```bat
winget install GitHub.cli
gh auth login
```

选 `GitHub.com` → `HTTPS` → 让它帮你认证 Git 凭据 → 浏览器里登录。
之后 `git clone` 就能直接跑。

**B. 让负责人把你加成协作者，再用 HTTPS 凭据**

把你的 GitHub 用户名发给项目负责人，被加进仓库之后再 clone；
Windows 上第一次 clone 会弹 Git Credential Manager 的登录窗口，登一次就行。

> `x3_bridge.exe` 随预览程序一起，当前仓库里没有。不要拿 `x3-stream-v0.1` 来顶：那一版正常停流会被误判成断流，反复回收会话，最后只能把 USB 拔掉再插上。

## 第 3 步：自己向 Insta360 申请 CameraSDK，放到 `third_party\insta360\`

**这个 SDK 不在仓库里，我们也不会转发给你 —— 它受 Insta360 开发者协议约束，必须由你们
自己的单位去申请。** 这一步通常是整个流程里唯一要等的地方（约 3 个工作日），
**建议第 1 天就先提交申请**，等审批的时候把第 1、2 步做完。

### 3.1 申请（核实于 2026-09-22，以官网当前页面为准）

1. 打开 **<https://www.insta360.com/sdk/apply>**，填写申请信息。
2. 表单会问：是否持有 Insta360 相机、你属于哪类用户（个人 / 企业 / 开发者）、
   联系方式与地址、**申请 SDK 的用途**，并要求同意 *End User License Agreement for
   Insta360 SDK*。
   —— 具体字段官网可能调整，**以你打开时看到的表单为准**；我们这次没能直接打开表单页
   核对（申请页对自动访问返回 403），上面这份清单来自 Insta360 官方在线手册与其
   SDK 上线公告。
3. 填用途时把这几件事说清楚会顺利很多：**单位名称**、**用途**（例如"机器人示范数据采集，
   读取相机预览流与 IMU"）、**相机型号 Insta360 X3**、**平台 Windows**。
4. **审批结果会发邮件，通常 3 个工作日内**。通过后邮件里有下载链接，
   官方说明该链接**长期有效**，点开总是最新版的 SDK，不需要重新申请。
5. 下载得到的是 `CameraSDK-<版本>-win64.zip` 这样的压缩包。

> 官方文档明确写了 **X3 在 Camera SDK 的支持列表里**（X5 / X4Air / X4 / **X3** /
> ONE RS 1-Inch / ONE RS / ONE X2 / ONE R / ONE X）。
> 参考：<https://onlinemanual.insta360.com/developer/en-us/resource/sdk>（2026-09-22 访问）

### 3.2 拿到之后怎么放

**原样放进去，不要解压、不要改名**：

```text
D:\umi\third_party\insta360\CameraSDK-20250812_192505-2.1.1-win64.zip
```

- 脚本按 **`CameraSDK-*.zip` 通配**识别，所以**版本号不同也认**，并且会把识别到的文件名
  打印出来。
- **我们验证过的是 2.1.1。** 如果你的包是别的版本，脚本会打印一行黄色提示说"未验证" ——
  多半照样能用，但出问题时请先在 review 里说明你用的是哪个版本。
- 这个目录已经在 `.gitignore` 里，**你放进去的 zip 永远不会被 commit**。

放好之后**重新跑一遍第 4 步的命令**：脚本会把 `CameraSDK.dll` 等运行时解压到
`software\tools\live_monitor\bin\`，然后自动往下走。

## 第 4 步：插上相机，跑一个脚本

1. 用**数据线**（不是只能充电的线）把 X3 接到电脑，开机。
2. **在相机屏幕上选 `Android` 模式**（不是"U 盘 / USB 存储"）。
3. 在仓库根目录跑：

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File software\tools\live_monitor\setup_x3_stream.ps1
```

机器上有好几个 Python、不想让脚本去猜 PATH 上是哪一个时，**直接指定**（不改 PATH、
不动机器上现有的环境）：

```bat
for /f %i in ('py -3.12 -c "import sys;print(sys.executable)"') do set PY312=%i
powershell -NoProfile -ExecutionPolicy Bypass -File software\tools\live_monitor\setup_x3_stream.ps1 -Python %PY312%
```

给错路径它会直接说「`-Python` 指到的文件不存在」，**不会悄悄退回去用 PATH 上那个** ——
否则你以为在测 3.12，实际测的是别的。

脚本依次做：查 Python → 建 `.venv-x3` 装依赖 → 找 SDK 解运行时 → 拿 `x3_bridge.exe`
（**默认直接用仓库自带的那一个**，校验 sha256；`-FromRelease` / `-ForceBuild` 才走别的路）
→ 设备与驱动自检 → 起服务打印地址。**任何一步不通过都只打印一句能照做的话**，改完重跑即可。

**第一次跑通常会停在"驱动没绑好"**，脚本会把 Zadig 的步骤打出来：

```text
  1. 下载 Zadig（https://zadig.akeo.ie/），用管理员身份运行
  2. 菜单 Options -> List All Devices 勾上
  3. 在下拉框里选 Insta360 X3
  4. 核对它右边的 USB ID 是 2E1A 0002
  5. 目标驱动选 WinUSB，点 Install Driver
  6. 拔下来再插上，重新跑脚本
```

> **第 4 步那条"核对 USB ID"不是客套话。** Zadig 会替换你选中的那个设备的驱动 ——
> 选错了，那个设备（键盘、鼠标、U 盘、网卡……）会当场失效，要进设备管理器卸载驱动才能恢复。
> **确认是 `2E1A 0002` 再点 Install。**

## 第 5 步：打开浏览器

脚本最后会打印一组地址（同时写进 `serve_urls.txt`）。**打开第一个 `PAGE` 地址就能看到画面。**

```text
PAGE     [LAN] http://192.168.x.x:8080/
  mjpeg  [LAN] http://192.168.x.x:8080/stream.mjpg
  h264   [LAN] http://192.168.x.x:8080/stream.h264
  snap   [LAN] http://192.168.x.x:8080/snapshot.jpg
  status [LAN] http://192.168.x.x:8080/status.json
```

| 端点 | 拿来干什么 |
|---|---|
| `/` | 浏览器直接打开：实时画面 + 状态面板 + 档位切换（见下面"页面怎么用"） |
| `/stream.mjpg` | VLC / ffplay / OpenCV `cv2.VideoCapture(url)` |
| `/stream.h264` | 相机自己的 H.264，**原样直通不转码**；VLC「打开网络串流」或 `ffplay -f h264 http://…` |
| `/snapshot.jpg` | `curl -O` 取一帧 |
| `/status.json` | fps、码率、重开次数、电量、卡剩余、当前档与开机默认档、切换状态 |

### 页面怎么用（2026-09-24 起）

打开 `/` 看到的是一个控制页（纯内联、不连外网，手机上也能用，深色 / 浅色跟系统走）：

- **左边是画面**（MJPEG：服务端把相机的 H.264 解码后转成 JPEG，永远发最新一帧）。
  双鱼眼左右并排，**夹爪在右侧那个圆里**。断线会自己重连。
- **右边"状态"每秒刷新**：档位、主机钟帧率、页面画面帧率、码率、最后一帧多久以前、断流重开次数、
  电量 / 供电方式、卡剩余、在线时长。
- **"档位"只列已验证的四档**（全是 30 fps）：9 = 1440×720（≈4 Mbit/s）、74 = 2304×1152（≈10）、
  1 = 2560×1280（≈45）、148 = 2944×1472（≈10）。每个按钮写着分辨率、码率和一句用途。点一下就切；
  切换期间画面上有"切换中"遮罩，切完画面自动重连。其它档（60 fps 档、只给房间镜头的 180）不显示，
  用 `curl` 硬塞也会被拒（HTTP 400）；正在切时再点是 409；两次切换至少隔 10 秒（429）。
- **"当前档设为开机默认"**：勾上 = 把当前档写进桥同目录的 `serve_config.json`，以后服务（计划任务）
  启动就用它；取消勾选 = 回到仓库默认 9。命令行 `--live-res` 永远优先于这个文件。
- **"下载快照"** 存一张当前帧；**"停止服务"** 要确认两次，走优雅停止（`LIVE_STOP` → 等桥自己退），
  停了之后要在笔记本上重新启动服务。

不用页面也行：`curl -X POST "http://<地址>:8080/switch?res=148"`、`curl http://<地址>:8080/status.json`。

> 切档时画面会断两三秒，这是正常的。

**同一个局域网里的别的机器怎么看**：用脚本打印的 `[LAN]` 那一组地址（比如
`http://192.168.x.x:8080/`）。第一次跑的时候 Windows 可能会弹防火墙提示，
**勾上"专用网络"并允许**；不弹或者弹了点了"取消"，就手动放行一次：

```bat
netsh advfirewall firewall add rule name="umi_x3_stream_8080" dir=in action=allow protocol=TCP localport=8080
```

撤销（不用了就执行它）：

```bat
netsh advfirewall firewall delete rule name="umi_x3_stream_8080"
```

---

## 怎么停（三种，都别 taskkill）

| 方式 | 命令 |
|---|---|
| 在脚本窗口里 | `Ctrl+C` |
| 从任何机器 | `curl -X POST http://<地址>:8080/stop` |
| 放个文件 | 在服务的工作目录建一个空文件 `stop.flag` |

三种都走同一条收尾：`LIVE_STOP` → 等桥自己退出。

> **千万不要 `taskkill` 桥进程。** 在实时流跑着的时候打断它，会把相机会话搞到只能**物理拔插**
> 才能恢复 —— 这件事在开发期间发生过两次。

## 常见错误对照表

| 你看到的 | 为什么 | 怎么办 |
|---|---|---|
| `没有找到 Python 3.9 或更新的版本` | Python 没装，或装了没进 PATH | 重装并勾 "Add python.exe to PATH"，然后**开一个新终端**再跑 |
| `PATH 上的 python 是微软商店的占位别名` | 商店的应用执行别名挡在真 Python 前面 | 设置 → 应用 → 高级应用设置 → 应用执行别名，关掉 python.exe / python3.exe |
| `找到的 Python 太旧了：3.x` | 低于 3.9 | 装 3.10–3.12；不想动现有环境就用 `py -3.12` 起脚本 |
| `建虚拟环境失败` | 商店版 Python 有时缺 venv | `python -m ensurepip --upgrade`；还不行改用 python.org 的安装包 |
| `装依赖失败` + pip 报网络错 | 上不了网或要走代理 | 先 `set HTTPS_PROXY=...`，或 `pip config set global.index-url <公司源>` |
| `钉死的版本装不上，退回到不钉版本` | 你的 Python 版本没有那几个 wheel | 能用，但不是验证过的组合；想完全复现就换 Python 3.10–3.12 |
| `在 …\third_party\insta360 里没有找到 CameraSDK-*.zip` | SDK 还没申请到或没放对地方 | 见第 3 步；注意是 **zip 原样放**，不要解压 |
| `USB 上没有找到 Insta360 X3（2E1A 0002）` | 没插 / 用了充电线 / 相机没开机 | 换数据线，开机，相机屏幕选 Android 模式 |
| `相机多半在 U 盘模式` | 相机 USB 模式选成了 U 盘 | **在相机屏幕上**改成 Android 模式 |
| `驱动没绑好（status=…, driver=…）` | Zadig 还没跑 | 按脚本打印的六步绑 WinUSB，**核对 USB ID 2E1A 0002** |
| `下载到的 x3_bridge.exe 校验不过` | 下载被截断/被改 | 删掉 `bin\x3_bridge.exe` 重跑；或 `-ForceBuild` 本地编译 |
| `Release 下载失败，本机也没有 MSVC 编译器` | 没网又没编译器 | 手动下 Release 里的 exe 放到 `bin\`，或装 VS2022 的「使用 C++ 的桌面开发」 |
| 浏览器打开是空白/转圈 | 地址用了 `127.0.0.1` 但你在别的机器上 | 用脚本打印的 `[LAN]` 地址；必要时放行 8080（见上） |
| 画面卡住不动、`status.json` 的 `last_frame_age_s` 一直涨 | 相机侧断流 | 不用管，桥会自动回收会话重连；连续 3 次救不回来才会提示拔插 |

## 已知限制

1. **只支持 Windows。** 驱动和 SDK 都是。
2. **仓库默认档是 1440×720 @30 fps（enum 9，≈4 Mbps）**；想要画质就在页面上切到 **2560×1280（enum 1，
   ≈44.5 Mbps）**。清晰度和块效应都比默认档好。开机默认可以写进桥旁边的 `serve_config.json`。
   四档（9 / 74 / 148 / 1）都过了 30 分钟浸泡零重开。
3. **60 fps 档（`-LiveRes 28`，3840×1080 并排 = 每镜头 1920×1080）是实验档，不稳**：
   视频流会在 **29–118 秒**之间随机停止推送，原因未知。现在桥会自动做会话回收把它救回来，
   但**没有完成验证，不要拿它采数据**。
4. **预览流不是数据源。** 它不进 `data/raw/`，不能用于标定或位姿。训练数据仍然只来自卡上的录像。
5. **`--bind 0.0.0.0` 没有任何鉴权** —— 同一局域网里谁都能打开你打印的地址。
   不想这样就加 `--bind 127.0.0.1`。
6. **CameraSDK 的许可要你们自己确认。** 它受 Insta360 开发者协议约束，
   **请勿再分发**；本仓库不包含它，也不会包含。

## 出了这份文档解决不了的问题

带上这三样：`serve_urls.txt`、脚本完整输出、`software\tools\live_monitor\bin\` 的目录列表。
