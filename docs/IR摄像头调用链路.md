# Windows Hello 红外摄像头调用链路

> 用途：在 Fylgja 中接入本机 Windows Hello 红外（IR）摄像头。
> 本文档记录 **2026-11-01 在本机（ASUS IR camera）实测验证通过** 的调用链路、关键参数与坑位。
> 参考实现：`IRCameraView-master/`（C# UWP/WinUI，解压自同目录 zip）。

---

## 0. 结论

**可以调用。** 但 IR 摄像头 **不出现在常规摄像头枚举里**（OpenCV `VideoCapture` / DirectShow / Media Foundation / WinRT `DeviceClass.VideoCapture` 全部看不到它），只能通过 **`MediaFrameSourceGroup`（Media Capture Frames）** 通道以 **Infrared 帧源** 打开。

本机实测结果：

| 项 | 结果 |
|---|---|
| 打开设备（MediaCapture InitializeAsync） | SUCCESS（ExclusiveControl + Cpu） |
| 读帧（MediaFrameReader） | SUCCESS，640×360，subtype `L8`（Gray8），15 FPS |
| 红外补光交替 | 采样 20 帧 = 10 帧补光开 + 10 帧补光关（严格交替） |
| 测试样张 | `ir_test_frame_raw.png`（无补光）/ `ir_test_frame_illuminated.png`（补光开） |

---

## 1. 硬件清单（本机实测）

| 设备 | PnP 实例 | 说明 |
|---|---|---|
| `ASUS IR camera` | `USB\VID_3277&PID_0096&MI_02` | 红外摄像头（UVC，驱动 `usbvideo`），状态 OK |
| `ASUS FHD webcam` | `USB\VID_3277&PID_0096&MI_00` | RGB 摄像头（同一 USB 复合设备） |
| `Facial Recognition (Windows Hello) Software Device` | `ROOT\WINDOWSHELLOFACESOFTWAREDRIVER\0000` | Hello 人脸软件设备，`DeviceCapable=1` |
| `Rts-DMFT-Group` | `SWD\SGDEVAPI\...` | Realtek DMFT 把 RGB+IR 配对成双帧源组（Color + Infrared） |

**术语澄清**：这是一颗 **二维近红外（NIR）摄像头**（灰度 L8 + 红外 LED 补光），Windows Hello 人脸认证用的就是它。它**不是**结构光点阵投射/深度传感器（如 RealSense SR300）；本机无深度输出。对 Fylgja 而言，它就是一台 640×360 灰度、带补光交替的普通视频源。

---

## 2. 关键事实：为什么常规方式打不开

设备接口注册表里，IR 摄像头（MI_02）同时注册了多个接口类，其中：

- `{e5323777-…}` = `KSCATEGORY_VIDEO_CAMERA` —— 虽已注册，但 **WinRT `DeviceInformation.FindAllAsync(DeviceClass.VideoCapture)` 枚举不到它**（Windows 对 sensor camera 做了应用侧过滤）；
- `{24e552d7-6523-47f7-a647-d3465bf1f5ca}` = **`KSCATEGORY_SENSOR_CAMERA`** —— **应用可见的入口**，对应设备符号链接：
  ```
  \\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&2&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
  ```

因此：

- OpenCV（DShow/MSMF）、DirectShow 枚举、普通"相机应用"列表 → **看不到 IR**；
- `MediaFrameSourceGroup.FindAllAsync()` → **能看到**，返回两个相关组：
  - `ASUS IR camera`（仅 Infrared 帧源）← **推荐，纯净 IR 流**
  - `Rts-DMFT-Group`（Color + Infrared 双帧源，适合 RGB+IR 同步场景）

---

## 3. 验证过的调用链路（逐步）

```
MediaFrameSourceGroup.FindAllAsync()
  └─ 选 DisplayName ≈ "ASUS IR camera" 的组（含 Infrared 帧源）
       │
MediaCapture.InitializeAsync(settings)
  ├─ settings.SourceGroup        = 该组
  ├─ settings.SharingMode        = ExclusiveControl     ← 必须
  ├─ settings.StreamingCaptureMode = Video
  └─ settings.MemoryPreference   = Cpu                  ← 必须，否则拿不到 SoftwareBitmap
       │
frameSource = mediaCapture.FrameSources 中 SourceKind == Infrared 的那个
frameSource.SetFormatAsync(SupportedFormats[0])        ← 先设格式再建 reader
       │
mediaCapture.CreateFrameReaderAsync(frameSource)
  └─ reader.AcquisitionMode = Realtime                 ← 建议显式设置
       │
reader.StartAsync()
       │
循环: frame = reader.TryAcquireLatestFrame()
  ├─ frame.VideoMediaFrame.SoftwareBitmap              → 画面（Gray8 640×360）
  ├─ frame.VideoMediaFrame.VideoMediaFrameFormat        → 宽高/帧率/子类型
  └─ frame.VideoMediaFrame.InfraredMediaFrame.IsIlluminated → 本帧是否带红外补光
       │
reader.StopAsync() → Dispose
```

### 各步骤验证状态（本机）

| 步骤 | 状态 |
|---|---|
| FindAllAsync 找到 IR 组 | ✅ |
| InitializeAsync（ExclusiveControl + Cpu） | ✅ SUCCESS |
| SetFormatAsync（640×360 @15fps L8） | ✅ OK |
| CreateFrameReaderAsync + StartAsync | ✅ Success |
| TryAcquireLatestFrame 拿到帧 | ✅ SoftwareBitmap = Gray8 640×360 |
| IsIlluminated 标志 | ✅ False/True 交替（10/10 采样） |

---

## 4. 参考代码

### 4.1 C# 核心模式（来自 `IRCameraView-master/IRCameraViewShared/CameraController.cs`，已裁剪）

```csharp
var groups = await MediaFrameSourceGroup.FindAllAsync();
var irGroup = groups.First(g => g.DisplayName.Contains("ASUS IR")); // 或按 SourceKind 过滤

var settings = new MediaCaptureInitializationSettings
{
    SourceGroup = irGroup,
    SharingMode = MediaCaptureSharingMode.ExclusiveControl,
    StreamingCaptureMode = StreamingCaptureMode.Video,
    MemoryPreference = MediaCaptureMemoryPreference.Cpu,
};

var mediaCapture = new MediaCapture();
await mediaCapture.InitializeAsync(settings);

var frameSource = mediaCapture.FrameSources.Values.First(); // IR 组里只有 Infrared 源
await frameSource.SetFormatAsync(frameSource.SupportedFormats.First());

var reader = await mediaCapture.CreateFrameReaderAsync(frameSource);
reader.AcquisitionMode = MediaFrameReaderAcquisitionMode.Realtime;
reader.FrameArrived += (sender, args) =>
{
    using var frameRef = sender.TryAcquireLatestFrame();
    var videoFrame = frameRef?.VideoMediaFrame;
    var bitmap = videoFrame?.SoftwareBitmap;                 // Gray8
    bool illuminated = videoFrame.InfraredMediaFrame.IsIlluminated;
    // …交给你的处理管线
};
await reader.StartAsync();
```

IRCameraView 过滤设备的方式（按帧源种类过滤）：

```csharp
foreach (var device in MediaFrameSourceGroup.FindAllAsync().AsTask().Result)
    foreach (var sourceInfo in device.SourceInfos)
        if (sourceInfo.SourceKind == MediaFrameSourceKind.Infrared)
            Devices.Add(device);
```

### 4.2 PowerShell 验证脚本（本机实测通过，主线程轮询取帧）

> Windows PowerShell 5.1。事件回调在 PS 里因 Runspace 限制常拿不到帧，**用轮询 `TryAcquireLatestFrame` 规避**（C# 无此问题）。

```powershell
Add-Type -AssemblyName System.Runtime.WindowsRuntime
# .AsTask() 桥接（WinRT IAsyncOperation/IAsyncAction → Task）
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() |
    ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() |
    ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' })[0]
function Await($op, $type) { $t = $asTaskGeneric.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1)|Out-Null; $t.Result }
function AwaitAction($op)  { $t = $asTaskAction.Invoke($null, @($op)); $t.Wait(-1)|Out-Null }

$groups = Await ([Windows.Media.Capture.Frames.MediaFrameSourceGroup]::FindAllAsync()) `
          ([System.Collections.Generic.IReadOnlyList[Windows.Media.Capture.Frames.MediaFrameSourceGroup]])
$irGroup = $groups | ? { $_.DisplayName -match 'ASUS IR' } | select -First 1

$s = New-Object Windows.Media.Capture.MediaCaptureInitializationSettings
$s.SourceGroup = $irGroup
$s.SharingMode = [Windows.Media.Capture.MediaCaptureSharingMode]::ExclusiveControl
$s.StreamingCaptureMode = [Windows.Media.Capture.StreamingCaptureMode]::Video
$s.MemoryPreference = [Windows.Media.Capture.MediaCaptureMemoryPreference]::Cpu

$mc = New-Object Windows.Media.Capture.MediaCapture
AwaitAction ($mc.InitializeAsync($s))

$src = $null
foreach ($kv in $mc.FrameSources) { if ($kv.Value.Info.SourceKind -eq 'Infrared') { $src = $kv.Value } }
$fmt = $null; foreach ($f in $src.SupportedFormats) { $fmt = $f; break }
AwaitAction ($src.SetFormatAsync($fmt))

$reader = Await ($mc.CreateFrameReaderAsync($src)) ([Windows.Media.Capture.Frames.MediaFrameReader])
$reader.AcquisitionMode = [Windows.Media.Capture.Frames.MediaFrameReaderAcquisitionMode]::Realtime
$st = Await ($reader.StartAsync()) ([Windows.Media.Capture.Frames.MediaFrameReaderStartStatus])

# 轮询取帧（替代 FrameArrived 事件）
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 150
    $f = $reader.TryAcquireLatestFrame()
    if ($f -and $f.VideoMediaFrame) {
        $sb  = $f.VideoMediaFrame.SoftwareBitmap          # Gray8 640x360
        $lit = $f.VideoMediaFrame.InfraredMediaFrame.IsIlluminated
        break
    }
}
AwaitAction ($reader.StopAsync()); $reader.Dispose(); $mc.Dispose()
```

### 4.3 Python（winsdk 移植示意 —— 未在本机实测）

Fylgja 是 Python 工程，可用 [`winsdk`](https://pypi.org/project/winsdk/)（Python WinRT 投影）按同一 API 链路移植。
注意 Python 3.14 的 wheels 可用性需先确认；不行则走 §7 的 C# 边车方案。

```python
# 示意：winsdk 方法名为 snake_case，异步返回可 await
import asyncio
from winsdk.windows.media.capture import (
    MediaCapture, MediaCaptureInitializationSettings,
    MediaCaptureSharingMode, MediaCaptureMemoryPreference, StreamingCaptureMode,
)
from winsdk.windows.media.capture.frames import (
    MediaFrameSourceGroup, MediaFrameReaderAcquisitionMode,
)

async def open_ir_camera():
    groups = await MediaFrameSourceGroup.find_all_async()
    ir_group = next(g for g in groups if "IR" in g.display_name)

    s = MediaCaptureInitializationSettings()
    s.source_group = ir_group
    s.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL
    s.streaming_capture_mode = StreamingCaptureMode.VIDEO
    s.memory_preference = MediaCaptureMemoryPreference.CPU

    mc = MediaCapture()
    await mc.initialize_async(s)

    src = next(v for v in mc.frame_sources.values()
               if v.info.source_kind == "infrared")
    await src.set_format_async(src.supported_formats[0])

    reader = await mc.create_frame_reader_async(src)
    reader.acquisition_mode = MediaFrameReaderAcquisitionMode.REALTIME
    await reader.start_async()
    return mc, reader
```

---

## 5. 关键参数速查

| 参数 | 值 | 不设的后果 |
|---|---|---|
| `SharingMode` | `ExclusiveControl` | 共享模式下 IR 流可能被 Hello 栈占用 |
| `MemoryPreference` | `Cpu` | `SoftwareBitmap` 可能为 null（GPU 缓冲） |
| `SetFormatAsync` | 在 CreateFrameReader **之前** | 格式未协商 |
| `AcquisitionMode` | `Realtime` | 帧投递节奏不稳 |
| 帧获取 | `TryAcquireLatestFrame()` | 直接用 `VideoMediaFrame.SoftwareBitmap`（Gray8） |
| 补光判断 | `InfraredMediaFrame.IsIlluminated` | 区分补光开/关帧 |

**实测格式**：640×360，subtype `L8`（= Gray8），15 FPS；补光严格交替 ⇒ 每种光照状态实际 ~7.5 FPS（IRCameraView 注释亦提及"halves framerate"）。

---

## 6. 坑位清单

1. **枚举不到 ≠ 不存在**：IR 摄像头在设备管理器/`Get-PnpDevice` 里一切正常（`Camera` 类、状态 OK），但所有常规相机 API 都枚举不到它。只能走 `MediaFrameSourceGroup`。
2. **Gray8 不能直接喂 PNG/JPEG 编码器**：`BitmapEncoder` 报 "bitmap pixel format is unsupported"。先 `SoftwareBitmap.Convert(bitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied)`（IRCameraView 的 `FrameArrived` 里也是这么做的）。
3. **PowerShell 事件回调拿不到帧**：`FrameArrived` 的 scriptblock 在非 PS 线程触发、没有 Runspace，回调静默失败。验证/脚本场景用主线程轮询 `TryAcquireLatestFrame`；正式集成用 C# 则无此问题。
4. **补光帧交替**：每帧有 `IsIlluminated` 标志，补光开帧更亮（适合特征跟踪）、补光关帧更暗。做视觉跟踪时建议只用一种（如 `IsIlluminated==True`），否则亮度每帧翻倍跳变。
5. **Direct MediaCapture 用设备符号链接 ID 打开会失败**：对 MI_02 的 `{e5323777…}\GLOBAL` 直接 `InitializeAsync` 报错；必须用 `SourceGroup` 方式初始化。
6. **`FrameSources` 在 PowerShell 里枚举别调 `.GetEnumerator()`**：WinRT 投影类型没有该方法，直接 `foreach ($kv in $mc.FrameSources)`。
7. **RGB 与 IR 同时开**：RGB 在 `ASUS FHD webcam` 组，IR 在 `ASUS IR camera` 组，可各开一个 `MediaCapture`；`Rts-DMFT-Group` 则一组内含 Color+Infrared 两个帧源（同步配对时用）。

---

## 7. 对 Fylgja 的接入建议

现状：`capture.py` 走 OpenCV（DShow → MSMF → 默认），**枚举不到 IR 摄像头**（`camera_enum.py` 的 DirectShow 枚举同理）。若要支持 IR，三条路：

| 方案 | 说明 | 评价 |
|---|---|---|
| A. `winsdk` 直接移植 | 在 Python 里按 §4.3 链路实现 `capture_ir.py` | 依赖 Python 3.14 的 winsdk wheels 可用性；WinRT 异步桥接到采集线程需小心 |
| B. C# 边车（推荐） | 裁剪 `CameraController.cs` 为无 UI 的帧推送工具（共享内存/named pipe/UDP 局部回环），Python 侧当普通帧源收 | 与 IRCameraView 同源、最稳；隔离崩溃域 |
| C. 改用 IRCameraView 全家桶 | 直接用/改 C# 应用 | 与现有 Python 引擎管线不符，仅作参考 |

若只是想"IR 补光下跟踪人脸"，还有一条零改动捷径：**用 RGB 摄像头 + 关房间灯 + IR 补光由 Hello 周期点亮**并不可行（补光只在 Hello 认证/本链路打开 IR 流时点亮）。Fylgja 想利用 IR 照明，必须真正打开 IR 流。

---

## 8. 参考资料

- `IRCameraView-master/`（同目录 zip 解压）：C# 完整参考实现，核心 `IRCameraViewShared/CameraController.cs`
  - 上游：https://github.com/Iemand005/IRCameraView
- Win32/Media Foundation 参考：https://github.com/andresbeltranc/ir_camera_sample_win32 （IRCameraView README 推荐）
- 本机验证样张：`ir_test_frame_raw.png` / `ir_test_frame_illuminated.png`
- 相关设备接口 GUID：
  - `KSCATEGORY_SENSOR_CAMERA` `{24e552d7-6523-47f7-a647-d3465bf1f5ca}`（IR 入口）
  - `KSCATEGORY_VIDEO_CAMERA` `{e5323777-f976-4f5b-9b55-b94699c46e44}`（常规相机，IR 不可见）
