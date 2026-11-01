# 开发交接（HANDOVER）

> 面向开发者的内部文档；用户视角见 [README](../README.md)。

## 当前状态（2026-11）

- **阶段 0**：门槛 2/3/5 完成（NPU 编译、求解器分轴对齐、60s×6 档长跑）；门槛 6 标定实验未做。数据见 `out/stage0/STAGE0_REPORT.md`。
- **基线全链路**：引擎 C 端到端 PASS；**引擎 N**（NPU）：求解器对齐达标、异步队列（深度 2）、自检归因齐备。
- **启动验证**：引擎选择矩阵（不静默回退）、6.1 预授权回退 + 横幅、摄像头枚举（PnP 路径/友好名）、单实例、崩溃钩子、配置持久化、日志。
- **GUI**：Win11 Fluent 双主题（跟随系统）、中英双语（右上角切换）、预览契约（坐标轴=发送值）、托盘后台。
- 回归测试 **54 项**通过。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.14.7（`.venv`；mediapipe 1.2.0 为 py3-none wheel，3.9–3.14 均可） |
| mediapipe | 1.2.0 |
| openvino | 2026.4.1（NPU 驱动 32.0.100.3717，低于文档下限 4511 但实测可用） |
| PySide6 | 6.12.0 |
| comtypes | 1.4.17（DirectShow 摄像头枚举） |
| 机器 | Intel Core Ultra 9 275HX + Intel AI Boost NPU（即目标机） |

## 关键坑位

1. **OpenTrack 字段顺序**：`TX,TY,TZ,yaw,pitch,roll`（位置在前），48 字节 little-endian double。勿改反（issue #1203 实测）。
2. **MediaPipe 遥测**：1.2.0 向 Google 发匿名指标（无关闭 API，issue #6291）；隐私声明须按引擎区分。
3. **`detect_for_video` 必须喂 `mp.Image`**，裸 ndarray 抛 `_image_ptr` AttributeError。
4. **VIDEO 时间戳**：严格单调递增 int，且须用真实步长（1000/fps）——MediaPipe 平滑/跟踪依赖真实 dt；self_check 已占用 0/1，采集基准 1000 起。
5. **BlazeFace 解码**：box 为 `[y,x,h,w]`（reverse_output_order），关键点为 **(x,y) 对**；输入 RGB（BGR 会把分数打到 0.4）；`fixed_anchor_size` 锚框 w=h=1.0；同 stride 层合并锚框组。
6. **求解器三处顺序**（face_geometry）：翻手性在**反投影后**、最终尺度 **s1×s2**、每次 Procrustes 前先翻手性。
7. **z 尺度公式**：`z_scale = side·√(cos²θ/w²+sin²θ/h²)`（随旋转角逐帧变化，LandmarkProjection.CalculateZScale）。
8. **裁剪约定**：478 全关键点 bbox ×1.5 长边方形 + 眼线角旋转（**像素空间**算角，归一化角会崩）+ [0,1] 归一化。
9. **QSS `*` 规则会吞字体回退**：`font-family` 必须写单一族"Microsoft YaHei UI"；离屏 QPA 字体库为空（截图豆腐字是环境问题）。
10. **IMoniker vtable** 要完整占位；`BindToStorage` 返回裸指针要 `ctypes.cast`；枚举测试必须逐帧同步喂信箱。
11. **测试一律 NPU**（用户约定）：引擎 N 相关脚本 `device="NPU"`，不得图省事用 CPU。

## 复跑命令

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests        # 54 项回归
.\.venv\Scripts\python.exe scripts\gate3_alignment.py           # 门槛 3（NPU）
.\.venv\Scripts\python.exe scripts\longrun_test.py --engine N --speed 1 --duration 60
.\.venv\Scripts\python.exe scripts\analyze_video.py test.mp4    # 视频解析
.\.venv\Scripts\python.exe scripts\annotate_pose.py --engine C --both
```

## 下一步

1. 清 🟡 缺口：引擎 N 分项耗时入状态栏、固定参考系叠加、帮助/关于面板、标定 UI、快捷键可配。
2. 标定实验（4.5#6）+ 2h/60min 稳定性挂机。
3. PyInstaller 6.x 打包 + collect_licenses + FIRST-RUN.md + 干净系统验收。
4. 自启动计划任务（10）+ CI + 仓库发布。
5. 面部注册（7.5，预留规格已入方案书）。
