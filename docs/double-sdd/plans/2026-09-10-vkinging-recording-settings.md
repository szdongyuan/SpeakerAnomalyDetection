# Vkinging Recording Settings Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 VK 设备选择合并到驱动与设备列表，按测试队列保存采样率和七档量程，在高级设置中集中录制选项，删除录音监听，并在修改共享队列前列出受影响工况。

**Architecture:** 设备发现只负责设备身份和通道，纯配置解析器把当前队列参数合成实际采集快照。采集资源复用比较完整硬件参数，校准适用性忽略合法采样率／量程变化；共享队列引用扫描与 UI 保存确认分离。

**Tech Stack:** Python 3.12、PyQt5、pytest、NumPy、soundfile、sounddevice、ctypes VK SDK、现有独立录制进程与 JSON 配置。

---

## 约束、执行与文件责任

规格：`docs/double-sdd/specs/2026-09-10-vkinging-recording-settings-design.md`。它是唯一产品行为来源，特别注意 §4.4 共享引用和取消父草稿场景。量程校准不失效、参数保存在队列中、四项控件全部位于高级设置，是已确认决策。

执行前由 `subagent-driven-development` 准备隔离工作树，遵守其 metadataPath 上游交接方式；元数据路径只经调度传递，不写入此持久化文档。不得覆盖其他任务改动，不修改 `.agents/` 或 `.codex/`。按任务顺序分配新 implementer，每任务结束经 spec-code-reviewer 和 quality-code-reviewer 审查后再继续。任务检查点只提交该任务文件，最后通过 finishing-a-development-branch 保留最终内容并清理临时提交；没有用户另行批准，不留下永久实现提交。

在执行工作树根目录使用 PowerShell，先设置 `$env:QT_QPA_PLATFORM = 'offscreen'`。命令均从该根目录执行，使用已配置的 Python 3.12 环境。环境检查：

```powershell
python -c "import sys, pytest, numpy, soundfile; from PyQt5 import QtCore; print(sys.version); print(pytest.__version__); print(QtCore.QT_VERSION_STR)"
git status --short
```

测试使用已有 fake SDK 和临时 JSON，禁止在自动测试中连接真实设备、依赖用户配置或修改用户校准库。若隔离工作树缺少忽略资源，按执行技能配置依赖；不得把无关本地资源加入 Git。UI 实际行为测试优先于源码字符串匹配。

主要文件分工：

| 文件／组 | 责任 | 任务 |
| --- | --- | --- |
| `consts/ve3668n_consts.py`、`base/ve3668n_input.py` | 参数、量程映射、严格校验、资源签名 | 1、3 |
| 新 `base/ve3668n_recording_config.py` | 队列优先参数解析，不访问 SDK | 1、7 |
| `base/ve3668n_stores.py`、`base/ve3668n_wav_metadata.py` | 校准适用性、七档文件元数据 | 2 |
| `base/vkinging_sdk.py`、`base/ve3668n_capture.py`、`base/ve3668n_resource.py`、进程协议／服务 | 参数实际下发、复用／释放 | 3 |
| 录制协议、采集器、`base/streaming_audio_processor.py`、`base/play_and_record.py` | 删除录音监听，保留录音及独立播放 | 4 |
| `ui/hardware_window.py`、`ui/ve3668n_hardware_controls.py`、`base/hardware_selection.py` | 统一设备入口、恢复和取消 | 5 |
| `ui/acquisition_config_window.py` | 参数编辑、高级设置、验证 | 4、6 |
| `main_window.py`、`ui/sequence/` 调用方、`ui/calibration_window.py` | 实际队列参数、预热准入、校准请求 | 7 |
| 新 `base/sequence_queue_references.py` | 已登记产品与父草稿引用查询 | 8 |
| `ui/operation_sequence.py`、新 `ui/shared_queue_save_dialog.py`、两个产品编辑器 | 共享队列保存与影响提示 | 6、9 |
| `unit_test/base/`、`unit_test/ui/`、现有产品配置测试 | 各层回归与端到端行为 | 1–10 |

每项任务采用红→绿→回归顺序。先运行新增测试确认失败原因是当前行为不满足目标，不能把环境错误当红灯。实现后用同一命令确认通过，随后运行列出的回归。每项末尾先 `git diff --check` 再 `git add -- <该任务实际文件>` 和 `git commit -m '<该任务消息>'`，检查退出码后记录检查点。禁止 `git add .`，不能为了让测试绿而删除仍应保留的所有权、I/O、裁剪或播放测试。

异常／状态审查贯穿各任务：参数错误使用有字段含义的 ValueError；SDK 错误保留操作及错误码；文件和 UI 边界只处理可解释的错误，不静默改参数或把查询失败当无引用。不增加无理由 broad catch、重复校验包装或全局可变状态。跨文件 schema 字段／枚举位于 `consts`；单次模块级常量仅允许规格列出的稳定边界例外。

## Task 1: 采样率、量程合同与队列参数解析

**Files:**
- Modify: `consts/ve3668n_consts.py`、`base/ve3668n_input.py`
- Create: `base/ve3668n_recording_config.py`、`unit_test/base/test_ve3668n_recording_config.py`
- Test/Modify: `unit_test/base/test_ve3668n_input.py`、`unit_test/base/ve3668n_fakes.py`

- [x] **Step 1：写纯合同红灯测试。** 测试 8000、102400、32000、96000 及原三档；非法输入为 7999、102401、48000.5、True、字符串、None。量程编号必须严格整数 0–6；对称上下限组合只能属于七档，拒绝不对称、未列出、NaN／Inf。解析时队列值覆盖设备值，缺失与非法分开，输入字典不被修改。

```python
import pytest
from base.ve3668n_recording_config import resolve_ve_recording_config
from unit_test.base.ve3668n_fakes import input_config

def test_queue_values_override_profile_without_mutation():
    detail = {"sample_rate": 96000, "ve_range_index": 4}
    profile = input_config(51200)
    result = resolve_ve_recording_config(detail, fallback_profile=profile)
    assert result == input_config(96000, range_min=-0.5, range_max=0.5)
    assert detail == {"sample_rate": 96000, "ve_range_index": 4}
    assert profile == input_config(51200)

@pytest.mark.parametrize("bad", [-1, 7, True, 1.5, "1", None])
def test_explicit_bad_range_is_not_defaulted(bad):
    with pytest.raises(ValueError, match="ve_range_index"):
        resolve_ve_recording_config({"sample_rate": 48000, "ve_range_index": bad})
```

- [x] **Step 2：运行红灯。** `python -m pytest unit_test/base/test_ve3668n_recording_config.py unit_test/base/test_ve3668n_input.py -q`；预期新 helper 缺失或原白名单行为导致新增断言失败。
- [x] **Step 3：实现合同和解析。** 添加稳定常量 `VE_SAMPLE_RATE_MIN=8000`、`VE_SAMPLE_RATE_MAX=102400`、`VE_RANGE_INDEX_CONFIG_KEY="ve_range_index"`、`VE_RANGE_LIMITS=(10.0,5.0,2.5,1.0,0.5,0.1,0.02)`、对应七个 UI 文本。默认 ±10 V 常量保留供历史默认值使用；原三档元组仅可作常用建议，不再用于合法性白名单。新增 `validate_range_index`／`voltage_limits_for_range`；`validate_input_config` 保持封闭 schema 与 IEPE/V 检查，按合法对称范围校验电压。`create_input_config(sample_rate=51200, *, range_index=0)` 沿用旧调用默认值。

解析器的完整核心实现形态（导入项目常量与校验函数，返回配置的所有权属于调用者）：

```python
from collections.abc import Mapping
from base.ve3668n_input import create_input_config, validate_sample_rate, validate_range_index
from consts.ve3668n_consts import VE_DEFAULT_SAMPLE_RATE, VE_RANGE_INDEX_CONFIG_KEY

def resolve_ve_recording_config(detail, *, fallback_profile=None):
    if not isinstance(detail, Mapping):
        raise ValueError("recording detail must be a mapping")
    if "sample_rate" in detail:
        rate = validate_sample_rate(detail["sample_rate"])
    elif fallback_profile is None:
        rate = VE_DEFAULT_SAMPLE_RATE
    else:
        if not isinstance(fallback_profile, Mapping) or "sample_rate" not in fallback_profile:
            raise ValueError("fallback profile requires sample_rate")
        rate = validate_sample_rate(fallback_profile["sample_rate"])
    index = validate_range_index(detail.get(VE_RANGE_INDEX_CONFIG_KEY, 0))
    return create_input_config(rate, range_index=index)
```

显式合法队列采样率不读取无关设备档案；需要设备档案时的 I/O 由第 7 项调用方负责，文件缺失才能默认，损坏不静默默认。此任务不更改 `resolve_effective_input_rate` 的旧调用行为，待第 7 项迁移所有录制调用方。
- [x] **Step 4：同命令绿灯并回归。** 再运行 `python -m pytest unit_test/base/test_ve3668n_discovery.py unit_test/base/test_ve3668n_hardware_selection.py -q`；更新仅因旧白名单而失效的测试期望，不放宽其他结构错误。所有上述命令退出码为 0。
- [x] **Step 5：检查差异、提交及审查。** 消息 `feat: define VK queue sample rate and range contracts`。审查重点：缺失／非法分离、映射编号不是电压、普通声卡合同不被扩大。

## Task 2: 校准跨量程有效与 WAV 七档元数据

**Files:**
- Modify: `base/ve3668n_input.py`、`base/ve3668n_stores.py`、`base/ve3668n_wav_metadata.py`、`base/recording_capture.py`
- Test/Modify: `unit_test/base/test_ve3668n_stores.py`、`unit_test/base/test_ve3668n_wav_metadata.py`、`unit_test/base/test_recording_calibration_snapshot.py`、`unit_test/base/test_recording_capture.py`、`unit_test/base/test_ve3668n_capture.py`

- [x] **Step 1：写状态及文件红灯测试。** 保存一次 measured 校准，依次观察七个合法范围与多个采样率，断言 `get_factor` 数值、valid 状态及校准 JSON 字节不变；保持单位／模式变化、显式 reset、历史 invalidated 和 I/O 失败测试。向 FLOAT WAV 附加每档量程元数据并读取，断言电压样本相等、范围正确、旧 ±10 V 文件可读。
- [x] **Step 2：运行。** `python -m pytest unit_test/base/test_ve3668n_stores.py unit_test/base/test_ve3668n_wav_metadata.py -q`；预期范围变化触发旧 sticky invalidation、非 ±10 V 元数据被拒绝。
- [x] **Step 3：分离适用性与历史指纹。** 保留 `calibration_fingerprint` 的历史字段及严格结构验证，新增适用性键用于 `VECalibrationStore.observe` 的有效系数判断。只对已完成验证的指纹做以下投影，不移除正在进行的校准请求的完整快照比较：

```python
def calibration_applicability_key(fingerprint):
    return tuple(fingerprint[name] for name in (
        "backend", "model", "machine_id", "physical_channel", "input_mode", "unit"
    ))
```

保留失效记录和 pending invalidation 的原有所有权／重试规则，合法量程差异不进入失效集合。WAV 校验保持历史采样率为正整数的既有文件规则，不用新的采集上下限拒绝历史文件；模式与单位依然严格，范围使用七档校验。元数据构造仍读取本次 request 快照。

按规格 §7.1，仅将录音质量检测调用前的 VK `audio / 10.0` 改为 `audio / req.device["input_config"]["range_max"]`，前提是 request 已验证；在这一处单独转换，不修改写盘、波形或 Pa/V 数据。以同一电压及不同 range_max 断言传入质量检测器的数组，并另断言 WAV 数组没有变化。
- [x] **Step 4：绿灯及回归。** 重跑 Step 2，再运行 `python -m pytest unit_test/base/test_recording_calibration_snapshot.py unit_test/base/test_recording_capture.py unit_test/base/test_ve3668n_capture.py unit_test/ui/test_ve3668n_calibration.py -q`。旧不对称范围错误仍拒绝；不以允许七档为由接受任意区间。
- [x] **Step 5：检查、提交、审查。** 消息 `fix: retain VK calibration across valid range changes`。审查重点：不变的有效校准文件、文件历史兼容、未改变样本，以及完整请求所有权验证仍存在。

## Task 3: SDK 范围下发与完整资源签名

**Files:**
- Modify: `base/vkinging_sdk.py`、`base/ve3668n_capture.py`、`base/ve3668n_resource.py`、`base/ve3668n_input.py`、`base/recording_process_protocol.py`、`base/recording_service.py`、`base/recording_worker.py`、`base/ve3668n_prewarm_lifetime.py`
- Test/Modify: `unit_test/base/ve3668n_fakes.py`、`unit_test/base/test_vkinging_sdk.py`、`unit_test/base/test_ve3668n_capture.py`、`unit_test/base/test_ve3668n_resource.py`、`unit_test/base/test_ve3668n_protocol.py`、`unit_test/base/test_ve3668n_service.py`、`unit_test/base/test_recording_ve_release.py`、`unit_test/ui/test_ve3668n_prewarm_lifetime.py`、`unit_test/ui/test_ve3668n_prewarm_trigger.py`

- [x] **Step 1：写 SDK／生命周期红灯。** 对七档捕获 `VkDaqCreateAIAccelChan` min/max 参数；同设备同通道同率不同量程签名不等、相同量程相等。覆盖单次采集和常驻资源，旧资源先 stop/clear/close 后新建，释放失败不启动新任务，实际采样率读回不符不发布 started。
- [x] **Step 2：运行。** `python -m pytest unit_test/base/test_vkinging_sdk.py unit_test/base/test_ve3668n_resource.py unit_test/base/test_ve3668n_protocol.py -q`；预期固定 ±10 V 与旧四项签名失败。
- [x] **Step 3：实现参数贯穿。** `create_iepe_voltage_channel(task, physical_channels, *, range_min=-10.0, range_max=10.0)` 校验后把真实 V 值传入原 SDK 两个范围实参；两个生产采集调用方显式传入 request 配置，保留默认只为既有直接调用，不能生产遗漏参数。同步扩充 fake SDK 方法和 trace。

所有资源签名统一为：

```python
("vkinging", machine_id, ordered_channels, sample_rate,
 "IEPE", "V", range_min, range_max)
```

`ve_acquisition_signature` 验证请求采样率与设备快照一致。`_acquisition_signature` 校验八项并原样规范化；`VePrewarmRequest.signature` 调用同一个构造函数，不能手写四元组。保持采样率仍位于下标 3；搜索并更新四项解包、长度断言、序列化事件、release outcomes、服务／worker 跟踪、失败签名屏蔽以及测试常量。复用服务已有 start 时不匹配资源释放机制，不在 UI 开第二条资源管理通路。
- [x] **Step 4：绿灯和全链回归。** 重跑 Step 2；再运行 `python -m pytest unit_test/base/test_ve3668n_capture.py unit_test/base/test_ve3668n_service.py unit_test/base/test_recording_ve_release.py unit_test/ui/test_ve3668n_prewarm_lifetime.py unit_test/ui/test_ve3668n_prewarm_trigger.py -q`。不能扩大预热次数或用 sleep 掩盖 race。
- [x] **Step 5：检查、提交、审查。** 消息 `feat: apply VK ranges through native resource lifecycle`。审查生产两条创建路径、消息签名同构与释放所有权。

## Task 4: 删除录音监听功能及两个控件

**Files:**
- Modify: `base/recording_process_protocol.py`、`base/recording_capture.py`、`base/streaming_audio_processor.py`、`base/play_and_record.py`、`base/recording_settings.py`、`ui/acquisition_config_window.py`、`ui/sequence/sequence_widget_analysis_ops.py`、`ui/sequence/sequence_widget_recording_process_ops.py`
- Test/Modify: `unit_test/base/test_recording_capture.py`、`unit_test/base/test_recording_startup_trim.py`、`unit_test/base/test_recording_service.py`、`unit_test/ui/test_streaming_recording_config.py`、`unit_test/ui/test_ve3668n_recording.py`

- [x] **Step 1：写拒绝激活旧功能的红灯测试。** 普通声卡／VK 的旧配置带 monitor=true、无输出设备、损坏的历史 gain/fade 值，也应录音且不创建 duplex/output monitor 流，streaming=false 不产生实时预览。添加 UI 两控件不存在的断言；保留 startup trim 的样本及时间规则测试。
- [x] **Step 2：运行。** `python -m pytest unit_test/base/test_recording_capture.py unit_test/ui/test_streaming_recording_config.py -q`；预期旧监听输出或现有控件断言失败。
- [x] **Step 3：删除行为，不只隐藏 UI。** 删除 capture 的监听 callback、render、gain/mute 状态及 output stream 分支；保留输入 callback 的错误上报边界。删除 legacy `StreamingAudioProcessor` 中监听分支和仅用于监听的函数、参数、字段，并更新 `play_and_record` 调用方。请求 `monitor` 字段可保留为空 Mapping 的读取兼容壳：在请求构造边界规范化为 `{}`，生产调用方传 `{}`，删除其 enabled/device/gain 校验与所有执行分支。

```python
@property
def effective_streaming(self):
    return self.purpose == "main" and self.streaming
```

删除 UI 监听、增益、VE 监听说明及保存字段；当前流式 label 暂在下一 UI 任务统一改布局。配置 resolver 忽略旧监听字段，`_should_use_streaming_recording` 只读 `use_streaming_recording`。起始裁剪必须保留：录制内部用 `startup_trim_samples` 明确传递并填充 `RecordingRequest.trim_samples`；如旧入口依赖 `monitor_mute_leading_samples` 表示裁剪，只在记录字典入口转换为裁剪值，不保留监听数学。移除无调用者的 `resolve_monitor_fade_in_ms/samples`，不删除普通激励／回放路径或名为 video_monitor 的 UI。
- [x] **Step 4：绿灯和回归。** 重跑 Step 2；`python -m pytest unit_test/base/test_recording_startup_trim.py unit_test/base/test_recording_service.py unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_original_recording_analysis_route.py -q`。使用 `rg -n 'monitor_playback|monitor_gain_db|apply_monitor_startup_mute|resolve_monitor_fade' base ui` 审查剩余引用只能为明确的历史字段兼容或无关用途，不用简单零匹配替代行为测试。
- [x] **Step 5：检查、提交、审查。** 消息 `refactor: remove recording monitor playback`。审查没有误删录音裁剪或独立播放，直接进程请求也无法重新开启监听。

## Task 5: 统一驱动、设备及通道选择

**Files:**
- Modify: `ui/hardware_window.py`、`ui/ve3668n_hardware_controls.py`、`base/hardware_selection.py`
- Test/Modify: `unit_test/ui/test_ve3668n_hardware.py`、`unit_test/ui/test_ve3668n_persistent_hardware.py`、`unit_test/base/test_ve3668n_hardware_selection.py`

- [x] **Step 1：用现有 Qt fixture 和受控发现 fake 写红灯。** 驱动包含 vkinging；不存在额外 backend/device/rate combo；选中驱动后 mic 表显示设备，初始不选，设备选中前无通道；取消设备、离线、刷新、切驱动、取消对话框、busy 与迟到回调均不会泄漏旧选择。
- [x] **Step 2：运行。** `python -m pytest unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_persistent_hardware.py -q`；预期仍需旧后端下拉框，输入列表被禁用。
- [x] **Step 3：拆除 VK 小面板并保留发现职责。** 重构 `VE3668NHardwareControls` 为该模块内专用 QObject 控制器（可以改名 `VE3668NDeviceSelection`），提供后端切换、refresh、select_device、inventory_changed、input_changed、status_changed、set_busy/close；无 UI combo 或采样率草稿。Controller 对驱动条目使用 itemData 分辨普通 API 与 VK，不能把 vkinging 传给 sounddevice hostapi。发现事件填充 `SingleCheckTableView`，通道由当前已选且可用的设备生成，明确禁用未选通道。

硬件选择保存只写设备／通道及原声卡恢复信息，不调用 `set_sample_rate` 或把量程写回 profile。保留现有有效默认档案读取、MachineId 恢复、取消快照、通道顺序和独立输出选择；busy 由现有单一检查源决定。把运行中错误移到简短状态／对话框，删除静态固定范围说明。
- [x] **Step 4：绿灯、存储回归。** 同 Step 2，加 `python -m pytest unit_test/base/test_ve3668n_hardware_selection.py unit_test/base/test_ve3668n_discovery.py -q`。硬件确定前后比较 profile 文件字节；设备选择取消不写任何设置。
- [x] **Step 5：检查、提交、审查。** 消息 `feat: select VK devices through hardware driver list`。审查选择恢复、取消与异步生命周期，不能以自动选第一台设备绕过通道前置条件。

## Task 6: 录制参数输入与默认折叠的高级设置

**Files:**
- Modify: `ui/acquisition_config_window.py`、`ui/operation_sequence.py`
- Create: `unit_test/ui/test_recording_advanced_settings.py`
- Test/Modify: `unit_test/ui/test_recording_storage_config.py`、`unit_test/ui/test_recording_preview_mode_config.py`、`unit_test/ui/test_streaming_recording_config.py`

- [x] **Step 1：写 Qt 交互红灯。** 主区只含时长／采样率／输入设备，高级区初始隐藏；四项都归属高级容器。range QComboBox 正好七项、标签及 data 正确；显示模式跟随 realtime toggle，折叠不会关掉 enabled 状态。保存回传 sample_rate/range_index，不写 monitor；取消无结果。高级目录或模式错误可展开并定位。
- [x] **Step 2：运行。** `python -m pytest unit_test/ui/test_recording_advanced_settings.py unit_test/ui/test_recording_preview_mode_config.py -q`；预期缺少高级容器／可编辑 VK 采样率。
- [x] **Step 3：实现容器和显式提交校验。** 用 QToolButton 的 checkable 折叠入口加 QWidget 容器，初始 checked=false；只连接容器可见性，不连到 realtime checkbox。给测试稳定 objectName：`recording_advanced_toggle`、`recording_advanced_panel`、`ve_range_combo`。四项顺序按规格。量程 addItem(label,index)，setCurrentIndex 使用 findData。普通声卡不显示范围，保留其原可选采样率。

VK 采样率采用可输入整数且能显示损坏旧值的控件：可编辑 QComboBox 配合 QIntValidator(8000,102400)，提供常用建议，不在载入时 clamp。确认时独立严格校验文本整数及边界并调用 Task 1 resolver；错误保留窗口，不保存。现有模式错误修复逻辑保留，但所有模式字段只能出现在高级面板；confirm 时展开显示错误。用基于 input_data 的副本保留其他录制字段，再更新本次字段并删除历史监听键，避免意外丢失质量阈值／裁剪配置。

`OptionList.show_dialog` 接收 RecordConfigWindow 结果后更新当前队列草稿及 signal_len，并通过已有 change notifier 通知保存入口；不能直接在 RecordConfigWindow 写文件或设备 profile。第 9 项会对共享队列接管这个保存入口。
- [x] **Step 4：绿灯和界面回归。** 重跑 Step 2，再运行 `python -m pytest unit_test/ui/test_recording_storage_config.py unit_test/ui/test_streaming_recording_config.py -q`。Qt 离屏分别构建普通声卡／VK、展开／折叠、模式修复四类状态并抓图检查布局，截图保存在未提交的工作树输出目录。
- [x] **Step 5：检查、提交、审查。** 消息 `feat: group recording options in advanced settings`。不为单纯文本镜像写测试，重点验证字段归属、状态保存和错误可达性。

## Task 7: 队列参数贯穿主窗口、录制、预热及校准

**Files:**
- Modify: `base/ve3668n_recording_config.py`、`ui/sequence/sequence_widget_analysis_ops.py`、`ui/sequence/sequence_widget_recording_process_ops.py`、`ui/sequence/sequence_widget_config_ops.py`、`main_window.py`、`ui/calibration_window.py`
- Test/Modify: `unit_test/ui/test_ve3668n_recording.py`、`unit_test/ui/test_ve3668n_calibration.py`、`unit_test/ui/test_ve3668n_prewarm_trigger.py`、`unit_test/ui/test_ve3668n_prewarm_lifetime.py`
- Create: `unit_test/ui/test_ve3668n_queue_parameters.py`

- [x] **Step 1：写实际工作流红灯。** 不同队列 A/B 分别 48000/±5 V、96000/±0.1 V，同设备切换后 request、frames、trim、preview、WAV metadata 一致。两个工况引用 A 时参数完全相同；profile 文件不变。已保存 rate 优先于设备 51200。预热准入和校准使用当前队列值，历史校准系数不变。
- [x] **Step 2：运行。** `python -m pytest unit_test/ui/test_ve3668n_queue_parameters.py unit_test/ui/test_ve3668n_recording.py -q`；预期原 reset_work_pram 重新加载设备 profile 覆盖队列值。
- [x] **Step 3：统一快照构造。** 在 Task 1 模块新增 `resolve_ve_recording_device(device, detail, *, fallback_profile=None)`：验证设备 Mapping，用 resolver 的配置替换副本 input_config 后 `validate_device_snapshot`。主录制进入 reset_work_pram 在计算采样点及 LoadUiConfig 参数前生成快照；不直接读共享 profile 覆盖显式值。缺率 fallback 由已注入 profile store 获取，使用已有 I/O 错误而非 broad catch。

主窗口以 sequence_window 的当前队列 detail 构造 `_ve_signature`／预热请求和 calibration admission；没有已加载队列的启动预热使用明确默认来源，不把默认写入队列。Task 3 的 service 已负责不同签名资源先释放后重配，UI 只提供正确 request，不阻塞等待。`_ve_prewarm_admission_available` 也必须用同一有效快照，不能用 self.mic 的旧 input_config。

打开 CalibrationWindow 时传入当前有效配置及显式 queue config/provider（新增可选 keyword-only 参数，独立入口可省略）；InputCalibration `_current_ve_device` 在该来源存在时不能再用 profile.load 覆盖。进行中的 request 冻结完整配置，并保留原 stale request/generation 检查；有效系数显示通过 Task 2 的适用性规则。序列加载与手动／串口工况切换依然读取现有队列文件，无任何 condition override 字段。
- [x] **Step 4：绿灯及回归。** 重跑 Step 2，加 `python -m pytest unit_test/ui/test_ve3668n_calibration.py unit_test/ui/test_ve3668n_prewarm_trigger.py unit_test/ui/test_ve3668n_prewarm_lifetime.py unit_test/ui/test_original_recording_analysis_route.py -q`。覆盖 invalid explicit rate、profile I/O failure、设备离线及资源 busy，不以默认值吞掉错误。
- [x] **Step 5：检查、提交、审查。** 消息 `feat: apply active queue VK parameters to recording workflows`。搜索 `resolve_effective_input_rate`／`profiles.load`／手写 signature，确认所有生产录制与准入入口已统一。

## Task 8: 共享队列引用查询器

**Files:**
- Create: `base/sequence_queue_references.py`、`unit_test/base/test_sequence_queue_references.py`
- Read/reuse: `base/product_test_project_config.py`、`base/product_test_program_config.py`、`consts/product_test_project_consts.py`、`consts/running_consts.py`

- [x] **Step 1：写纯文件查询红灯。** 临时注册表包含新格式 project/test_groups/test_conditions 和旧格式 name/sub_configs，查询包含当前及非当前产品。不同别名指同一实际文件、相对绝对路径、Windows 大小写匹配；两个同名工况不同分组不得去重。文件缺失／损坏、登记表读取失败进入 issues，查询完全不写盘。

必须测试已保存 A/B→Q，草稿 B→R：Q 返回 A 和 B（B 标注 saved-only）；草稿新增 C→Q 也返回；同位置相同引用只计一个工况。父草稿无文件名的新产品采用该 draft 的实例身份，不同新产品不得误合并。
- [x] **Step 2：运行。** `python -m pytest unit_test/base/test_sequence_queue_references.py -q`；预期查询模块缺失。
- [x] **Step 3：实现只读服务和结果。** 构造函数注入 product_dir、product_registry_path、queue_registry_path；公开 `find_references(target_path, *, drafts=()) -> QueueReferenceResult`。冻结结果项含产品路径／显示名、group 与 condition 的位置及显示名、来源集合；result 含 references tuple 与 issues tuple。对同一产品位置联合 saved/draft 来源，保留必要的改名前后名称文本，不用显示名称作唯一键。

路径比较使用：

```python
import os

def queue_path_key(path, registry_dir):
    candidate = path if os.path.isabs(path) else os.path.join(registry_dir, path)
    return os.path.normcase(os.path.realpath(os.path.abspath(candidate)))
```

生产 Windows 大小写由 normcase 处理；使用现有队列路径迁移／解析约定与 catalog 一致。直接按读取结果处理已登记文件，避免调用旧 ProductTestProgramConfigManager.load_registry 在坏文件时 rebuild 写盘。使用现有 LoadUiConfig 的 error code 或限定 JSON/OSError 边界，不能将失败当空字典。新旧产品 schema 只读适配；复用项目 `iter_test_conditions`，校验不可遍历的结构并记录 diagnostic。不要扫描用户整个磁盘或未登记的无关 JSON。
- [x] **Step 4：绿灯及回归。** 重跑 Step 2；`python -m pytest unit_test/test_product_test_program_config.py unit_test/test_product_test_project_config.py -q`。用写函数 spy 和文件字节证明查询无隐式 registry repair 或父草稿保存。
- [x] **Step 5：检查、提交、审查。** 消息 `feat: find shared queue references across product conditions`。审查 persisted∪draft、身份去重、路径别名及读失败不静默。

## Task 9: 共享队列保存提示与草稿边界

**Files:**
- Create: `ui/shared_queue_save_dialog.py`、`unit_test/ui/test_shared_queue_save.py`
- Modify: `ui/operation_sequence.py`、`ui/product_test_project_config_dialog.py`、`ui/product_test_program_config_dialog.py`、`main_window.py`
- Test/Modify: `unit_test/test_product_test_project_config_dialog.py`、`unit_test/test_product_test_program_config_dialog.py`

- [x] **Step 1：写实际保存路径红灯。** 使用真实临时队列、Task 8 查询器和注入的确认回调。共享队列 change notifier 不写；OK 取消不写且草稿保留，确认恰写一次；无变化不提示。另存为新文件不提示旧引用，覆盖另一共享文件展示新目标引用。直接队列入口和产品入口结果一致；长列表全部可查看。

关闭草稿保存／放弃／取消关闭分别验证：保存再次走影响提示，放弃保持原文件，取消不关闭；切换打开／新建目标前同样处理未保存草稿。查询失败禁止自动写，显式提示已知引用和损坏文件后可确认或取消。测试现有自动保存不会先写入再弹窗。
- [x] **Step 2：运行。** `python -m pytest unit_test/ui/test_shared_queue_save.py -q`；预期现有 `_persist_current_config_silently` 提前写入。
- [x] **Step 3：统一保存网关。** 在 AnalysisModelSelect 新增实例字段保存 last_persisted_payload、dirty、查询器、父 draft provider 和一次性的 close 再入守卫。比较规范化 JSON 对象而非格式文本决定是否真正变化；payload 使用副本，`format_config_data` 不可在比较时提前改写原草稿。

所有磁盘写入入口 `_persist_current_config_silently`、`ok_btn_clicked`、`save_btn_clicked` 收敛到 `_save_queue(target_path, *, explicit)`：

```text
校验并生成候选 payload → 读取目标当前内容 → 若相同则无需写入或警告
→ 重新查询实际目标引用与父草稿
→ shared(>=2) 或查询不完整 且非 explicit：保留 dirty 草稿，返回 deferred
→ shared 或查询不完整 且 explicit：显示带完整列表的确认
→ 取消：返回 cancelled，不写队列、注册表、不关闭
→ 接受／无需提示：写队列，成功后更新所需注册表和 persisted snapshot
→ 写失败：显示错误、保留 dirty、不发布成功
```

采用规格 §4.4 的准确中文提示；专用 QDialog 用滚动文本或列表，按钮为“保存并影响以上工况”“取消”，查询不完整时改为能表明未知影响的继续保存文案。QMessageBox 的常规保存／放弃／取消用于关闭未保存草稿；不要因 closeEvent 再入弹两次影响警告。`ok_btn_clicked` 原默认配置注册表修改必须推迟到用户允许保存之后。

父窗口只传 provider，不先保存父产品：在两个产品编辑器新增独立的可选 contextual queue-editor callback/provider 接口，主窗口接入后传当前文件身份及 collect_project/collect_program 草稿；旧单参数 queue_editor_callback 保留作为无上下文回退，不用 TypeError 捕获探测参数个数。查询同时保留磁盘引用，不能用 draft 替代。父窗口当前组的未提交表格编辑应在生成草稿时读入内存，但不落盘。
- [x] **Step 4：绿灯及回归。** 重跑 Step 2；`python -m pytest unit_test/test_product_test_project_config_dialog.py unit_test/test_product_test_program_config_dialog.py unit_test/ui/test_operation_sequence_analysis_channel_mode.py unit_test/ui/test_recording_preview_mode_config.py -q`。特别核对共享 draft 与父窗口取消的组合，以及失败时 registry 和队列都未被错误更新。
- [x] **Step 5：检查、提交、审查。** 消息 `feat: warn before saving queues shared by conditions`。审查每个保存／覆盖／切换目标／关闭路径都进入同一网关，不扩张为修改所有配置存储系统。

## Task 10: 综合验收、回归与交付证据

**Files:**
- Create: `unit_test/ui/test_vkinging_settings_workflow.py`
- Modify: 仅前序任务引入问题对应的文件与测试；若出现独立缺陷，先按 systematic-debugging 定位，不做无关修复。

- [x] **Step 1：写完整用户路径集成场景。** 假设备发现→右上角 VK→选设备→选通道→队列高级设置选择 rate/range/realtime/mode→保存共享队列提示→取消及确认→切工况→实际 fake-SDK 录制→读取 WAV 电压／元数据／校准。另一用例折叠高级设置但已启用 realtime，必须仍有预览；不同量程采集沿用原校准。

最终全量审查补充回归：启动发现仅确认物理设备后，缺失采样率的队列必须在参数窗口与实际录音中采用同一个设备档案回退值；覆盖非默认档案采样率、直接队列入口及产品工况入口。打开并确认窗口不能意外把发现默认值写入队列，仍须允许修复显式非法参数并保留档案读取失败诊断。

硬件参数归属补充回归：硬件窗口恢复已选设备和手动选择新设备时，物理可用性均不依赖采样率回退档案。档案不可读或损坏不能阻止选择实际发现的设备、通道及确认硬件；错误应在确实需要读取档案的参数回退入口报告。
- [x] **Step 2：运行新增测试确认能捕获真实连接缺口。** `python -m pytest unit_test/ui/test_vkinging_settings_workflow.py -q`。如果首次即绿，记录它是集成验证，不伪造 red；对发现的缺口逐个补回归红灯再修复。
- [x] **Step 3：执行完整相关回归并记录实际输出。** 先跑下列集合一次；失败定位到任务所有者，修复后只重跑受影响集合，最后补齐未执行集合。不能把 skipped/xfailed 当成覆盖成功。

```powershell
python -m pytest unit_test/base/test_ve3668n_input.py unit_test/base/test_ve3668n_recording_config.py unit_test/base/test_ve3668n_stores.py unit_test/base/test_ve3668n_wav_metadata.py unit_test/base/test_vkinging_sdk.py unit_test/base/test_ve3668n_discovery.py unit_test/base/test_ve3668n_hardware_selection.py unit_test/base/test_ve3668n_capture.py unit_test/base/test_ve3668n_resource.py unit_test/base/test_ve3668n_protocol.py unit_test/base/test_ve3668n_service.py unit_test/base/test_ve3668n_calibration.py unit_test/base/test_recording_ve_release.py unit_test/base/test_sequence_queue_references.py -q
python -m pytest unit_test/base/test_recording_capture.py unit_test/base/test_recording_startup_trim.py unit_test/base/test_recording_service.py unit_test/base/test_recording_service_pipeline.py unit_test/base/test_recording_worker_pipeline.py unit_test/base/test_recording_storage_consistency.py unit_test/base/test_recording_calibration_snapshot.py -q
python -m pytest unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_persistent_hardware.py unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_calibration.py unit_test/ui/test_ve3668n_prewarm_trigger.py unit_test/ui/test_ve3668n_prewarm_lifetime.py unit_test/ui/test_ve3668n_queue_parameters.py unit_test/ui/test_recording_advanced_settings.py unit_test/ui/test_recording_storage_config.py unit_test/ui/test_recording_preview_mode_config.py unit_test/ui/test_streaming_recording_config.py unit_test/ui/test_shared_queue_save.py unit_test/ui/test_vkinging_settings_workflow.py -q
python -m pytest unit_test/test_product_test_project_config.py unit_test/test_product_test_program_config.py unit_test/test_product_test_project_config_dialog.py unit_test/test_product_test_program_config_dialog.py unit_test/ui/test_operation_sequence_analysis_channel_mode.py unit_test/ui/test_original_recording_analysis_route.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_multichannel_recording_workspace.py unit_test/ui/test_raw_audio_csv_recording_integration.py -q
git diff --check
```

- [x] **Step 4：视觉和边界审查。** Qt 离屏截图核对高级设置折叠／展开、七档 dropdown、普通／VK 驱动、共享多产品长列表、查询失败提示。记录真实硬件验证状态；本机 SDK 若不可用，只交付模拟证据并明确没有真机结论。只有获得真实设备连接后才人工验证支持的实际整数率和范围，绝不以 fake 证明硬件能力。
- [x] **Step 5：提交最后的必要测试／修复检查点并完成双审。** 消息 `test: verify VK queue settings workflow`；如没有新文件改动则不创建空提交。将规格验收 1–11 与测试证据逐项对应，调用 verification-before-completion 后进入 finishing-a-development-branch，传递同一上游 metadataPath。最终保留工作区内容，不将临时文档或实现检查点当用户批准的永久提交。

## 计划审查与执行门槛

写完本文件后交给 plan-document-reviewer 检查全计划与规格。审查通过后向用户展示本计划并取得执行许可，再交给 subagent-driven-development；本计划本身不意味着已运行上述测试或已修改业务代码。

## 执行验证记录（2026-09-10）

以下记录实际执行结果，不表示整个仓库的所有测试均已通过。测试使用 Python 3.12、Qt offscreen、临时配置与 fake SDK；未连接真实采集卡。

| 验证集合 | 实际结果 |
| --- | --- |
| Task 10 第一组：14 个 VK 基础测试文件 | 2079 通过、1 失败；失败为设备发现子进程的 5 秒启动握手超时 |
| 不改代码重跑完整设备发现测试文件 | 141 通过 |
| Task 10 第二组：7 个录制基础测试文件 | 341 通过 |
| Task 10 第三组：13 个界面测试文件 | 560 通过，包含新增的 2 个完整连接流程 |
| Task 10 第四组：9 个产品、配置和集成测试文件 | 310 通过 |
| 补充轮次重置、独立播放及播放权限 | 55 通过 |
| 上述补充集合加刺激信号管理 | 74 通过、1 失败；重复名称错误码的旧断言不符，原始方法复现相同失败 |
| 最终审查发现的缺失采样率回退修复：完整连接流程 | 13 通过，包含启动后直接／产品／工况三个真实编辑入口 |
| 回退修复后的完整 UI 组、四个产品／配置集合及队列通道模式回归 | 721 通过；无 skipped 或 xfailed |
| 硬件物理发现与档案解耦修复：7 个相关 UI 文件 | 280 通过 |
| 硬件物理发现修复后的基础设备选择与校准存储回归 | 291 通过 |

新增连接测试贯穿设备发现、设备与通道选择、真实录制参数窗口、共享队列取消及确认保存、工况切换、实际录制请求、fake SDK、FLOAT WAV、实时波形和 Pa/V 校准。重新打开录制参数时高级设置仍折叠，已启用的实时波形继续生效；Q→R→Q 切换中使用各队列参数，设备档案与校准文件字节保持不变。新增测试最初暴露的是测试自身对 fake 计数 API 的误用；修正后通过，未宣称存在产品缺陷的红绿测试周期。

最终整体审查补充修正了缺失采样率的回退连接：启动发现只确认物理设备，参数窗口和录音均须读取实际设备档案；VK 队列加载不能提前填入普通声卡的 44100 Hz 默认值。三个真实入口的回归先复现 3 项失败（录音 48000、窗口 44100），修复后 3 项通过。档案读取错误回归先出现 1 项失败、1 项通过，窄范围处理 `OSError` 后两项通过。缺失档案仍使用 51200 Hz，显式非法采样率／量程可修复且不读取档案，普通声卡不访问 VK 档案，取消不写入配置或档案。

后续整体审查补充修正了硬件窗口的同类参数归属问题：恢复设备与新选设备的两个发现解析调用均不再加载回退档案，通用解析器的默认行为保持不变。恢复／新选两种入口与不可读／损坏／非法采样率三类档案组合，先复现 6 项失败，修正后 6 项通过。测试实际选择通道并确认硬件，再以队列明确的 96000 Hz、±100 mV 模拟录音；已有 Pa/V 和档案／校准文件字节不变。队列缺少采样率时仍报告真实档案错误，不把物理设备改为不可用。

规格验收对应关系：

| 验收项 | 主要证据 |
| --- | --- |
| 1–2：设备入口、发现、通道和恢复 | hardware、persistent_hardware、discovery、queue_parameters 及新增连接流程 |
| 3–4：高级设置、采样率边界和七档下拉 | advanced_settings、preview、storage、streaming、input、recording_config、SDK 及实际 Qt 截图 |
| 5：队列保存、共享影响与草稿 | queue_parameters、sequence_queue_references、shared_queue_save、产品编辑器及新增连接流程 |
| 6–7：采集快照、实际采样率和生命周期 | capture、resource、protocol、service、release、prewarm 及新增连接流程 |
| 8–9：校准、WAV 与原始电压 | stores、calibration、calibration_snapshot、WAV、capture 及 Q→R→Q 连接流程 |
| 10：移除录音监听、保留独立播放 | recording、streaming、旧监听字段连接流程、playback_controller 与 manual_playback_permission |
| 11：回归、布局与真机边界 | 上述四组回归；普通/VK 界面、高级设置折叠/展开、隐藏字段修复、七档展开列表、共享长列表和查询失败提示的 Qt 截图 |

补充测试已知限制：刺激信号重复名称分支实际返回 `INVALID_NAME=70`，旧测试期望 `INVALID_SAVE=102`；原始代码、测试和错误常量与本次修改前一致，原始方法复现同一失败，该分支未经过本次录制或队列代码。前序扩大检查还复现了既有轮次元数据忙碌分支断言失败（`test_recording_service_busy_releases_only_an_empty_round[False]`），未作无关修复。不能据此记录“全仓库测试全绿”。

硬件限制：fake SDK 证明参数传递、错误处理、保存及校准行为，不能证明真实设备支持所有区间内整数采样率或七档量程的实际下发；需要后续真机验证。
