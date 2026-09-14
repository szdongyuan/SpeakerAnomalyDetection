"""Stateless GUI text for VE devices; raw identity and diagnostics stay internal."""


def device_display_name(device):
    """Return the SDK name unless it is absent or contains the full MachineId."""
    name = device.get("name")
    if not isinstance(name, str) or not name.strip():
        return "VE3668N"
    name = name.strip()
    machine_id = device.get("machine_id")
    if isinstance(machine_id, str) and machine_id and machine_id in name:
        return "VE3668N"
    return name


def ve_failure_text(operation, *, code=None):
    """Map a known UI operation to fixed text, exposing only actual integer codes."""
    summaries = {
        "discovery": "VE 设备检查失败，请连接设备后刷新。",
        "unavailable": "VE 设备不可用，请在硬件设置中检查设备和通道。",
        "hardware_save": "VE 硬件设置未保存，请检查设备和通道后重试。",
        "configuration": "VE 采集配置不可用，请检查采样率和量程。",
        "prewarm": "VE 设备初始化失败。",
        "recording": "VE 录音失败，请检查设备和采集配置。",
        "release": "VE 录音资源未释放，未发布结果；请等待资源释放后重试。",
        "calibration": "VE 输入校准失败，未保存校准；请检查设备和配置后重试。",
        "calibration_start": "VE 输入校准录音启动失败，请检查设备和配置。",
        "calibration_release": "VE 输入校准资源未释放，未保存校准；请等待资源释放。",
        "calibration_reset": "VE 输入校准重置失败，请重试。",
    }
    summary = "VE 设备操作失败。"
    if isinstance(operation, str):
        summary = summaries.get(operation, summary)
    if type(code) is int:
        summary += f"(code={code})"
    return summary + "详细原因请查看日志。"
