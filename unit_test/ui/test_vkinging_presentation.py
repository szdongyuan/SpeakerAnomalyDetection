"""Pure display text must not expose or alter internal device identity."""

from copy import deepcopy

import pytest

from ui.vkinging_presentation import device_display_name, ve_failure_text


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"name": "Dev2"}, "Dev2"),
        ({"name": "  Dev2\t"}, "Dev2"),
        ({}, "VE3668N"),
        ({"name": ""}, "VE3668N"),
        ({"name": " \t\n"}, "VE3668N"),
        ({"name": None}, "VE3668N"),
        ({"name": 42}, "VE3668N"),
        ({"name": "Dev2 · shared-prefix-B-private-tail"}, "VE3668N"),
        ({"name": "shared-prefix-B-private-tail"}, "VE3668N"),
        ({"name": "Dev2", "machine_id": ""}, "Dev2"),
        ({"name": "Dev2", "machine_id": None}, "Dev2"),
        ({"name": "Dev2", "machine_id": 42}, "Dev2"),
        ({"name": "Dev2", "machine_id": "long-private-id-" * 100}, "Dev2"),
        (
            {"name": "Dev2 · " + "long-private-id-" * 100,
             "machine_id": "long-private-id-" * 100},
            "VE3668N",
        ),
    ],
)
def test_name_is_separate_from_machine_identity(fields, expected):
    device = {
        "machine_id": "shared-prefix-B-private-tail",
        "address": "raw-sdk-address",
        "physical_channels": [7, 1],
        "details": {"diagnostic": "another-private-machine-id"},
        **fields,
    }
    before = deepcopy(device)

    assert device_display_name(device) == expected
    assert device == before


def test_name_without_machine_id_keeps_sdk_name():
    device = {"name": "Dev2"}
    assert device_display_name(device) == "Dev2"
    assert device == {"name": "Dev2"}


@pytest.mark.parametrize(
    ("operation", "summary"),
    [
        ("discovery", "VE 设备检查失败，请连接设备后刷新。"),
        ("unavailable", "VE 设备不可用，请在硬件设置中检查设备和通道。"),
        ("hardware_save", "VE 硬件设置未保存，请检查设备和通道后重试。"),
        ("configuration", "VE 采集配置不可用，请检查采样率和量程。"),
        ("prewarm", "VE 设备初始化失败。"),
        ("recording", "VE 录音失败，请检查设备和采集配置。"),
        ("release", "VE 录音资源未释放，未发布结果；请等待资源释放后重试。"),
        ("calibration", "VE 输入校准失败，未保存校准；请检查设备和配置后重试。"),
        ("calibration_start", "VE 输入校准录音启动失败，请检查设备和配置。"),
        ("calibration_release", "VE 输入校准资源未释放，未保存校准；请等待资源释放。"),
        ("calibration_reset", "VE 输入校准重置失败，请重试。"),
    ],
)
def test_known_operation_has_fixed_summary_and_log_guidance(operation, summary):
    assert ve_failure_text(operation) == summary + "详细原因请查看日志。"


@pytest.mark.parametrize(
    "operation",
    ["unknown", "", "another-private-machine-id", "recording:another-private-machine-id",
     None, 42, True, ["another-private-machine-id"], {"stage": "another-private-machine-id"}],
)
def test_unknown_operation_cannot_become_visible(operation):
    before = deepcopy(operation)
    text = ve_failure_text(operation, code="another-private-machine-id")

    assert text == "VE 设备操作失败。详细原因请查看日志。"
    assert "another-private-machine-id" not in text
    assert operation == before


@pytest.mark.parametrize("code", [-9, 0, 17])
def test_actual_integer_error_codes_are_retained(code):
    text = ve_failure_text("prewarm", code=code)
    assert "VE 设备初始化失败。" in text
    assert f"(code={code})" in text
    assert text.endswith("详细原因请查看日志。")


@pytest.mark.parametrize("code", [None, True, False, "-9", -9.0, "another-private-machine-id", [], {}])
def test_non_integer_codes_are_not_rendered(code):
    before = deepcopy(code)
    assert ve_failure_text("prewarm", code=code) == "VE 设备初始化失败。详细原因请查看日志。"
    assert code == before
