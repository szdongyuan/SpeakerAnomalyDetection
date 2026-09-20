import csv
import hashlib
from pathlib import Path

import pytest

from base.audio_analysis_result_source import (
    discover_recording_results, item_channels, read_item_scalars,
)


def recording(root, project="项目甲", sample="1#", legacy=False):
    base = root / project / "型号一" / sample
    stem = f"{project}_型号一_{sample}_A口_R0001_档位1_20260915-112233"
    wav = base / ("wav" if legacy else "audio/wav") / f"{stem}.wav"
    images, data = base / "images" / stem, base / "csv" / stem
    images.mkdir(parents=True)
    data.mkdir(parents=True)
    return wav, images, data


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)


@pytest.mark.parametrize("legacy,sample", [(False, "1#"), (True, "1#"), (True, "audio")])
def test_exact_recording_missing_wav_and_readonly(tmp_path, legacy, sample):
    wav, images, data = recording(tmp_path, legacy=legacy, sample=sample)
    _, foreign, _ = recording(tmp_path, project="项目乙")
    (foreign / "异物_CH1.png").write_bytes(b"foreign")
    for name in ("声压级_CH2(麦克风).png", "声压级_CH10.jpg", "声压级_CH2(麦克风).jpeg"):
        (images / name).write_bytes(b"saved picture")
    csv_path = data / "声压级_总体声压级.csv"
    write_csv(csv_path, [
        ["通道", "总体声压级dB(Z)", "result"],
        ["时间0~1s_CH2(麦克风)", "65.200", "OK"],
        ["时间1~2s_CH2(麦克风)", "66.500", "NG"],
        ["CH1", "0", ""],
    ])
    before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    result = discover_recording_results(str(wav.relative_to(tmp_path)), application_root=str(tmp_path))
    assert not wav.exists()
    assert len(result.items) == 1
    item = result.items[0]
    assert len(item.images) == 3
    values = read_item_scalars(item)
    assert not values.issues
    assert [v.value for v in values.values] == ["65.200", "66.500", "0"]
    assert [v.segment_label for v in values.values] == ["时间0~1s", "时间1~2s", ""]
    assert values.values[0].unit == "dB(Z)"
    assert item_channels(item, values.values)[0] == (
        ("CH1", "CH1"), ("CH2", "CH2(麦克风)"), ("CH10", "CH10"),
    )
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == before


def test_curve_header_only_unknown_images_and_label_conflicts(tmp_path):
    wav, images, data = recording(tmp_path)
    (images / "自定义.png").write_bytes(b"image")
    (images / "频段_CH1(旧名).png").write_bytes(b"image")
    # Invalid UTF-8 is beyond the first text buffer: discovery must not read curve rows.
    (data / "频段_频段能量.csv").write_bytes(
        "X轴中心频率Hz,CH1(新名)_频段声压级_Y轴dB\n".encode() + b"0,0\n" * 10000 + b"\xff")
    result = discover_recording_results(str(wav))
    item = next(i for i in result.items if i.name == "频段")
    assert read_item_scalars(item).values == ()
    channels, issues = item_channels(item)
    assert channels == (("CH1", "CH1"),)
    assert "不一致" in issues[0]
    unknown = next(i for i in result.items if i.name == "自定义.png")
    assert item_channels(unknown)[0] == (("", "未记录通道"),)


def test_unsupported_ai_and_broken_scalar_csv(tmp_path):
    wav, _, data = recording(tmp_path)
    write_csv(data / "AI_模型输出.csv", [
        ["通道", "模型输出值", "result"], ["CH3", "0.1250", "NG"], ["CH4", "nan", ""],
    ])
    write_csv(data / "声压级_总体声压级.csv", [["通道", "错误列"], ["CH1", "55"]])
    (data / "其他.csv").write_text("unrecognized", encoding="utf-8")
    result = discover_recording_results(str(wav))
    assert "暂不支持" in result.issues[0]
    assert all(i.name != "AI" for i in result.items)
    assert any("AI_模型输出.csv" in issue for issue in result.issues)
    broken = read_item_scalars(next(i for i in result.items if i.name == "声压级"))
    assert not broken.values and "数值读取失败" in broken.issues[0]


def test_empty_unknown_path_permission_and_cancel(tmp_path, monkeypatch):
    assert "无法定位" in discover_recording_results(str(tmp_path / "old.wav")).issues[0]
    wav, images, _ = recording(tmp_path)
    assert not discover_recording_results(str(wav)).items
    iterator = Path.iterdir
    def denied(path):
        if path == images:
            raise PermissionError("test denied")
        return iterator(path)
    monkeypatch.setattr(Path, "iterdir", denied)
    assert "无法读取" in discover_recording_results(str(wav)).issues[0]
    with pytest.raises(InterruptedError):
        discover_recording_results(str(wav), cancel_requested=lambda: True)
