from datetime import datetime
import csv
from pathlib import Path
from types import SimpleNamespace

from base.analysis_artifact_paths import AnalysisStorageContext
from base.analysis_csv_exporter import export_channel_mapping_csv, export_item_csvs


def _context(tmp_path):
    return AnalysisStorageContext(
        str(tmp_path.resolve()),
        "项目",
        "型号",
        "样本",
        1,
        "端口",
        "0.3",
        datetime(2026, 9, 1, 8, 0, 0),
    )


def _read(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def test_channel_mapping_snapshot_uses_physical_channel_numbers(tmp_path):
    record = export_channel_mapping_csv(
        _context(tmp_path),
        "共同主名",
        [SimpleNamespace(raw_channel=2), SimpleNamespace(raw_channel=0)],
        {"CH1": "前", "CH3": "左"},
    )

    assert record.ok
    assert Path(record.file_path).name == "channel_mapping.csv"
    assert _read(record.file_path) == [
        ["physical_channel", "label"],
        ["CH1", "前"],
        ["CH3", "左"],
    ]

    second = export_channel_mapping_csv(
        _context(tmp_path),
        "共同主名",
        [SimpleNamespace(raw_channel=0)],
        {"CH1": "已修改"},
    )
    assert second.ok
    assert _read(record.file_path)[1] == ["CH1", "前"]


def test_spl_exports_overall_and_one_shared_axis_realtime_csv(tmp_path):
    outputs = [
        {
            "raw_channel": 0,
            "judgement": "OK",
            "metrics": {
                "overall_spl": 78.2,
                "overall_lower_limit": 70.0,
                "overall_upper_limit": 85.0,
            },
            "csv_curve": {
                "x": [100.0, 100.01],
                "y": [77.8, 78.0],
                "lower": [70.0, 70.0],
                "upper": [82.0, 82.0],
            },
        },
        {
            "raw_channel": 1,
            "judgement": "NG",
            "metrics": {
                "overall_spl": 86.1,
                "overall_lower_limit": 70.0,
                "overall_upper_limit": 85.0,
            },
            "csv_curve": {
                "x": [100.0, 100.01],
                "y": [81.2, 82.4],
                "lower": [70.0, 70.0],
                "upper": [82.0, 82.0],
            },
        },
    ]
    records = export_item_csvs(
        _context(tmp_path),
        "共同主名",
        "声压级1",
        "SPL",
        {
            "show_overall_spl": True,
            "limit_checked": True,
            "limit_metric": "overall_spl",
        },
        outputs,
        {"CH1": "前", "CH2": "后"},
    )

    assert [item.config_item_name for item in records] == ["总体声压级", "实时声压级"]
    assert all(item.ok for item in records)
    assert all(Path(item.file_path).parent.name == "共同主名" for item in records)
    assert [Path(item.file_path).name for item in records] == [
        "声压级1_总体声压级.csv",
        "声压级1_实时声压级.csv",
    ]
    overall = _read(records[0].file_path)
    realtime = _read(records[1].file_path)
    assert overall[0] == ["通道", "总体声压级dB", "总体下限dB", "总体上限dB", "result"]
    assert [row[0] for row in overall[1:]] == ["CH1(前)", "CH2(后)"]
    assert overall[1][-1] == "OK"
    assert realtime[0] == [
        "X轴时间秒",
        "CH1(前)_SPL_Y轴dB",
        "CH2(后)_SPL_Y轴dB",
        "下限_Y轴dB",
        "上限_Y轴dB",
    ]
    assert "result" not in realtime[0]
    assert len(realtime) == 3


def test_curve_csv_keeps_empty_limit_columns_after_channel_values(tmp_path):
    records = export_item_csvs(
        _context(tmp_path),
        "共同主名",
        "频段能量1",
        "FBA",
        {},
        [
            {
                "raw_channel": 0,
                "csv_curve": {
                    "x": [100.0, 200.0],
                    "y": [61.0, 62.0],
                    "lower": [],
                    "upper": [],
                },
            }
        ],
        {"CH1": "前"},
    )

    rows = _read(records[0].file_path)
    assert rows == [
        [
            "X轴中心频率Hz",
            "CH1(前)_频段声压级_Y轴dB",
            "下限_Y轴dB",
            "上限_Y轴dB",
        ],
        ["100", "61", "", ""],
        ["200", "62", "", ""],
    ]


def test_fft_csv_places_threshold_curves_after_channel_values(tmp_path):
    records = export_item_csvs(
        _context(tmp_path),
        "共同主名",
        "快速傅里叶变换1",
        "FFT",
        {},
        [
            {
                "raw_channel": 0,
                "csv_curve": {
                    "x": [100.0],
                    "y": [-18.0],
                    "lower": [-30.0],
                    "upper": [-10.0],
                },
            }
        ],
    )

    rows = _read(records[0].file_path)
    assert rows == [
        [
            "X轴频率Hz",
            "CH1_FFT_Y轴dB",
            "下限_Y轴dB",
            "上限_Y轴dB",
        ],
        ["100", "-18", "-30", "-10"],
    ]


def test_ai_uses_one_numeric_row_per_channel(tmp_path):
    records = export_item_csvs(
        _context(tmp_path),
        "共同主名",
        "AI1",
        "AI",
        {},
        [
            {
                "raw_channel": 0,
                "judgement": "OK",
                "metrics": {
                    "model_output_value": 0.82,
                    "decision_threshold": 0.70,
                },
            },
            {
                "raw_channel": 1,
                "judgement": "NG",
                "metrics": {
                    "model_output_value": 0.61,
                    "decision_threshold": 0.70,
                },
            },
        ],
        {"CH1": "前"},
    )
    rows = _read(records[0].file_path)
    assert rows == [
        ["通道", "模型输出值", "判定阈值", "result"],
        ["CH1(前)", "0.82", "0.7", "OK"],
        ["CH2", "0.61", "0.7", "NG"],
    ]
