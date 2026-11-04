import unittest
from unittest.mock import patch

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

from base.load_config import LoadUiConfig
from consts import error_code
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.product_condition_state import ProductConditionState


class TestMigratedConditionBusiness(unittest.TestCase):
    def test_channel_analysis_columns_follow_selected_condition_queue(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "condition_name": "0.1",
                    "key": "low",
                    "analysis_list": {
                        "display_sequence": ["声压级 (SPL) 1"],
                        "声压级 (SPL) 1": {"type": "SPL"},
                    },
                },
                {
                    "condition_name": "0.3",
                    "key": "high",
                    "analysis_list": {
                        "display_sequence": [
                            "频段能量 (FBA) 1",
                            "快速傅里叶变换 (FFT) 1",
                            "频谱分析 (Spec) 1",
                        ],
                        "频段能量 (FBA) 1": {"type": "FBA"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                        "频谱分析 (Spec) 1": {"type": "Spec"},
                    },
                },
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertEqual(
            list(panel.rows[panel.selected_key]["analysis_column_channels"]),
            ["SPL"],
        )

        panel.select_condition("high")

        self.assertEqual(
            list(panel.rows[panel.selected_key]["analysis_column_channels"]),
            ['FBA', 'FFT', 'Spec'],
        )

    def test_channel_analysis_columns_load_from_referenced_test_queue(self):
        queue_data = [
            {
                "seq1": {
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "频段能量 (FBA) 1",
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "频段能量 (FBA) 1": {"type": "FBA"},
                    }
                }
            }
        ]
        with patch.object(
            ProductConditionState,
            "_load_queue_catalog_safely",
            return_value={"测试队列A": {"path": "queue-a.json"}},
        ), patch.object(
            LoadUiConfig,
            "load_data_from_json",
            return_value=(error_code.OK, queue_data),
        ):
            panel = ProductConditionState(
                condition_configs=[
                    {
                        "condition_name": "0.1",
                        "key": "low",
                        "test_queue": "测试队列A",
                    }
                ]
            )
            panel.set_channels(list(range(5)))

        self.assertEqual(
            list(panel.rows[panel.selected_key]["analysis_column_channels"]),
            ['SPL', 'FBA'],
        )

    def test_condition_results_update_progress_and_compact_summary(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "group_name": "A口",
                    "condition_name": "0.1",
                    "display_name": "A口 / 0.1",
                    "trigger_state": "01",
                },
                {
                    "group_name": "A口",
                    "condition_name": "0.3",
                    "display_name": "A口 / 0.3",
                    "trigger_state": "02",
                },
            ]
        )
        panel.set_channels(list(range(5)))

        panel.set_condition_result("01", "OK", tone="ok")

        panel.set_condition_result("02", "NG", tone="ng")
        panel.set_final_result("NG", tone="ng")

        self.assertEqual(panel.final_result, "NG")
        self.assertEqual(panel.rows["02"]["result"], "NG")

    def test_channel_results_update_selected_channel_table(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "group_name": "A口",
                    "condition_name": "0.1",
                    "display_name": "A口 / 0.1",
                    "trigger_state": "01",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "频段能量 (FBA) 1",
                            "快速傅里叶变换 (FFT) 1",
                            "频谱分析 (Spec) 1",
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "频段能量 (FBA) 1": {"type": "FBA"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                        "频谱分析 (Spec) 1": {"type": "Spec"},
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertTrue(
            panel.set_condition_channel_results(
                "01",
                [
                    {
                        "SPL": "OK",
                        "FFT": "NG",
                        "FBA": "OK",
                        "Spec": "分析完成",
                        "result": "NG",
                    }
                ],
            )
        )

        first_channel = panel.rows[panel.selected_key]["channel_results"][0]
        self.assertEqual(first_channel["SPL"], "OK")
        self.assertEqual(first_channel["FFT"], "NG")
        self.assertEqual(first_channel["FBA"], "OK")
        self.assertEqual(first_channel["Spec"], "分析完成")
        self.assertEqual(first_channel["result"], "NG")

    def test_recording_highlight_stays_on_recording_row_when_viewing_history(self):
        panel = ProductConditionState(
            condition_configs=[
                {"condition_name": "0.1", "key": "01"},
                {"condition_name": "0.3", "key": "02"},
            ]
        )
        panel.set_channels([0, 1])

        panel.set_condition_result("02", "采集中", tone="running")
        panel.select_condition("01")
        panel.set_condition_result("02", "采集中", tone="running")

        self.assertEqual(panel.selected_key, "01")
        self.assertNotIn("selection_marker", panel.rows["01"])
        self.assertNotIn("recording_marker", panel.rows["02"])

    def test_analysis_running_state_does_not_look_like_recording_or_steal_view(self):
        panel = ProductConditionState(
            condition_configs=[
                {"condition_name": "0.1", "key": "01"},
                {"condition_name": "0.3", "key": "02"},
            ]
        )
        panel.select_condition("01")

        panel.set_condition_result("02", "分析中", tone="running")

        self.assertEqual(panel.selected_key, "01")

    def test_automatic_results_survive_legacy_pending_refresh_per_port(self):
        configs = []
        for port_index, port_name in enumerate(("A口", "B口", "C口")):
            for gear_index, gear_name in enumerate(("0.1", "0.3")):
                configs.append(
                    {
                        "group_name": port_name,
                        "condition_name": gear_name,
                        "key": f"{port_index}-{gear_index}",
                    }
                )
        panel = ProductConditionState(condition_configs=configs)
        panel.set_channels([0, 1])

        for key in ("0-0", "0-1"):
            panel.set_condition_channel_results(
                key,
                [
                    {"raw_channel": 0, "SPL": "NG", "result": "NG"},
                    {"raw_channel": 1, "SPL": "NG", "result": "NG"},
                ],
            )
            panel.set_condition_result(key, "NG", tone="ng")
            panel.set_condition_result(key, "待判定", tone="pending")

        self.assertEqual(panel.rows["0-0"]["result"], "NG")
        self.assertEqual(panel.rows["0-1"]["result"], "NG")



    def test_completed_analysis_without_threshold_counts_as_unjudged_gear(self):
        panel = ProductConditionState(
            condition_configs=[
                {"group_name": "A口", "condition_name": "0.1", "key": "01"},
                {"group_name": "A口", "condition_name": "0.3", "key": "02"},
            ]
        )
        panel.set_channels([0, 1])
        panel.set_condition_channel_results(
            "01",
            [
                {"raw_channel": 0, "SPL": "未启用阈值", "result": "待判定"},
                {"raw_channel": 1, "SPL": "未启用阈值", "result": "待判定"},
            ],
        )
        panel.set_condition_result("01", "未判定", tone="pending")
        panel.set_condition_result("01", "待判定", tone="pending")

        self.assertEqual(panel.rows["01"]["result"], "未判定")
        self.assertEqual(panel.rows["01"]["completed_channels"], 0)

    def test_round_summary_uses_all_automatic_gear_results(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "group_name": port,
                    "condition_name": gear,
                    "key": f"{port}-{gear}",
                }
                for port in ("A口", "B口", "C口")
                for gear in ("0.1", "0.3")
            ]
        )

        for key in list(panel.rows)[:-1]:
            panel.set_condition_result(key, "OK", tone="ok")
        self.assertEqual(panel.final_result, "待判定")

        panel.set_condition_result("C口-0.3", "OK", tone="ok")
        self.assertEqual(panel.final_result, "OK")
        self.assertEqual(panel.get_automatic_round_result(), ("OK", "ok", True))

        panel.set_final_result("待判定", tone="pending")
        self.assertEqual(panel.final_result, "OK")

        panel.set_condition_result("B口-0.1", "NG", tone="ng")
        self.assertEqual(panel.final_result, "NG")
        self.assertEqual(panel.get_automatic_round_result(), ("NG", "ng", True))

    def test_round_summary_keeps_failed_analysis_pending_for_retry(self):
        panel = ProductConditionState(
            condition_configs=[
                {"condition_name": "0.1", "key": "01"},
                {"condition_name": "0.3", "key": "02"},
            ]
        )

        panel.set_condition_result("01", "OK", tone="ok")
        panel.set_condition_result("02", "分析失败", tone="ng")

        self.assertEqual(panel.rows["02"]["analysis_completed"], False)
        self.assertEqual(panel.final_result, "待判定")
        self.assertEqual(
            panel.get_automatic_round_result(),
            ("待判定", "pending", False),
        )

    def test_round_summary_completes_without_judgment_when_no_item_contributes(self):
        panel = ProductConditionState(
            condition_configs=[
                {"condition_name": "0.1", "key": "01"},
                {"condition_name": "0.3", "key": "02"},
            ]
        )

        panel.set_condition_result("01", "未产生判定", tone="pending")
        panel.set_condition_result("02", "未产生判定", tone="pending")

        self.assertTrue(panel.rows["01"]["analysis_completed"])
        self.assertTrue(panel.rows["02"]["analysis_completed"])
        self.assertEqual(panel.final_result, "未产生判定")
        self.assertEqual(
            panel.get_automatic_round_result(),
            ("未产生判定", "pending", True),
        )

    def test_public_task_stage_is_limited_to_approved_statuses(self):
        panel = ProductConditionState()
        cases = {
            "": "等待开始",
            "等待开始": "等待开始",
            "采集中": "",
            "正转检测中": "",
            "反转检测中": "",
            "分析中": "分析中",
            "分析排队": "分析中",
            "分段分析": "分析中",
            "A口 / 0.1 分析中": "分析中",
            "本轮录音完成，等待分析": "分析中",
            "正转录音完成，等待分析": "分析中",
            "反转录音完成，等待分析": "分析中",
            "等待下一档位": "等待下一档位",
            "待判定": "等待下一档位",
            "本档位分析完成": "等待下一档位",
            "等待正转": "等待下一档位",
            "等待反转": "等待下一档位",
            "本轮完成": "本轮完成",
            "本轮采集完成": "本轮完成",
            "本轮采集完成，待判定": "本轮完成",
            "本轮分析完成，未判定": "本轮完成",
            "循环完成": "本轮完成",
            "本轮测试已关闭": "本轮完成",
            "测试异常": "测试异常",
            "结果不完整": "测试异常",
            "结果文件保存失败": "测试异常",
            "A口 / 0.1 分析失败": "测试异常",
            "分析失败，等待重试": "测试异常",
            "测试异常，等待工况状态码": "测试异常",
            "录音异常，循环已作废": "测试异常",
            "测试完成，PDF报告生成失败": "测试异常",
            "未注册的新状态": "测试异常",
        }
        approved = {
            "",
            "等待开始",
            "分析中",
            "等待下一档位",
            "本轮完成",
            "测试异常",
        }

        for raw_stage, expected_stage in cases.items():
            with self.subTest(raw_stage=raw_stage):
                panel.set_current_stage(raw_stage)
                self.assertEqual(panel.stage_text, expected_stage)
                self.assertIn(panel.stage_text, approved)

    def test_condition_state_keeps_rows_when_condition_keys_repeat(self):
        panel = ProductConditionState(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "", "test_queue": "默认配置"},
                {"condition_name": "7000", "trigger_state": "", "test_queue": "3"},
                {"condition_name": "8000", "trigger_state": "", "test_queue": "3"},
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertEqual(panel.condition_names, ["6000", "7000", "8000"])
        self.assertEqual(len(panel.rows), 3)
        self.assertEqual(len({item["key"] for item in panel.conditions}), 3)

    def test_analysis_config_drives_columns_without_legacy_detail_rows(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "condition_name": "6000",
                    "key": "6000",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "AI 分析 1",
                            "响度分析 1",
                            "频段能量 (FBA) 1",
                            "快速傅里叶变换 (FFT) 1",
                        ],
                        "声压级 (SPL) 1": {
                            "type": "SPL",
                            "limit_checked": True,
                            "upper_limit": "100",
                            "lower_limit": "90",
                        },
                        "AI 分析 1": {
                            "type": "AI",
                            "analyse_model_name": "model-a",
                        },
                        "响度分析 1": {
                            "type": "LOUD",
                            "limit_checked": True,
                            "upper_limit": "15",
                            "lower_limit": "8",
                        },
                        "频段能量 (FBA) 1": {
                            "type": "FBA",
                            "limit_checked": True,
                            "upper_limit": "45",
                        },
                        "快速傅里叶变换 (FFT) 1": {
                            "type": "FFT",
                            "limit_checked": True,
                            "upper_limit": "70",
                        },
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        panel.select_condition("6000")

        self.assertEqual(
            list(panel.rows[panel.selected_key]["analysis_column_channels"]),
            ['SPL', '响度', 'FBA', 'FFT'],
        )

    def test_channel_columns_hide_unconfigured_candidate_items(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "condition_name": "6000",
                    "key": "6000",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "响度分析 1",
                            "AI 分析 1",
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "响度分析 1": {"type": "LOUD"},
                        "AI 分析 1": {"type": "AI"},
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        panel.select_condition("6000")

        self.assertEqual(
            list(panel.rows[panel.selected_key]["analysis_column_channels"]),
            ['SPL', '响度'],
        )

    def test_legacy_analysis_result_api_keeps_state_without_extra_rows(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "condition_name": "6000",
                    "key": "6000",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "AI 分析 1",
                            "响度分析 1",
                            "频段能量 (FBA) 1",
                            "快速傅里叶变换 (FFT) 1",
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "AI 分析 1": {"type": "AI"},
                        "响度分析 1": {"type": "LOUD"},
                        "频段能量 (FBA) 1": {"type": "FBA"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        panel.select_condition("6000")
        panel.set_condition_result("6000", "OK")
        panel.set_condition_analysis_details(
            "6000",
            {
                "SPL": "总体声压：72.35 dB；判定：OK",
                "响度": "稳态平均响度：4.20 sone；最大瞬态响度：8.10 sone；判定：OK",
                "FBA": "OK",
                "FFT": "NG",
            },
        )

        self.assertEqual(
            panel.rows["6000"]["runtime_details"],
            {
                "SPL": "总体声压：72.35 dB；判定：OK",
                "响度": "稳态平均响度：4.20 sone；最大瞬态响度：8.10 sone；判定：OK",
                "FBA": "OK",
                "FFT": "NG",
            },
        )

    def test_pending_condition_result_clears_runtime_detail_values(self):
        panel = ProductConditionState(
            condition_configs=[
                {
                    "condition_name": "6000",
                    "key": "6000",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "AI 分析 1",
                            "频段能量 (FBA) 1",
                            "快速傅里叶变换 (FFT) 1",
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "AI 分析 1": {"type": "AI"},
                        "频段能量 (FBA) 1": {"type": "FBA"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        panel.select_condition("6000")
        panel.set_condition_analysis_details(
            "6000",
            {
                "SPL": "总体声压：72.35 dB；判定：OK",
                "FBA": "NG",
                "FFT": "NG",
            },
        )

        panel.set_condition_result("6000", "待检测", tone="pending")

        self.assertEqual(panel.rows["6000"]["runtime_details"], {})


# Main-window integration: business conditions never populate caller-owned stations.
def test_dashboard_integration_zero_stations_and_detached_video(qt_app):
    panel = MotorDetectionLeftPanel(QWidget(), condition_configs=[{"key": "a"}], queue_catalog={})
    panel.show()
    qt_app.processEvents()
    assert panel.get_station_count() == 0
    assert panel.dashboard_panel.summary_panel.isVisible()
    dashboard, video = panel.take_split_sections()
    assert dashboard is panel.dashboard_panel
    assert video is panel.video_monitor_panel
    assert not hasattr(panel, "result_panel")
    dashboard.close()
    video.close()
    dashboard.deleteLater()
    video.deleteLater()
    panel.close()
    panel.deleteLater()


def test_dashboard_integration_business_updates_leave_station_untouched(qt_app):
    config = [{"key": "a", "condition_name": "first"}, {"key": "b", "condition_name": "second"}]
    panel = MotorDetectionLeftPanel(QWidget(), condition_configs=config, queue_catalog={})
    panel.set_station_count(2)
    panel.set_station_row_count(station_id=1, count=2)
    panel.set_station_row(station_id=1, row=0, values=["distinct", 12, 34])
    panel.set_station_column(station_id=1, column=1, values=[56, 78])
    panel.set_station_cell(station_id=1, row=1, column=0, value="second")
    panel.set_station_results(1, "OK", "NG")
    from PyQt5.QtWidgets import QPushButton
    header = panel.station_panel.findChild(QPushButton, "stationHeader1")
    notifications = []
    panel.condition_selected.connect(notifications.append)
    selected = panel.condition_state.selected_key
    header.click()
    assert panel.condition_state.selected_key == selected
    assert notifications == []
    before = panel.get_station_data(1)
    panel.set_channels([0, 4])
    panel.set_current_round(7)
    panel.set_current_stage("分析中")
    panel.set_condition_channel_results("a", [{"raw_channel": 4, "result": "NG"}])
    panel.set_condition_analysis_details("a", {"SPL": 42})
    panel.set_condition_result("a", "NG")
    panel.set_condition_result("b", "采集中")
    assert notifications == ["b"]
    assert panel.condition_state.rows["a"]["result"] == "NG"
    assert panel.condition_state.current_round == 7
    assert panel.refresh_condition_configs(config, queue_catalog={}, preserve_results=True)
    assert panel.get_station_data(1) == before
    assert header.isChecked()
    panel.reset_result_panel()
    assert panel.condition_state.rows["a"]["result"] == "待检测"
    panel.set_condition_configs(config, queue_catalog={})
    panel.clear_current_stage()
    assert panel.condition_state.stage_text == ""
    assert panel.get_station_count() == 2
    assert panel.get_station_data(1) == before
    assert header.isChecked()
    panel.close()
    panel.deleteLater()


def test_left_panel_preserves_video_and_hides_business_count_board(qt_app):
    board = QWidget()
    panel = MotorDetectionLeftPanel(board, condition_configs=[], queue_catalog={})
    panel.resize(340, 800)
    panel.show()
    qt_app.processEvents()
    assert board.isHidden()
    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert "视频监控" in labels
    assert "信息汇总" in labels
    assert panel.content_layout.itemAt(0).widget() is panel.dashboard_panel
    assert panel.content_layout.itemAt(1).widget() is panel.video_monitor_panel
    assert panel.get_station_count() == 0
    panel.close()
    panel.deleteLater()
