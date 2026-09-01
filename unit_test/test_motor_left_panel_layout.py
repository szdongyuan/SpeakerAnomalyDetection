import unittest
from unittest.mock import patch

from PyQt5.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QWidget

from base.load_config import LoadUiConfig
from consts import error_code
from consts import ui_style_const
INPUT_CHANNEL_LABELS = tuple(f"CH{index}" for index in range(1, 6))
from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.motor_panel_common import MotorSectionCard
from ui.sequence.motor_video_monitor_panel import MotorVideoMonitorPanel


class TestMotorLeftPanelLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_left_panel_uses_video_monitor_instead_of_summary(self):
        summary_widget = QWidget()
        panel = MotorDetectionLeftPanel(summary_widget)
        panel.set_channels(list(range(5)))
        scroll_area = panel.layout().itemAt(0).widget()
        content_widget = scroll_area.widget()
        content_layout = content_widget.layout()

        first_widget = content_layout.itemAt(0).widget()
        second_widget = content_layout.itemAt(1).widget()

        self.assertIsInstance(first_widget, MotorAiResultPanel)
        self.assertIsInstance(second_widget, MotorVideoMonitorPanel)
        self.assertIs(second_widget, panel.video_monitor_panel)
        self.assertFalse(summary_widget.isVisible())
        self.assertTrue(summary_widget.isHidden())

        card = second_widget.findChild(MotorSectionCard)
        self.assertIsInstance(card, MotorSectionCard)

        labels = [lbl.text() for lbl in second_widget.findChildren(QLabel)]
        self.assertIn("视频监控", labels)
        self.assertIn("2K预览", labels)
        self.assertIn("实时视频画面", labels)
        self.assertIn("摄像头待接入", labels)
        self.assertNotIn("信息汇总", labels)
        self.assertIsNotNone(second_widget.findChild(QFrame, "videoMonitorPlaceholder"))
        live_label = second_widget.findChild(QLabel, "videoMonitorLiveLabel")
        self.assertIsNotNone(live_label)
        self.assertIn(
            ui_style_const.MAIN_UI_SMALL_FONT_FAMILY,
            live_label.styleSheet(),
        )

    def test_ai_result_panel_uses_condition_names(self):
        summary_widget = QWidget()
        condition_configs = LoadUiConfig.load_product_test_program_condition_configs()
        panel = MotorDetectionLeftPanel(summary_widget, condition_configs=condition_configs)
        panel.set_channels(list(range(5)))
        expected_names = [item["condition_name"] for item in condition_configs]

        self.assertEqual(panel.ai_result_panel.condition_names, expected_names)
        if condition_configs:
            self.assertTrue(panel.set_condition_result(condition_configs[-1]["key"], "NG"))

    def test_ai_result_panel_uses_test_task_structure(self):
        panel = MotorAiResultPanel(
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
                {
                    "group_name": "B口",
                    "condition_name": "0.1",
                    "display_name": "B口 / 0.1",
                    "trigger_state": "03",
                },
            ]
        )
        panel.set_channels(list(range(5)))

        labels = [label.text() for label in panel.findChildren(QLabel)]
        port_combo = panel.findChild(QComboBox, "testTaskPortCombo")

        self.assertIn("测试任务", labels)
        self.assertNotIn("工况判定结果", labels)
        self.assertIn("当前端口", labels)
        self.assertIn("档位列表", labels)
        self.assertIn("当前测试：0.1", labels)
        self.assertIn("档位进度：0/2", labels)
        self.assertIn("判定汇总", labels)
        self.assertIn("当前轮次", labels)
        self.assertNotIn("任务状态", labels)
        self.assertNotIn("档位任务", labels)
        self.assertEqual(panel.stage_label.text(), "等待开始")
        self.assertEqual(panel.stage_label.minimumWidth(), 230)
        self.assertEqual(panel.stage_label.maximumWidth(), 230)
        self.assertIsNotNone(port_combo)
        self.assertEqual([port_combo.itemText(i) for i in range(port_combo.count())], ["A口", "B口"])
        self.assertEqual(panel.port_index_label.text(), "第1/2个")
        self.assertEqual(panel.count_label.text(), "档位进度：0/2")
        self.assertNotIn("index", panel.rows["01"]["labels"])
        self.assertEqual(panel.rows["01"]["labels"]["name"].text(), "0.1")
        self.assertIn(
            ui_style_const.MAIN_UI_SMALL_FONT_FAMILY,
            panel.stage_label.styleSheet(),
        )
        self.assertIn(
            ui_style_const.UI_FONT_FAMILY,
            panel.current_port_combo.styleSheet(),
        )

    def test_port_selector_popup_uses_readable_text_colors(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {"group_name": "A口", "condition_name": "0.1", "key": "01"},
                {"group_name": "B口", "condition_name": "0.1", "key": "02"},
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertTrue(hasattr(ui_style_const, "motor_port_combo_style"))
        style = panel.current_port_combo.styleSheet()
        self.assertEqual(style, ui_style_const.motor_port_combo_style)
        self.assertIn("QComboBox QAbstractItemView", style)
        self.assertIn("color: #1F2937", style)
        self.assertIn("selection-color: #FFFFFF", style)
        self.assertIn("QComboBox QAbstractItemView::item:selected", style)

    def test_ai_result_summary_row_has_clear_visual_grouping(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {
                    "group_name": "A口",
                    "condition_name": "0.1",
                    "trigger_state": "01",
                }
            ]
        )
        panel.set_channels(list(range(5)))

        summary_row = panel.findChild(QWidget, "testTaskSummaryRow")
        self.assertIsNotNone(summary_row)
        self.assertEqual(summary_row.minimumHeight(), 52)
        self.assertIn("background:#F8FBFF", summary_row.styleSheet())
        margins = summary_row.layout().contentsMargins()
        self.assertEqual((margins.top(), margins.bottom()), (10, 10))
        self.assertIsNotNone(panel.findChild(QFrame, "testTaskSummaryAccent"))
        self.assertIsNotNone(panel.findChild(QFrame, "testTaskSummaryDivider"))
        self.assertGreaterEqual(panel.port_result_value.minimumWidth(), 58)
        self.assertGreaterEqual(panel.port_result_value.minimumHeight(), 24)
        self.assertIn("background:#EEF3F8", panel.port_result_value.styleSheet())

    def test_selected_condition_expands_channel_judgement_table(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {
                    "group_name": "A口",
                    "condition_name": "0.1",
                    "display_name": "A口 / 0.1",
                    "trigger_state": "01",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "快速傅里叶变换 (FFT) 1",
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        panel.select_condition("01", show_detail=True)
        labels = [label.text() for label in panel.detail_frame.findChildren(QLabel)]

        self.assertFalse(panel.detail_frame.isHidden())
        self.assertIn("通道", labels)
        self.assertIn("SPL判定", labels)
        self.assertIn("FFT", labels)
        self.assertNotIn("1/3倍频程", labels)
        self.assertIn("结果", labels)
        self.assertEqual(panel.detail_frame.layout().count(), 1)
        self.assertEqual([label for label in labels if label.startswith("CH")], [
            *INPUT_CHANNEL_LABELS
        ])

    def test_channel_analysis_columns_follow_selected_condition_queue(self):
        panel = MotorAiResultPanel(
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
                        ],
                        "频段能量 (FBA) 1": {"type": "FBA"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                    },
                },
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertEqual(
            panel.channel_analysis_columns,
            [{"key": "SPL", "header": "SPL判定"}],
        )

        panel.select_condition("high", show_detail=True)

        self.assertEqual(
            panel.channel_analysis_columns,
            [
                {"key": "FBA", "header": "1/3倍频程"},
                {"key": "FFT", "header": "FFT"},
            ],
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
            MotorAiResultPanel,
            "_load_queue_catalog_safely",
            return_value={"测试队列A": {"path": "queue-a.json"}},
        ), patch.object(
            LoadUiConfig,
            "load_data_from_json",
            return_value=(error_code.OK, queue_data),
        ):
            panel = MotorAiResultPanel(
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
            panel.channel_analysis_columns,
            [
                {"key": "SPL", "header": "SPL判定"},
                {"key": "FBA", "header": "1/3倍频程"},
            ],
        )

    def test_port_selection_filters_condition_rows(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {
                    "group_name": "A口",
                    "condition_name": "0.1",
                    "display_name": "A口 / 0.1",
                    "trigger_state": "01",
                },
                {
                    "group_name": "B口",
                    "condition_name": "0.3",
                    "display_name": "B口 / 0.3",
                    "trigger_state": "02",
                },
            ]
        )
        panel.set_channels(list(range(5)))

        panel.current_port_combo.setCurrentIndex(1)

        self.assertTrue(panel.rows["01"]["button"].isHidden())
        self.assertFalse(panel.rows["02"]["button"].isHidden())
        self.assertEqual(panel.current_test_label.text(), "当前测试：0.3")
        self.assertEqual(panel.port_index_label.text(), "第2/2个")

    def test_port_rows_show_gear_names_without_global_sequence_numbers(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {"group_name": "A口", "condition_name": "0.1", "key": "01"},
                {"group_name": "A口", "condition_name": "0.3", "key": "02"},
                {"group_name": "B口", "condition_name": "0.1", "key": "03"},
                {"group_name": "B口", "condition_name": "0.3", "key": "04"},
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertEqual(
            [panel.rows[key]["labels"]["name"].text() for key in ("01", "02")],
            ["0.1", "0.3"],
        )
        panel.current_port_combo.setCurrentIndex(1)
        self.assertEqual(
            [panel.rows[key]["labels"]["name"].text() for key in ("03", "04")],
            ["0.1", "0.3"],
        )
        self.assertTrue(all("index" not in row["labels"] for row in panel.rows.values()))

    def test_condition_results_update_progress_and_compact_summary(self):
        panel = MotorAiResultPanel(
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
        self.assertEqual(panel.progress_label.text(), "档位进度：1/2")
        self.assertEqual(panel.port_result_value.text(), "待判定")

        panel.set_condition_result("02", "NG", tone="ng")
        panel.set_final_result("NG", tone="ng")

        self.assertEqual(panel.progress_label.text(), "档位进度：2/2")
        self.assertEqual(panel.port_result_value.text(), "NG")
        self.assertEqual(panel.round_result_value.text(), "NG")
        self.assertIn("background:#FDECEC", panel.port_result_value.styleSheet())
        self.assertIn("background:#FDECEC", panel.round_result_value.styleSheet())
        self.assertEqual(panel.rows["02"]["labels"]["result"].text(), "NG")
        self.assertIn("#D94343", panel.rows["02"]["labels"]["result"].styleSheet())
        self.assertIn("background:#EAF2FB", panel.rows["01"]["button"].styleSheet())
        self.assertNotIn("background:#FCE8E8", panel.rows["02"]["button"].styleSheet())

    def test_channel_results_update_selected_channel_table(self):
        panel = MotorAiResultPanel(
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
                        ],
                        "声压级 (SPL) 1": {"type": "SPL"},
                        "频段能量 (FBA) 1": {"type": "FBA"},
                        "快速傅里叶变换 (FFT) 1": {"type": "FFT"},
                    },
                }
            ]
        )
        panel.set_channels(list(range(5)))

        self.assertTrue(
            panel.set_condition_channel_results(
                "01",
                [{"SPL": "OK", "FFT": "NG", "FBA": "OK", "result": "NG"}],
            )
        )

        first_channel = panel.channel_detail_labels[0]
        self.assertEqual(first_channel["SPL"].text(), "OK")
        self.assertEqual(first_channel["FFT"].text(), "NG")
        self.assertEqual(first_channel["FBA"].text(), "OK")
        self.assertEqual(first_channel["result"].text(), "NG")
        self.assertIn("#D94343", first_channel["FFT"].styleSheet())

    def test_running_next_condition_moves_detail_without_clearing_previous_results(self):
        analysis_list = {
            "display_sequence": ["声压级 (SPL) 1"],
            "声压级 (SPL) 1": {"type": "SPL"},
        }
        panel = MotorAiResultPanel(
            condition_configs=[
                {
                    "condition_name": "0.1",
                    "key": "01",
                    "analysis_list": analysis_list,
                },
                {
                    "condition_name": "0.3",
                    "key": "02",
                    "analysis_list": analysis_list,
                },
            ]
        )
        panel.set_channels([0, 1])
        panel.set_condition_channel_results(
            "01",
            [
                {"raw_channel": 0, "SPL": "OK", "result": "OK"},
                {"raw_channel": 1, "SPL": "NG", "result": "NG"},
            ],
        )
        panel.set_condition_result("01", "NG", tone="ng")
        panel.select_condition("01", show_detail=True)

        panel.set_condition_result("02", "采集中", tone="running")

        previous_row = panel.rows["01"]
        self.assertEqual(previous_row["completed_channels"], 2)
        self.assertTrue(previous_row["analysis_completed"])
        self.assertEqual(previous_row["labels"]["progress"].text(), "通道判定：2/2")
        self.assertEqual(previous_row["labels"]["result"].text(), "NG")
        self.assertEqual(len(previous_row["channel_results"]), 2)
        self.assertEqual(panel.progress_label.text(), "档位进度：1/2")
        self.assertEqual(panel._detail_owner_key, "02")
        self.assertEqual(
            panel.rows_layout.indexOf(panel.detail_frame),
            panel.rows_layout.indexOf(panel.rows["02"]["button"]) + 1,
        )
        self.assertTrue(
            all(
                labels["result"].text() == "待检测"
                for labels in panel.channel_detail_labels
            )
        )

        panel.select_condition("01", show_detail=True)

        self.assertEqual(panel._detail_owner_key, "01")
        self.assertEqual(
            [labels["result"].text() for labels in panel.channel_detail_labels],
            ["OK", "NG"],
        )

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
        panel = MotorAiResultPanel(condition_configs=configs)
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

        self.assertEqual(panel.rows["0-0"]["labels"]["result"].text(), "NG")
        self.assertEqual(panel.rows["0-1"]["labels"]["result"].text(), "NG")
        self.assertEqual(panel.progress_label.text(), "档位进度：2/2")

        panel.current_port_combo.setCurrentIndex(1)

        self.assertEqual(panel.progress_label.text(), "档位进度：0/2")

    def test_completed_analysis_without_threshold_counts_as_unjudged_gear(self):
        panel = MotorAiResultPanel(
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

        self.assertEqual(panel.rows["01"]["labels"]["result"].text(), "未判定")
        self.assertEqual(panel.rows["01"]["completed_channels"], 0)
        self.assertEqual(panel.progress_label.text(), "档位进度：1/2")

    def test_round_summary_uses_all_automatic_gear_results(self):
        panel = MotorAiResultPanel(
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
        self.assertEqual(panel.round_result_value.text(), "待判定")

        panel.set_condition_result("C口-0.3", "OK", tone="ok")
        self.assertEqual(panel.round_result_value.text(), "OK")
        self.assertEqual(panel.get_automatic_round_result(), ("OK", "ok", True))

        panel.set_final_result("待判定", tone="pending")
        self.assertEqual(panel.round_result_value.text(), "OK")

        panel.set_condition_result("B口-0.1", "NG", tone="ng")
        self.assertEqual(panel.round_result_value.text(), "NG")
        self.assertEqual(panel.get_automatic_round_result(), ("NG", "ng", True))

    def test_current_stage_updates_visible_task_status(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {"condition_name": "档位1", "trigger_state": "01"},
            ]
        )
        panel.set_channels(list(range(5)))

        fixed_width = panel.stage_label.width()
        panel.set_current_stage("档位1 检测中", tone="running")

        self.assertEqual(panel.stage_text, "档位1 检测中")
        self.assertEqual(panel.stage_label.text(), "档位1 检测中")
        self.assertEqual(panel.stage_label.width(), fixed_width)
        self.assertIn("#FFFFFF", panel.stage_label.styleSheet())

    def test_ai_result_panel_keeps_rows_when_condition_keys_repeat(self):
        panel = MotorAiResultPanel(
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

    def test_ai_final_result_stays_near_panel_bottom(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "01"},
                {"condition_name": "7000", "trigger_state": "02"},
            ]
        )
        panel.set_channels(list(range(5)))
        panel.resize(400, 360)
        panel.show()
        self.app.processEvents()

        content = panel.final_value.parentWidget()
        bottom_gap = content.height() - panel.final_value.geometry().bottom()

        self.assertLessEqual(bottom_gap, 14)

    def test_ai_result_detail_toggles_on_same_condition_click(self):
        condition_configs = [{"key": "gear", "condition_name": "6000 rpm"}]
        panel = MotorAiResultPanel(condition_configs=condition_configs)
        panel.set_channels(list(range(5)))
        key = condition_configs[0]["key"]

        panel.select_condition(key, show_detail=True)
        self.assertFalse(panel.detail_frame.isHidden())

        panel.select_condition(key, show_detail=True)
        self.assertTrue(panel.detail_frame.isHidden())

    def test_analysis_config_drives_columns_without_legacy_detail_rows(self):
        panel = MotorAiResultPanel(
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

        panel.select_condition("6000", show_detail=True)
        labels = [label.text() for label in panel.detail_frame.findChildren(QLabel)]

        self.assertEqual(
            panel.channel_analysis_columns,
            [
                {"key": "SPL", "header": "SPL判定"},
                {"key": "响度", "header": "响度判定"},
                {"key": "AI分析", "header": "AI判定"},
                {"key": "FBA", "header": "1/3倍频程"},
                {"key": "FFT", "header": "FFT"},
            ],
        )
        self.assertEqual(panel.detail_frame.layout().count(), 1)
        self.assertNotIn("SPL", labels)
        self.assertNotIn("响度", labels)
        self.assertNotIn("AI分析", labels)

    def test_channel_columns_hide_unconfigured_candidate_items(self):
        panel = MotorAiResultPanel(
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

        panel.select_condition("6000", show_detail=True)

        self.assertEqual(
            panel.channel_analysis_columns,
            [
                {"key": "SPL", "header": "SPL判定"},
                {"key": "响度", "header": "响度判定"},
                {"key": "AI分析", "header": "AI判定"},
            ],
        )

    def test_legacy_analysis_result_api_keeps_state_without_extra_rows(self):
        panel = MotorAiResultPanel(
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

        panel.select_condition("6000", show_detail=True)
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
        panel.set_condition_scores("6000", 71.6, 28.4)

        self.assertEqual(
            panel.rows["6000"]["runtime_details"],
            {
                "SPL": "总体声压：72.35 dB；判定：OK",
                "响度": "稳态平均响度：4.20 sone；最大瞬态响度：8.10 sone；判定：OK",
                "AI分析": "OK Score：71.60%；NG Score：28.40%；判定：OK",
                "FBA": "OK",
                "FFT": "NG",
            },
        )
        self.assertEqual(panel.detail_frame.layout().count(), 1)

    def test_pending_condition_result_clears_runtime_detail_values(self):
        panel = MotorAiResultPanel(
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

        panel.select_condition("6000", show_detail=True)
        panel.set_condition_analysis_details(
            "6000",
            {
                "SPL": "总体声压：72.35 dB；判定：OK",
                "AI分析": "OK Score：71.60%；NG Score：28.40%；判定：OK",
                "FBA": "NG",
                "FFT": "NG",
            },
        )

        panel.set_condition_result("6000", "待检测", tone="pending")

        self.assertEqual(panel.rows["6000"]["runtime_details"], {})
        self.assertEqual(panel.detail_frame.layout().count(), 1)


if __name__ == "__main__":
    unittest.main()
