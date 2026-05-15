import ast
from pathlib import Path

import numpy as np

from base.data_struct.data_deal_struct import DataDealStruct


def _load_resolve_analysis_channel_signal():
    source_path = Path(__file__).parents[2] / "ui" / "signal_analysis_window.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_analysis_channel_signal"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"DataDealStruct": DataDealStruct, "np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["resolve_analysis_channel_signal"]


def test_resolve_analysis_channel_signal_uses_multi_channel_data(capsys):
    resolve_analysis_channel_signal = _load_resolve_analysis_channel_signal()
    data_struct = DataDealStruct()
    original_mono = data_struct.store_wave_data
    original_multi = data_struct.store_wave_data_multi

    try:
        data_struct.store_wave_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        data_struct.store_wave_data_multi = np.array(
            [
                [1.0, 101.0],
                [2.0, 102.0],
                [3.0, 103.0],
            ],
            dtype=np.float32,
        )

        resolved = resolve_analysis_channel_signal(
            data_struct,
            {"analysis_channel": 1},
            "SPL",
        )

        np.testing.assert_array_equal(resolved, np.array([101.0, 102.0, 103.0], dtype=np.float32))
        assert capsys.readouterr().out == ""
    finally:
        data_struct.store_wave_data = original_mono
        data_struct.store_wave_data_multi = original_multi
