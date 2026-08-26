import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_widget.py"
ANALYSIS_CONTROLLER_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_analysis_controller.py"
ANALYSIS_MODEL_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_analysis_model.py"
OPERATION_SEQUENCE_PATH = REPO_ROOT / "ui" / "operation_sequence.py"
SIGNAL_ANALYSIS_PATH = REPO_ROOT / "ui" / "signal_analysis_window.py"


def test_analysis_model_marks_fft_as_v2pa_required_without_facade_state():
    source = SEQUENCE_WIDGET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    init_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assigned_sets = [
        node.value
        for node in ast.walk(init_node)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "analysis_types_requiring_v2pa"
            for target in node.targets
        )
    ]

    assert assigned_sets == []
    model_tree = ast.parse(ANALYSIS_MODEL_PATH.read_text(encoding="utf-8"))
    assigned_sets = [
        node.value
        for node in ast.walk(model_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "DEFAULT_ANALYSIS_CALIBRATION_TYPES"
            for target in node.targets
        )
    ]
    assert assigned_sets
    call = assigned_sets[0]
    literal = call.args[0] if isinstance(call, ast.Call) else call
    values = {
        element.value
        for element in literal.elts
        if isinstance(element, ast.Constant)
    }
    assert "FFT" in values


def test_operation_sequence_exposes_single_fft_item():
    source = OPERATION_SEQUENCE_PATH.read_text(encoding="utf-8")
    labels = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "(FFT)" in node.value
    ]

    assert labels.count("快速傅里叶变换 (FFT) ") == 1


def test_runtime_registers_and_dispatches_fft_analysis():
    signal_source = SIGNAL_ANALYSIS_PATH.read_text(encoding="utf-8")
    sequence_source = ANALYSIS_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert '"FFT": FftAnalysis' in signal_source
    assert 'hasattr(instance, "calculate_fft")' in sequence_source


def test_fft_runtime_uses_shared_v2pa_resolution():
    tree = ast.parse(SIGNAL_ANALYSIS_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FftAnalysis"
    )
    method_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_fft"
    )
    calls = [
        node
        for node in ast.walk(method_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_resolve_v2pa_factor_for_analysis"
    ]

    assert calls
