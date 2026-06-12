import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_widget.py"
OPERATION_SEQUENCE_PATH = REPO_ROOT / "ui" / "operation_sequence.py"


def test_sequence_marks_fft_as_v2pa_required():
    source = SEQUENCE_WIDGET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow")
    init_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assigned_sets = [
        node.value
        for node in ast.walk(init_node)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "analysis_types_requiring_v2pa" for target in node.targets)
    ]

    assert assigned_sets
    values = {elt.value for elt in assigned_sets[0].elts if isinstance(elt, ast.Constant)}
    assert "FFT" in values
    assert "Mel" in values


def test_operation_sequence_fft_label_extracts_single_fft_type():
    source = OPERATION_SEQUENCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    labels = [
        elt.value
        for elt in ast.walk(tree)
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and "(FFT)" in elt.value
    ]

    assert "快速傅里叶频谱 (FFT) " in labels
    assert "FFTFFT" not in {"".join(ch for ch in label if ch.isascii() and ch.isalpha()) for label in labels}
