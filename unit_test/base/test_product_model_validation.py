import pytest

from base.product_model_validation import validate_product_model


@pytest.mark.parametrize("model", [
    "电机A01", "S004-1", "ABC-01(V2)", "A(版本(2))", "A(1)(2)", "A" * 80,
])
def test_valid_model_is_unchanged(model):
    assert validate_product_model(model) == model


def test_only_outer_whitespace_is_normalized():
    assert validate_product_model(" \t型号A-1(V2) \n") == "型号A-1(V2)"


@pytest.mark.parametrize("model", [
    "", "   ", "A" * 81, "test_prodect", "A.1", "A B", "A\tB", "A\nB",
    "A/B", "A\\B", "A:B", "A*B", "A?B", 'A"B', "A<B", "A>B", "A|B",
    "A\x00B", "A（1）", "Ａ１", "A😀", "-A", "(A)", "A-",
    "A()", "A(())", "A(1", "A)1", "A)(1)", "CON", "con", "NUL",
    "PRN", "aux", "COM1", "com9", "LPT1", "lpt9",
])
def test_invalid_new_model_is_rejected(model):
    with pytest.raises(ValueError):
        validate_product_model(model)


@pytest.mark.parametrize("model", ["test_prodect", "A.1", "A B", "A（旧版）"])
def test_existing_round_keeps_legacy_model_verbatim(model):
    assert validate_product_model(model, allow_legacy=True) == model
    with pytest.raises(ValueError):
        validate_product_model(model)


@pytest.mark.parametrize("model", [
    "", " ", ".", "..", "A/1", "A\n1", "A.", "A ", "CON", "con.1",
    "COM1.2", "NUL.tar.gz", "COM¹", "LPT²",
])
def test_legacy_round_still_rejects_unsafe_path_names(model):
    with pytest.raises(ValueError):
        validate_product_model(model, allow_legacy=True)
