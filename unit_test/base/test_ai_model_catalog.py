import sqlite3

import pytest

from base.ai_model_catalog import AiModelCatalog, ModelCatalogError, ai_execution_issue


def test_missing_database_returns_empty_without_creating_it(tmp_path):
    path = tmp_path / "missing.db"
    assert AiModelCatalog(path, tmp_path).list_models() == ()
    assert AiModelCatalog(path, tmp_path).list_records() == ()
    assert not path.exists()


def test_registry_is_read_only_and_filters_legacy_or_missing_models(tmp_path):
    path = tmp_path / "models.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE training_model_table (model_name TEXT, model_path TEXT)")
        db.executemany("INSERT INTO training_model_table VALUES (?, ?)", [
            ("legacy", "old.keras"), ("missing", "absent.onnx"), ("ready", "model.onnx"),
        ])
    # File existence is catalog eligibility, not a claim that ONNX can load it.
    (tmp_path / "model.onnx").write_bytes(b"catalog-fixture")
    original = path.read_bytes()
    choices = AiModelCatalog(path, tmp_path).list_models()
    assert [choice.available for choice in choices] == [False, False, True]
    assert [choice.reason for choice in choices[:2]] == ["不是 ONNX 模型", "模型文件不存在"]
    assert path.read_bytes() == original


def test_invalid_database_reports_error(tmp_path):
    path = tmp_path / "models.db"
    path.write_bytes(b"broken")
    with pytest.raises(ModelCatalogError, match="模型目录读取失败"):
        AiModelCatalog(path, tmp_path).list_models()
    with pytest.raises(ModelCatalogError, match="模型目录读取失败"):
        AiModelCatalog(path, tmp_path).list_records()


def test_management_reads_all_metadata_without_filtering_legacy_or_writing(tmp_path):
    path = tmp_path / "models.db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE training_model_table (
            model_id TEXT, model_name TEXT, model_path TEXT, config_path TEXT,
            input_dim TEXT, output_dim INTEGER, accuracy REAL, update_date TEXT,
            model_description TEXT)""")
        db.executemany("INSERT INTO training_model_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("a", "legacy", "old.keras", "old.yml", "132300 x 1", 1, None, "2026-01-01", None),
            ("b", "registered", str(tmp_path / "model.onnx"), "model.yml",
             "144000 x 1", 1, 0.0, "2026-01-02", "stored description"),
        ])
    (tmp_path / "model.onnx").write_bytes(b"not-a-real-model")
    (tmp_path / "old.yml").write_text("not parsed", encoding="utf-8")
    before = path.read_bytes()

    legacy, registered = AiModelCatalog(path, tmp_path).list_records()

    assert legacy.model_id == "a"
    assert legacy.model_name == "legacy"
    assert legacy.input_dim == "132300 x 1"
    assert legacy.accuracy is None
    assert legacy.model_description is None
    assert not legacy.model_file_exists and legacy.config_file_exists
    assert registered.model_file_exists and not registered.config_file_exists
    assert registered.output_dim == 1
    assert registered.accuracy == 0.0
    assert registered.update_date == "2026-01-02"
    assert registered.model_description == "stored description"
    assert path.read_bytes() == before


def test_missing_registry_table_is_an_error_not_empty_success(tmp_path):
    path = tmp_path / "empty.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(ModelCatalogError, match="模型目录读取失败"):
        AiModelCatalog(path, tmp_path).list_records()


@pytest.mark.parametrize("config, kind", [
    ({}, "MissingAiModel"),
    ({"analyse_model_name": "candidate"}, "AiInferenceUnavailable"),
])
def test_runtime_is_explicitly_blocked(config, kind):
    assert ai_execution_issue(config)[0] == kind
