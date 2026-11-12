"""Read model choices without Qt or a training/inference framework."""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Protocol


class ModelCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelChoice:
    name: str
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class ModelRecord:
    """Stored registry metadata; file presence does not imply model readiness."""

    model_id: str
    model_name: str
    model_path: str
    config_path: str
    input_dim: str
    output_dim: int
    accuracy: float | None
    update_date: str
    model_description: str | None
    model_file_exists: bool
    config_file_exists: bool


class ModelRegistryReader(Protocol):
    def list_records(self) -> tuple[ModelRecord, ...]: ...


class AiModelCatalog:
    """Read the existing registry; never create a database or load model code."""

    def __init__(self, database_path, model_root):
        self.database_path = Path(database_path)
        self.model_root = Path(model_root)

    def _read_rows(self, query):
        if not self.database_path.exists():
            return ()
        try:
            with closing(sqlite3.connect(self.database_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                return db.execute(query).fetchall()
        except sqlite3.Error as error:
            raise ModelCatalogError("模型目录读取失败，请检查模型数据库。") from error

    def list_records(self) -> tuple[ModelRecord, ...]:
        rows = self._read_rows(
            "SELECT model_id, model_name, model_path, config_path, input_dim, "
            "output_dim, accuracy, update_date, model_description "
            "FROM training_model_table ORDER BY model_name"
        )
        return tuple(
            ModelRecord(
                *row,
                model_file_exists=(self.model_root / row[2]).is_file(),
                config_file_exists=(self.model_root / row[3]).is_file(),
            )
            for row in rows
        )

    def list_models(self):
        rows = self._read_rows(
            "SELECT model_name, model_path FROM training_model_table ORDER BY model_name"
        )
        choices = []
        for name, stored_path in rows:
            path = self.model_root / stored_path
            if path.suffix.lower() != ".onnx":
                reason = "不是 ONNX 模型"
            elif not path.is_file():
                reason = "模型文件不存在"
            else:
                reason = ""
            choices.append(ModelChoice(name, not reason, reason))
        return tuple(choices)
