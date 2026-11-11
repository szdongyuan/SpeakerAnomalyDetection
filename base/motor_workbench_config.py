"""Manage the source identity of successfully loaded workbench configurations."""
from pathlib import Path
from base import workbench_configuration_loader as loader
from consts.running_consts import DEFAULT_DIR, PRODUCT_TEST_PROGRAM_REGISTRY_PATH, SEQUENCE_CONFIG_REGISTRY_PATH


class MotorWorkbenchConfigRepository:
    """Load the plot layout, active product and its referenced queues from saved files."""

    def __init__(
        self,
        program_registry_path=PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
        queue_registry_path=SEQUENCE_CONFIG_REGISTRY_PATH,
        plot_layout_path=Path(DEFAULT_DIR) / "ui" / "ui_config" / "motor_workbench_plot_layout.json",
    ):
        self._program_registry_path = Path(program_registry_path)
        self._queue_registry_path = Path(queue_registry_path)
        self._plot_layout_path = Path(plot_layout_path)
        self._source_identity = ""

    @property
    def source_identity(self):
        return self._source_identity

    def load_product_catalog(self) -> tuple[tuple[str, str], ...]:
        registry = loader._read_product_registry(self._program_registry_path)
        return tuple((entry["file"], entry["project_name"]) for entry in registry["configs"])

    def load_candidate(self, file_name: str):
        registry = loader._read_product_registry(self._program_registry_path)
        if not any(entry["file"] == file_name for entry in registry["configs"]):
            raise loader.WorkbenchConfigurationError(
                f"{self._program_registry_path}: product {file_name!r} is not registered"
            )
        return loader._load_active_configuration(
            self._program_registry_path, self._queue_registry_path, self._plot_layout_path, file_name,
        )

    def accept_product_source(self, file_name: str) -> None:
        self._source_identity = str(self._program_registry_path.parent / file_name)

    def load_active(self):
        active_file = loader._active_product_filename(self._program_registry_path)
        config = loader._load_active_configuration(
            self._program_registry_path, self._queue_registry_path, self._plot_layout_path, active_file,
        )
        self._source_identity = str(self._program_registry_path.parent / active_file) if active_file else ""
        return config
