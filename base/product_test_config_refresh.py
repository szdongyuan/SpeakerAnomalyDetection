"""Prepare and compare product configurations without touching runtime or widgets."""

from dataclasses import dataclass
from consts import error_code
from base.product_test.config_model import RUNTIME_UNAVAILABLE


class ProductTestConfigurationError(ValueError):
    """The saved product cannot be used until its configuration is repaired."""


class ProductTestRuntimeUnavailableError(ProductTestConfigurationError):
    """Configuration is readable, but its execution pipeline is not implemented."""


@dataclass(frozen=True)
class ProductTestRefreshSnapshot:
    active_file: str
    signature: str
    conditions: list
    context: dict
    queue_catalog: dict
    validation: dict


def build_product_test_refresh_snapshot(manager, active_file):
    """Reject station configurations until the new execution pipeline is connected."""
    active_file = str(active_file or "").strip()
    if not active_file:
        return ProductTestRefreshSnapshot("", "", [], {}, {}, {})

    load_code, project = manager.load_project(active_file)
    if load_code != error_code.OK or not isinstance(project, dict):
        raise ProductTestConfigurationError(f"产品配置无法读取：{active_file}")
    raise ProductTestRuntimeUnavailableError(RUNTIME_UNAVAILABLE)
