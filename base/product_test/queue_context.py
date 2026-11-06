"""Bridge existing queue editors to station WAV positions without device coupling."""
import copy
import os

from base.load_config import LoadUiConfig
from base.product_test.config_validation import validate_structure, validate_references
from base.test_queue_catalog import queue_info_from_data
from consts import error_code


def wav_positions(drafts):
    counts = [len(station["channels"]) for draft in drafts
              for station in draft.data.get("stations", [])]
    return list(range(max(counts, default=1))) or [0]


def validate_queue_save(payload, path, manager, drafts=()):
    """Validate item channel positions in every referencing station's WAV."""
    key = os.path.normcase(os.path.abspath(path))
    catalog = manager.load_queue_catalog()
    names = {name for name, info in catalog.items() if os.path.normcase(os.path.abspath(info["path"])) == key}
    names.add(os.path.splitext(os.path.basename(path))[0])
    for name in names:
        catalog[name] = queue_info_from_data(path, payload)
    projects = [(d.data, True) for d in drafts]
    for entry in manager.load_registry()["configs"]:
        project_path = os.path.join(manager.program_dir, entry["file"])
        code, project = LoadUiConfig.load_data_from_json(project_path)
        if code == error_code.OK and isinstance(project, dict) and "conditions" in project:
            projects.append((project, False))
    errors = []
    for project, is_draft in projects:
        # The reference scanner reports unreadable conditions separately. They
        # must not prevent validation of other readable product references.
        rows = project.get("conditions", []) if isinstance(project, dict) else []
        if not isinstance(rows, list):
            continue
        conditions = [c for c in rows if isinstance(c, dict)
                      and isinstance(c.get("test_queue"), str) and c["test_queue"] in names]
        if not conditions:
            continue
        checked = copy.deepcopy(project)
        checked["conditions"] = copy.deepcopy(conditions)
        if is_draft:
            # Parent drafts can be incomplete while a queue is being edited.
            # Check established mappings, not unrelated product required fields.
            mapped = {s["station_id"] for s in checked["stations"]
                      if any(c["signal_type"] for c in s["channels"])}
            for condition in checked["conditions"]:
                condition["applicable_station_ids"] = [
                    key for key in condition["applicable_station_ids"] if key in mapped]
            problems = []
        else:
            problems = validate_structure(checked)
        if not problems:
            problems = validate_references(checked, catalog, allow_unassigned_channels=is_draft)
        name = project.get("project_name") or "未命名项目"
        errors.extend(f"{name}：{error}" for error in problems)
    return errors
