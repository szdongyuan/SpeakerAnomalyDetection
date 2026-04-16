def _is_ai_analysis_config(item_config) -> bool:
    return isinstance(item_config, dict) and str(item_config.get("type") or "").upper() == "AI"


def normalize_ai_runtime_label(label):
    normalized = str(label or "").strip().upper()
    if normalized in ("OK", "NG"):
        return normalized
    return None


def has_enabled_ai_analysis(analysis_config) -> bool:
    for key in (analysis_config or {}).get("display_sequence", []) or []:
        if _is_ai_analysis_config((analysis_config or {}).get(key, {})):
            return True
    return False


def extract_ai_runtime_state(analysis_window, analysis_config) -> dict:
    state = {
        "has_ai_analysis": has_enabled_ai_analysis(analysis_config),
        "label": None,
        "scores": {"ok_score": None, "ng_score": None},
        "blocked_message": "",
    }
    for instance in analysis_window or []:
        instance_key = getattr(instance, "_sequence_analysis_key", None)
        item_config = (analysis_config or {}).get(instance_key, {}) if instance_key else {}
        if not _is_ai_analysis_config(item_config):
            continue

        detail = getattr(instance, "export_detail", None)
        if isinstance(detail, dict):
            if state["scores"]["ok_score"] in (None, "") and state["scores"]["ng_score"] in (None, ""):
                ok_score = detail.get("ok_score")
                ng_score = detail.get("ng_score")
                if ok_score not in (None, "") or ng_score not in (None, ""):
                    state["scores"] = {
                        "ok_score": ok_score,
                        "ng_score": ng_score,
                    }
            if not state["blocked_message"]:
                blocked_message = str(detail.get("blocked_message") or "").strip()
                if blocked_message:
                    state["blocked_message"] = blocked_message
            if state["label"] is None:
                normalized = normalize_ai_runtime_label(detail.get("label"))
                if normalized:
                    state["label"] = normalized

        if state["label"] is None:
            normalized = normalize_ai_runtime_label(getattr(instance, "result", None))
            if normalized:
                state["label"] = normalized
    return state


def count_judged_results(result_dict) -> int:
    if not isinstance(result_dict, dict):
        return 0
    count = 0
    for value in result_dict.values():
        if not isinstance(value, tuple) or len(value) != 2:
            continue
        ok, _deviation = value
        if ok is None:
            continue
        count += 1
    return count
