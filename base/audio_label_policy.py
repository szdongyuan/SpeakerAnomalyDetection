"""Shared judgement aggregation and mapping to existing audio archive labels."""

INCOMPLETE = "结果不完整"
NG_INCOMPLETE = "NG（结果不完整）"


def _identity(item):
    return item.get("config_key"), item.get("analysis_type"), item.get("channel_index")


def required_judgements(instances):
    """Freeze the identities required for a verdict; imports no algorithms or UI."""
    return [dict(config_key=i.config_key, analysis_type=i.analysis_type, channel_index=i.channel_index)
            for i in instances if i.parameters.get("limit_checked", False)]


def judgement_summary(items, expected=None):
    items = [{**item, "judgement": (item["judgement"].strip().upper()
              if isinstance(item.get("judgement"), str) else None)}
             for item in items]
    inferred = expected is None
    if inferred:
        expected = [i for i in items if i.get("judgement_required", (
            i.get("judgement_status") in {"completed", "failed"} or i.get("judgement") in {"OK", "NG"}
        ))]
    groups = {}
    for item in items:
        groups.setdefault(_identity(item), []).append(item)
    missing, failed, verdicts = [], [], []
    for required in expected:
        matches = [required] if inferred else groups.get(_identity(required), [])
        descriptor = {key: required.get(key) for key in ("config_key", "analysis_type", "channel_index")}
        if not matches:
            missing.append(descriptor)
        elif len(matches) != 1:
            failed.append(descriptor)
        else:
            item = matches[0]
            if (item.get("execution_status", "分析完成" if inferred else None) != "分析完成"
                    or item.get("judgement_status", "completed" if inferred else None) != "completed"
                    or item.get("judgement") not in {"OK", "NG"}):
                failed.append(descriptor)
            else:
                verdicts.append(item["judgement"])
    complete = not missing and not failed
    verdict = "NG" if "NG" in verdicts else ("OK" if complete and expected else None)
    status = (verdict or "--") if complete else (NG_INCOMPLETE if verdict == "NG" else INCOMPLETE)
    return {"status": status, "complete": complete, "judgement": verdict,
            "required_count": len(expected), "missing_required": missing, "failed_required": failed}


def result_status(result):
    """Accept a task payload or a legacy item list at existing display boundaries."""
    if isinstance(result, dict):
        summary = result.get("judgement_summary")
        return summary["status"] if summary is not None else judgement_summary(result.get("instance_results", ()))["status"]
    return judgement_summary(result)["status"]


def reconcile_terminal_judgement(payload, expected, early_items):
    """Keep authenticated evidence on runtime loss; flag contradictory terminals."""
    items = payload.get("instance_results", ())
    summary = judgement_summary(items, expected)
    if early_items is None:
        return summary, False
    early = judgement_summary(early_items, expected)
    if payload.get("error") is not None and not items:
        return early, False
    required = {_identity(item) for item in expected}

    def evidence(values):
        grouped = {}
        for item in values:
            key = _identity(item)
            if key in required:
                verdict = item.get("judgement")
                verdict = verdict.strip().upper() if isinstance(verdict, str) else None
                grouped.setdefault(key, []).append((item.get("execution_status"),
                                                     item.get("judgement_status"), verdict))
        return grouped

    if summary == early and evidence(items) == evidence(early_items):
        return summary, False
    # A contradiction must never publish OK. Preserve any authenticated NG,
    # and reuse existing incomplete semantics for all downstream consumers.
    verdict = "NG" if "NG" in (summary["judgement"], early["judgement"]) else None
    return {**summary, "complete": False, "judgement": verdict,
            "status": NG_INCOMPLETE if verdict else INCOMPLETE,
            "failed_required": list(expected)}, True

def aggregate_judgements(values):
    statuses = [value.strip().upper() if isinstance(value, str) else "--"
                for value in values]
    if any(INCOMPLETE in value for value in statuses):
        return NG_INCOMPLETE if any(value.startswith("NG") for value in statuses) else INCOMPLETE
    if "NG" in statuses:
        return "NG"
    return "OK" if "OK" in statuses else "--"


def audio_label_for_results(result):
    status = result_status(result)
    return "NG" if status.startswith("NG") else ("OK" if status == "OK" else "not_labeled")
