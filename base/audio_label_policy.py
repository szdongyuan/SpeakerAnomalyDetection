"""Shared judgement aggregation and mapping to existing audio archive labels."""


def aggregate_judgements(values):
    statuses = [value.strip().upper() if isinstance(value, str) else "--"
                for value in values]
    if "NG" in statuses:
        return "NG"
    return "OK" if "OK" in statuses else "--"


def audio_label_for_results(items):
    status = aggregate_judgements(item.get("judgement") for item in items)
    return "not_labeled" if status == "--" else status
