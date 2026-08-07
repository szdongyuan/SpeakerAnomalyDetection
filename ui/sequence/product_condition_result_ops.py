class SequenceWidgetProductConditionResultOpsMixin:
    @staticmethod
    def _normalize_product_condition_result_label(value):
        normalized = str(value or "").strip()
        lowered = normalized.lower()
        if lowered == "ok":
            return "OK"
        if lowered == "ng":
            return "NG"
        if lowered in (
            "not_labeled",
            "not labeled",
            "none",
            "-",
            "null",
        ):
            return "not_labeled"
        return ""

    def _product_condition_record_label(self, record):
        if not isinstance(record, dict):
            return ""
        signal_info = record.get("recorded_signal_info") or {}
        label = signal_info.get("labels") or record.get("result_label")
        normalize_label = getattr(
            self,
            "_normalize_recent_session_storage_label",
            None,
        )
        if callable(normalize_label):
            normalized = normalize_label(label)
            if normalized in ("OK", "NG", "not_labeled"):
                return normalized
        return self._normalize_product_condition_result_label(label)

    def _collect_product_condition_records(self, group_id):
        group_id = str(group_id or "").strip()
        if not group_id:
            return None

        recent_panel = getattr(self, "recent_session_panel", None)
        panel_groups = getattr(recent_panel, "group_records", None)
        panel_group = None
        if isinstance(panel_groups, dict):
            candidate = panel_groups.get(group_id)
            if isinstance(candidate, dict):
                panel_group = candidate

        records = {}
        results = {}
        group_info = {
            "group_id": group_id,
            "records": records,
            "results": results,
        }
        if panel_group:
            for field in ("barcode", "product_model", "time_text"):
                if panel_group.get(field):
                    group_info[field] = panel_group.get(field)

        session_records = getattr(self, "recent_test_session_by_id", {}) or {}
        if panel_group:
            for condition_key, session_id in (
                panel_group.get("session_ids") or {}
            ).items():
                record = session_records.get(session_id)
                if isinstance(record, dict):
                    records[str(condition_key)] = record

        for record in session_records.values():
            if not isinstance(record, dict):
                continue
            if str(record.get("group_id") or "").strip() != group_id:
                continue
            condition_key = str(
                record.get("condition_key") or record.get("mode") or ""
            ).strip()
            if not condition_key:
                continue
            records[condition_key] = record

        import_records = getattr(self, "_condition_record_cache", {}) or {}
        for condition_key, record in import_records.items():
            if not isinstance(record, dict):
                continue
            if str(record.get("source_type") or "").strip() != "imported":
                continue
            if str(record.get("group_id") or "").strip() != group_id:
                continue
            key = str(
                record.get("condition_key") or condition_key or ""
            ).strip()
            if key:
                records.setdefault(key, record)

        for condition_key, record in records.items():
            label = self._product_condition_record_label(record)
            if label:
                results[str(condition_key)] = label
            for field in ("barcode", "product_model", "time_text"):
                if not group_info.get(field) and record.get(field):
                    group_info[field] = record.get(field)

        if panel_group:
            for condition_key, label in (
                panel_group.get("results") or {}
            ).items():
                key = str(condition_key or "").strip()
                normalized = self._normalize_product_condition_result_label(
                    label
                )
                if not key or not normalized:
                    continue
                results.setdefault(key, normalized)

        active_group_id = str(
            getattr(self, "_manual_product_condition_group_id", "") or ""
        ).strip()
        if group_id == active_group_id:
            manual_results = (
                getattr(self, "_manual_product_condition_results", {}) or {}
            )
            for condition_key, label in manual_results.items():
                key = str(condition_key or "").strip()
                normalized = self._normalize_product_condition_result_label(
                    label
                )
                if not key or not normalized or key in results:
                    continue
                results[key] = normalized

            completed_keys = set(
                getattr(
                    self,
                    "_manual_product_condition_completed_keys",
                    set(),
                )
                or set()
            )
            for condition_key in completed_keys:
                key = str(condition_key or "").strip()
                if not key or key in results:
                    continue
                results[key] = "not_labeled"

        return group_info if records or results else None

    def _product_group_result_state(self, group_id):
        group = self._collect_product_condition_records(group_id)
        if not isinstance(group, dict):
            return False, None

        condition_results = []
        results = group.get("results") or {}
        for index, condition in enumerate(self._product_condition_sequence()):
            key = self._product_condition_runtime_key(condition, index)
            label = self._normalize_product_condition_result_label(
                results.get(key)
            )
            if label not in ("OK", "NG", "not_labeled"):
                return False, None
            condition_results.append(label)

        if not condition_results:
            return False, None
        if "NG" in condition_results:
            return True, "NG"
        if "not_labeled" in condition_results:
            return True, "not_labeled"
        return True, "OK"
