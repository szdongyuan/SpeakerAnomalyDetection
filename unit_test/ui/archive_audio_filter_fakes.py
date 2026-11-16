"""Local parser substitution for tests of legacy audio artifacts."""

from base.audio_record_filter import parse_audio_filter_metadata


def use_legacy_filter_metadata(monkeypatch):
    """Keep non-filter fixtures exercising their original artifact contracts."""
    monkeypatch.setattr(
        "ui.archive_audio_data_dialog.parse_archive_audio_filter_metadata",
        parse_audio_filter_metadata,
    )
