from __future__ import annotations

import pytest

from app.config import get_settings
from app.i18n import DEFAULT_LOCALE, normalize_locale, resolve_locale, t


def test_normalize_locale_defaults_to_en() -> None:
    assert normalize_locale(None) == "en"
    assert normalize_locale("") == "en"
    assert normalize_locale("en-US") == "en"


def test_normalize_locale_german() -> None:
    assert normalize_locale("de") == "de"
    assert normalize_locale("de-DE") == "de"


def test_translate_interpolation() -> None:
    assert t("en", "page_n", n=3) == "Page 3"
    assert t("de", "page_n", n=3) == "Seite 3"


def test_resolve_locale_priority() -> None:
    assert resolve_locale(settings_locale="en", cookie="de", query="en") == "en"
    assert resolve_locale(settings_locale="en", cookie="de") == "de"
    assert resolve_locale(settings_locale="de") == "de"


def test_settings_default_locale_en(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_LOCALE", raising=False)
    assert get_settings().app_locale == DEFAULT_LOCALE
