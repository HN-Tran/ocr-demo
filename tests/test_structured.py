from app.services.structured import parse_structured_output


def test_parse_structured_output_valid_json():
    raw = '{"vendor":"ACME","invoice_number":"INV-1","total":"10.00"}'
    result = parse_structured_output(raw, ["vendor", "invoice_number", "total"])
    assert result.data == {
        "vendor": "ACME",
        "invoice_number": "INV-1",
        "total": "10.00",
    }
    assert result.warnings == []


def test_parse_structured_output_missing_fields():
    raw = '{"vendor":"ACME"}'
    result = parse_structured_output(raw, ["vendor", "invoice_number"])
    assert result.data == {"vendor": "ACME", "invoice_number": None}
    assert any("Erwartetes Feld fehlt: invoice_number" in x for x in result.warnings)


def test_parse_structured_output_malformed():
    raw = "not-json"
    result = parse_structured_output(raw, ["vendor"])
    assert result.data is None
    assert result.warnings
