SCHEMA_REGISTRY: dict[str, dict] = {
    "invoice_basic": {
        "description": "Grundlegende Rechnungsfelder.",
        "fields": {
            "vendor": "string",
            "invoice_number": "string",
            "invoice_date": "string",
            "due_date": "string",
            "total": "string",
            "currency": "string",
        },
    },
    "receipt_basic": {
        "description": "Grundlegende Belegfelder.",
        "fields": {
            "merchant": "string",
            "date": "string",
            "total": "string",
            "tax": "string",
            "currency": "string",
        },
    },
}
