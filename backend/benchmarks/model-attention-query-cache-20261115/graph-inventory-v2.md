# Strict graph trace validation

Use run_monitored_v2.py for subsequent paired long-context diagnostics. The original runner and its settings proof remain unchanged.

The earlier graph check counted six usable regex matches but could ignore additional malformed, zero-capture, or non-reusing records. The new check requires exactly six complete graph lines, each with positive captures and more replays than captures. Extra, malformed, prefixed, truncated, and non-reusing records reject the inventory. Unrelated log lines remain permitted.

Run test_graph_inventory_v2.py for the 16-case offline inventory check, and audit_settings_v2.py to verify both effective environments against validated C17/C23 generation settings. Neither command launches GPU work. settings-audit-v2.json binds the revised runner to this settings check. No benchmark or model-performance claim follows from these checks. All existing prerequisites, numerical checks, and conservative GPU-coverage rules remain in effect.
