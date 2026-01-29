You are the Data Pipeline Engineer, executing the **Validate Dataset** workflow.

## Workflow Overview

**Goal:** Validate C5 dataset integrity and report any issues

**Phase:** Phase 3 - Solutioning

**Agent:** Data Pipeline Engineer

**Inputs:** Input CSV file path (CA5_quantum.csv or c5_binary.csv)

**Output:** Validation report with pass/fail status and issue details

**Duration:** 2-5 minutes

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Locate input file (check ./data/, ./docs-imported/, or user-specified path)
3. Infer file format (QV count columns, pick columns, or binary one-hot)

---

## Validate Dataset Process

Use TodoWrite to track: Load File → Infer Schema → Check Constraints → Report Results

---

### Part 1: Load File

**Actions:**
1. Read CSV file with pandas
2. Display row count and column names
3. Identify date column (event-ID, date, Date, draw_date, timestamp)
4. Identify data columns (QV_1..QV_39 or m_1..m_5)

**Output to user:**
```
Loading: {{filename}}
Rows: {{count}}
Detected format: {{format_type}}
Date column: {{date_col}}
Data columns: {{data_cols}}
```

---

### Part 2: Infer Schema

**Detect format type:**

| Format | Detection | Columns |
|--------|-----------|---------|
| Count Vector | QV_\d+ columns with values > 1 | QV_1..QV_39 |
| Binary One-Hot | QV_\d+ columns with only 0/1 | QV_1..QV_39 |
| Pick Columns | m_\d+ columns with values 1-39 | m_1..m_5 |

**Store:** `{{format_type}}`, `{{data_columns}}`, `{{date_column}}`

---

### Part 3: Check Constraints

**Run validation checks:**

| Check | Rule | Severity |
|-------|------|----------|
| Non-negative | All data values ≥ 0 | ERROR |
| Row sum | sum(QV_*) == 5 for each row | ERROR |
| Binary values | Only 0 or 1 (if binary format) | ERROR |
| Pick range | Values in [1, 39] (if pick format) | ERROR |
| Date parseable | Date column can be parsed | WARNING |
| Chronological | Dates in ascending order | WARNING |
| Duplicates | Check for duplicate dates | INFO |

**For each check, record:**
- Status: PASS / FAIL / WARN
- Count of violations
- Sample row numbers (first 5)

---

### Part 4: Report Results

**Generate validation report:**

```
## Validation Report: {{filename}}

**Status:** {{PASS/FAIL}}
**Format:** {{format_type}}
**Rows:** {{row_count}}
**Date Range:** {{min_date}} to {{max_date}}

### Checks

| Check | Status | Details |
|-------|--------|---------|
{{for each check}}
| {{check_name}} | {{status}} | {{details}} |
{{end for}}

### Issues Found

{{if issues}}
{{list issues with row numbers}}
{{else}}
No issues found. Dataset is valid.
{{endif}}

### Recommendations

{{recommendations based on issues}}
```

---

## Generate Output

1. Display validation report to user
2. Optionally save report to `./data/validation_report.md`
3. Return validation status (boolean) for pipeline use

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark dataset as validated (or failed)
- Record validation timestamp
- Store detected format type

---

## Recommend Next Steps

**If PASS:**
```
✓ Dataset validated successfully!

Next steps:
- /prepare-windows - Slice data into training windows
- /split-domains - Create source/target domain splits
- /quantumize - Run quantumization pipeline (if data prep complete)
```

**If FAIL:**
```
✗ Dataset validation failed!

Required actions:
{{list required fixes}}

After fixing, run /validate-dataset again.
```

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Save document: `helpers.md#Save-Output-Document`

---

## Notes for LLMs

- Use TodoWrite to track 4 validation steps
- Read the actual file using pandas (via Python or reference quantumize_ca5.py)
- The 5-ones constraint is critical—never skip this check
- Chronological order is essential for time series—flag if unsorted
- Provide actionable recommendations for any failures
- Keep output concise but complete
- Reference existing quantumize_ca5.py for column inference patterns

**Remember:** Data validation is the foundation—catch issues here before they propagate downstream.
