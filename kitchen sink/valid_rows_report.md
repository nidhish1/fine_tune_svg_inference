# Raw 1000 Valid Rows Report

Source file: `/Users/mudrex/Desktop/midterm-4/kitchen sink/1000_raw_submissions.csv`

## Summary
- Total raw rows: 1000
- Rows with full `<svg ... </svg>` block: 347
- Valid XML rows (before repair): 346
- Invalid XML among full blocks: 1
- Pre-repair valid rate: 0.3460

## Marker Coverage (raw generations)
- `serialization_target:`: 278
- `serialization:`: 309
- `SVG target:`: 7
- `layout_target:`: 427
- has `<svg` start: 999

## Valid Row Quality Stats (346 rows)
- SVG length min/median/max: 151 / 864 / 1711
- `<path>` count min/median/max: 0 / 1 / 6

## Files Generated
- Valid rows CSV: `/Users/mudrex/Desktop/midterm-4/kitchen sink/valid_rows_346.csv`

## Sample Invalid XML Rows
- `ea045c7a247166f061ce504d9b7ccaab` | len=1167 | error=not well-formed (invalid token): line 1, column 1142
