# New Repairability Analysis (Independent)

Source: `/Users/mudrex/Desktop/midterm-4/kitchen sink/1000_raw_submissions.csv`
Total rows: 1000

## Outcome Buckets
- easy_e2_trim_to_last_gt: 651
- already_valid: 346
- no_svg_start: 1
- full_block_but_invalid: 1
- hard_missing_svg_close+dangling_open_tag: 1

## Easy Repair Counts
- trim_to_last_gt: 651
- total_easy_repairable: 651

## Examples by Bucket (up to 5 IDs)
- easy_e2_trim_to_last_gt: ['eb25bd8aa69ff58fffe2f50db471c4a9', '2e7630a66e4c461b4ba8832fb2c9eb05', '6cd2ff67e8c40d994115dc735aab04ab', '5f5052fc86f7283c6e0f4b021df0913d', '73f19b688a39c0cdf6157fb4e2c4537e']
- already_valid: ['c499d6062b8a598d6b5838d4792d7a41', 'd584789a90321b0e5b02f5b57b783d22', '8e1fadf0ca78b939adfb003d4e011a0e', 'e3e0fff5ead535daf599d302dd645b45', 'ef0950dbc0cfd0a6b9e64593180580ee']
- no_svg_start: ['bb003bdd403f4a9bad8b1c3b138f4060']
- full_block_but_invalid: ['ea045c7a247166f061ce504d9b7ccaab']
- hard_missing_svg_close+dangling_open_tag: ['cc1cd78fa1802e71cb180157748d7a21']

## Files
- easy IDs CSV: `/Users/mudrex/Desktop/midterm-4/kitchen sink/easy_repair_ids.csv`