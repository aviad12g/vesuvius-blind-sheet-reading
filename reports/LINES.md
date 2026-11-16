# Text-line spacing test (16 Nov 2026)

Idea: Greek text sits in evenly spaced lines, so ink that is too faint to read should still repeat at one spacing along the scroll axis.

## Method
1. Quarter-resolution sweep maps (4 voxels per cell, Studio model) and the tracked sheets (`canv.npz`) from `sweep.py`.
2. Band edges removed (first 48 and last 28 rows of each band: the model outputs zeros there, which fakes a repeat at the band step).
3. The same sheet is matched across neighbouring z-bands by 3D distance (median 1.6 voxels on the control) and stitched into tall strips (`lines/st.py`, `lines/st2.py`).
4. Per window: mean ink score per row, smoothed, autocorrelation, strongest repeat between 2.7 and 8 mm (`lines/rule.py`).
5. Rule: two neighbouring windows agree on the spacing within 6 cells, both with autocorrelation >= 0.55. The rule was set after looking at the first control strips.

## Control: PHerc. 211, winding 20, z 6112-9040, 18 stitched sheets
- Winner's own map: repeat 5.0-5.6 mm in all 13 windows (median 5.1 mm).
- Our automatic quarter-resolution read: the only Studio ink-face hit is the prize-winning sheet, spacing 4.92-4.96 mm, in all three settings.
- Other face and the TRACE model: 0 hits in two settings, 1 hit (5.9 mm, TRACE, other face) in one setting. About 1 false hit per 250-300 null windows.
- TRACE does not flag the proven sheet.

## Extension with criteria written first (PREREG_211_extension.md): outcome WEAK
Six more bands, 42 stitched sheets in total. The rule still flags only the prize-winning sheet: no new sheet on winding 40 or on the extended winding-20 strips, and no other-face hits in the new data.

## Unread scroll (PHerc. 813)
163 stitched sheets across windings 20-80. No hit in two settings (193 and 281 windows); one hit in the loosest setting (406 windows), which is a patchy 7 mm strip and is consistent with the false-hit rate.

## Reading
Specific, but insensitive: the rule fires where the read is already nearly legible and does not reveal fainter text. It is documented as a diagnostic, not offered as a search tool.
