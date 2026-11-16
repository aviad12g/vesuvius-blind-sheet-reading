# Written 16 Nov 2026 ~14:02 (UTC+2), before any of the new maps were seen

Purpose: second check of the text-line rule on sheets that were not used to set it.

New data (pod y8ay46tn4sxcwr, A40): low-res sweep of PHerc0211 bands
- z7920_w020 (re-run, to confirm the settings reproduce the earlier maps)
- z6112/z6720/z7312/z7920 _w040 (next winding out, same height as the proven text)
- z4912/z5520 _w020 (extends the existing winding-20 strips downward)
- z9120/z9712/z10320 _w020 (winding 20 above the missing z8520 band)
Command: sweep.py --scroll PHerc0211 --D 170 --tile 1024 --marg 64 --ds 4 --studio-faces forward,reverse --trace-faces forward,reverse

Fixed rule (unchanged from LINES.md): band edges cropped 48/28 rows; stitched strips; per window the strongest
autocorrelation peak at lags 73-214 cells; HIT = two neighbouring windows whose peak lags agree within 6 cells and both peaks >= 0.55.
Settings reported: (window 250, coverage 0.35), (125, 0.35), (125, 0.25).

Known answer: line spacing on PHerc0211 is 4.9-5.1 mm (winner's map 5.1 mm, our read of the proven sheet 4.94 mm) = 126-142 cells.

Predictions if the rule detects text:
 P1. New Studio ink-face (forward) hits exist on sheets other than the proven one.
 P2. Their spacing falls in 4.7-5.3 mm.
 P3. Other-face (reverse) hits are fewer, and not concentrated at 4.7-5.3 mm.
Outcomes:
 PASS  = P1, P2 and P3 hold.
 WEAK  = no new forward hits (rule only finds the single best sheet): the test is too insensitive to justify sweeping other scrolls with it.
 FAIL  = forward and reverse hit equally often, or forward hits sit at arbitrary spacings.

## Change made at 14:41 (UTC+2), still before any new map was seen
The winding-40 bands are twice as wide as expected (21 min each instead of 9). To stay near the approved ~$1,
z7920_w040 and the three upper winding-20 bands (z9120, z9712, z10320) are dropped. The run is now:
z6112/z6720/z7312 _w040 (three bands, new sheets) and z4912/z5520 _w020 (extends the existing strips). Rule and predictions unchanged.
Check of settings: the z7920_w020 re-run gives the same 24 sheets, same coverage, and the same top-ranked chain
(sheet 18, ink face, score 4.791 vs 4.795 before).

## Outcome (16 Nov 2026, 15:50 UTC+2): WEAK
Pod y8ay46tn4sxcwr (A40), 11:56-13:41 UTC, about $1.05. Re-run of z7920_w020 matches the earlier map of the proven sheet (correlation 1.0000, max difference 1 grey level).

Winding 40, z6112-7312 (23 stitched sheets, 118 cm2, new sheets):
- hits: Studio ink face 0, Studio other face 0, TRACE ink face 0, TRACE other face 0, in all three settings (22 / 39 / 48 windows)
- strong single windows (>= 0.55) on the Studio ink face sit at 138, 152, 155, 160, 204, 208 cells: no concentration at 126-142

Winding 20 extended to z4912-7920 (19 stitched sheets, 98 cm2):
- the only hits are again the proven sheet (4.94-4.96 mm), in two of three settings; no other sheet; other faces 0 hits

P1 fails (no new ink-face hits), so P2 is untested; P3 holds (no other-face hits in the new data).
Reading: the rule is specific but insensitive. Across 42 stitched sheets of a scroll that is known to carry text, it flags one: the sheet that
already reads almost legibly. It does not reveal fainter text, so it is not a useful search tool for other scrolls at this resolution.
