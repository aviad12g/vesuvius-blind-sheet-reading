# Unstretched re-mesh, ranking shortcuts, 3D model and depth stack (16 Nov 2026)

## Method
`tools9.py remesh` turns one tracked sheet (a canvas from the sweep) into its own tifxyz mesh: rows are cross-sections of the sheet at constant z, columns are arclength along each cross-section, rows are squared up against their neighbours. The mesh is then read with `sweep.py --base-only` (no `--flatten`), Studio stride 32.

## PHerc0211 control (winner's clearest-letter region, 0.76 cm2)
| Read | Match with winner's map |
|---|---|
| Old canvas read with --flatten | 0.755 |
| Canvas read without --flatten | 0.842 |
| Unstretched mesh, automatic | 0.90-0.91 |
| Winner's own mesh, our code | 0.974 |

New-mesh quality on the control: node step 19.99 along rows, 20.25 down columns, axes 2 degrees off square (median), scale within 2% of the winner's flattening.

## Ranking from quarter-resolution maps: fails its own control
No GPU. 6,799 windows (2 cm wide) scored across PHerc0211 and PHerc0813 from the first sweep's quarter-resolution maps.

On PHerc0211, where the answer is known, none of the measures separates the ink face from the blank face:
| Measure (median over windows) | Ink face | Other face |
|---|---|---|
| Letter-sized marks per cm2 | 9.0 | 11.2 |
| Studio dark % | 6.8 | 10.8 |
| Agreement between Studio and TRACE on the same face | 0.44 | 0.40 |

The winner's own sheet scores 12-17 marks per cm2, and the blank face of the same windows scores 8-17. So a ranking of an unread scroll built on these maps cannot be trusted, and none is reported.

"A second ink model agrees" and "the two faces are uncorrelated" are therefore not evidence of ink: both hold equally on the blank face of the control.

## 3D ink model and depth stack
### 3D ink model (scrollprize/ink_3d_dino_guided, ckpt_78k_fullsup): fails the control
Loads cleanly (0 missing, 0 unexpected weights). Trained on PHerc. Paris 4 at 2.4 um / 78 keV. On two 256^3 blocks at the winner's letters on PHerc0211 (9.362 um), native and 4x upsampled, its output sampled on his surface correlates -0.15 to 0.00 with his letters. Not usable on these scans.

### Depth stack (same unstretched mesh read at offsets along its normal, Studio stride 48)
| | match with depth 0 at +-3 / +-6 / +-9 px | dark % at 0 / +-6 / +-9..12 |
|---|---|---|
| PHerc0211 control, ink side (0.90 match with winner at depth 0) | 0.94, 0.92 / 0.84, 0.61 / 0.46, -0.02 | 24 / 9-12 / 2-5 |
| PHerc0211 control, other side | 0.66, 0.30 / 0.52, 0.17 / 0.47, 0.12 | 8 / 10-33 / 10-40 |

One control patch per condition.
