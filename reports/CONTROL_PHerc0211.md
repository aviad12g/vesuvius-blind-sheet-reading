# Known-answer control on PHerc. 211 (16 Nov 2026)

Reference: the published First Letters mesh, 16-plane model input and Studio probability map for PHerc. 211 (Priyank Aranke, CC BY-NC 4.0).
One A40, about 1 h 25 min.

## What was measured
| Test | Result |
|---|---|
| Our Studio code on the winner's published 16-plane input | 0.988 correlation with his published map |
| Our full chain (raw scan -> render -> Studio) on the winner's mesh | 0.974, letters legible |
| Other face of his mesh | -0.04 (face convention confirmed: Studio "forward" = ink) |
| His surface shifted by a constant -4 / -2 / +2 / +4 voxels | 0.88 / 0.95 / 0.92 / 0.72 |
| Our automatic sheet (z7920_w020 canvas 18), old settings with --flatten | 0.73 overall, 0.755 in his clearest-letter region (blobs) |
| Same, without --flatten, offset 0 | 0.79 overall, 0.842 in his clearest-letter region (partial letters) |
| His exact heights rendered through our canvas geometry | 0.85 (so the canvas geometry itself costs about 0.12) |
| Template alignment to the CT profile about his surface | no gain (0.71) |
| Smoothing our sheet (8 px) | no gain (0.84) |

Our tracked sheet versus his surface: median 1.1 voxels off, 67% within 2 voxels, 78% within 3; in his clearest-letter region 86% within 2.

## Conclusions
1. The blind sweep finds real ink: its top line on that band sits on the winner's sheet.
2. `--flatten` damages the read and was used in every earlier depth test. Do not use it.
3. The remaining gap to legible letters is the geometry of our canvases (stretch about 8% median, locally far more, and normals taken from the base mesh), not the model.
