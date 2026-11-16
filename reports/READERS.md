# Reader V3 and v8-in on the control sheet (16 Nov 2026)

One A40, about 55 minutes. Weights sha256-checked in `setup4.sh`: `reader-v3.pth` ea405417..., v8-in `model.safetensors` 3b94548d....
`sweep.py` renders a 25-layer stack. Reader V3 goes through the same loader as TRACE (17 layers); v8-in gets 24 layers, 64 px tiles, stride 21.
Face convention found on the control: Studio "forward" is the ink face; Reader V3 and v8-in read ink in "reverse".

## PHerc. 211, match with the winner's published Studio map (clearest-letter region, 0.76 cm2)
| Surface | Studio | Reader V3 | v8-in |
|---|---|---|---|
| Winner's hand-refined mesh, ink face | 0.972 | 0.710 | 0.500 |
| Winner's hand-refined mesh, other face | -0.143 | -0.019 | -0.186 |
| Our automatic re-mesh, ink face | 0.900 | 0.623 | 0.439 |
| Our automatic re-mesh, other face | -0.190 | -0.133 | -0.227 |

Reader V3 shows the same letters as Studio, fainter. v8-in is noisy on this scroll and shows its tile grid.

## Agreement between models does not separate the faces on the control
| | Studio~V3 | Studio~v8-in | V3~v8-in |
|---|---|---|---|
| Automatic re-mesh, ink face | 0.72 | 0.42 | 0.45 |
| Automatic re-mesh, other face | 0.69 | 0.48 | 0.51 |
| Winner's mesh, ink face | 0.73 | 0.58 | 0.54 |
| Winner's mesh, other face | 0.65 | 0.51 | 0.50 |

## Depth sweep on the automatic re-mesh (offsets in voxels along the normal)
| Offset | V3 dark % | V3 match with depth 0 | Studio dark % (stride 96) | Studio match with depth 0 |
|---|---|---|---|---|
| -9 | 0.8 | 0.59 | 3.4 | 0.49 |
| -6 | 8.2 | 0.89 | 12.1 | 0.84 |
| -3 | 14.0 | 0.96 | 17.4 | 0.94 |
| 0 | 15.5 | 1.00 | 24.5 | 1.00 |
| +3 | 12.2 | 0.94 | 25.9 | 0.92 |
| +6 | 5.0 | 0.58 | 8.7 | 0.60 |
| +9 | 0.7 | 0.08 | 1.6 | -0.02 |

## Combining depths
Match with the winner's map: Reader V3 single depth 0.623; mean of -3/0/+3 0.643; maximum of -3/0/+3 0.634; maximum of -6..+6 0.658.
Studio (stride 96) single depth 0.851; mean 0.848; maximum 0.826 / 0.824. No useful gain.
