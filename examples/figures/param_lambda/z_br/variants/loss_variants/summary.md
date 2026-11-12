# z_br: training variants (60 runs, seeds [0, 1, 2])

Test paths. Spread vs the exact committor: "std" = the report's level set (q = 0.0001, 0.00023, 0.00053, 0.0012, 0.0028, 0.0064, 0.015, 0.034, 0.078, 0.41, 0.95); "unrel." = levels the exact committor reaches but the report drops because q_np is unreliable there (too few frames or label noise; q = 0.18). Selected: the lean run with the lowest validation SD(Z_q), per seed; mean ± SD over seeds.

| variant | overrides | selected runs | spread std | spread unrel. | test SD(Z_q) | step ratio | median over all runs: spread std / unrel. | train s (all runs) |
|---|---|---|---|---|---|---|---|---|
| q_np | | | 0.0494 | 0.0818 | 2.260 | 0.992 | | |
| exact | | | 0.0124 | 0.0172 | 2.190 | 0.636 | | |
| base | (as configured) | 128x128x128 β=0, 128x128x128 β=0, 128x128x128 β=10 | 0.0474 ± 0.0074 | 0.117 ± 0.010 | 2.252 ± 0.081 | 0.752 ± 0.008 | 0.0456 / 0.1261 | 228 |
| noise_1e-3 | train.noise_q=0.001 | 128x128x128 β=0, 128x128x128 β=0, 64x64 β=10 | 0.0443 ± 0.0011 | 0.148 ± 0.047 | 2.167 ± 0.035 | 0.657 ± 0.044 | 0.0439 / 0.1149 | 427 |
| noise_1e-2 | train.noise_q=0.01 | 128x128x128 β=0, 64x64 β=0, 64x64 β=0 | 0.0591 ± 0.0028 | 0.160 ± 0.016 | 2.252 ± 0.033 | 0.646 ± 0.031 | 0.0479 / 0.1456 | 473 |
| huber_1 | m3.huber=1.0 | 128x128x128 β=0, 128x128x128 β=0, 128x128x128 β=10 | 0.0469 ± 0.0062 | 0.114 ± 0.035 | 2.251 ± 0.063 | 0.752 ± 0.007 | 0.0406 / 0.1145 | 425 |
| noise_1e-3_huber_1 | train.noise_q=0.001; m3.huber=1.0 | 128x128x128 β=0, 128x128x128 β=0, 128x128x128 β=0 | 0.0444 ± 0.0025 | 0.159 ± 0.054 | 2.146 ± 0.051 | 0.657 ± 0.005 | 0.0435 / 0.1286 | 416 |
