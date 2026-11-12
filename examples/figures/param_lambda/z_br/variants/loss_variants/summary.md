# z_br: training variants (20 runs)

Test paths. Spread vs the exact committor: "std" = the report's level set (q = 0.0001, 0.00023, 0.00053, 0.0012, 0.0028, 0.0064, 0.015, 0.034, 0.078, 0.41, 0.95); "unrel." = levels the exact committor reaches but the report drops because q_np is unreliable there (too few frames or label noise; q = 0.18).

| variant | overrides | selected run | spread std | spread unrel. | test SD(Z_q) | step ratio | median over runs: spread std / unrel. | train s (all runs) |
|---|---|---|---|---|---|---|---|---|
| q_np | | | 0.0494 | 0.0818 | 2.260 | 0.992 | | |
| exact | | | 0.0124 | 0.0172 | 2.190 | 0.636 | | |
| base | (as configured) | 128x128x128 β=0 | 0.0550 | 0.1126 | 2.293 | 0.761 | 0.0499 / 0.1368 | 77 |
| noise_1e-3 | train.noise_q=0.001 | 128x128x128 β=0 | 0.0433 | 0.1994 | 2.156 | 0.686 | 0.0434 / 0.1221 | 81 |
| noise_1e-2 | train.noise_q=0.01 | 128x128x128 β=0 | 0.0580 | 0.1727 | 2.226 | 0.652 | 0.0519 / 0.1555 | 76 |
| huber_1 | m3.huber=1.0 | 128x128x128 β=0 | 0.0470 | 0.1364 | 2.215 | 0.760 | 0.0427 / 0.1099 | 77 |
| noise_1e-3_huber_1 | train.noise_q=0.001; m3.huber=1.0 | 128x128x128 β=0 | 0.0471 | 0.2215 | 2.130 | 0.662 | 0.0428 / 0.1266 | 77 |
