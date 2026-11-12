# cc_br: training variants (20 runs)

Test paths. Spread vs the exact committor: "std" = the report's level set (q = 0.0012, 0.0028, 0.0064, 0.015, 0.034, 0.078, 0.18, 0.41, 0.95); "unrel." = levels the exact committor reaches but the report drops because q_np is unreliable there (too few frames or label noise; q = 0.0001, 0.00023, 0.00053).

| variant | overrides | selected run | spread std | spread unrel. | test SD(Z_q) | step ratio | median over runs: spread std / unrel. | train s (all runs) |
|---|---|---|---|---|---|---|---|---|
| q_np | | | 0.0185 | 0.3591 | 3.521 | 0.919 | | |
| exact | | | 0.0130 | 0.1766 | 3.545 | 0.893 | | |
| base | (as configured) | 128x128x128 β=0 | 0.0178 | 0.2625 | 3.549 | 0.794 | 0.0162 / 0.3070 | 28 |
| noise_1e-3 | train.noise_q=0.001 | 128x128x128 β=10 | 0.0147 | 0.1906 | 3.524 | 0.847 | 0.0155 / 0.2077 | 27 |
| noise_1e-2 | train.noise_q=0.01 | 128x128x128 β=10 | 0.0206 | 0.2497 | 3.509 | 0.790 | 0.0204 / 0.2514 | 31 |
| huber_1 | m3.huber=1.0 | 128x128x128 β=10 | 0.0151 | 0.4897 | 3.507 | 0.917 | 0.0151 / 0.2699 | 29 |
| noise_1e-3_huber_1 | train.noise_q=0.001; m3.huber=1.0 | 128x128x128 β=10 | 0.0134 | 0.1967 | 3.533 | 0.893 | 0.0144 / 0.1952 | 28 |
