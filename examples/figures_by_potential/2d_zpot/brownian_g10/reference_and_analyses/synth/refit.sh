#!/bin/bash
cd /mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures/z_brownian
P=/home/elias/anaconda3/envs/test/bin/python
$P -u synth_fit.py synth/synth_30k_s50.npz scratch 2500 0.01 full 2>&1 | grep -v "^#="
$P -u synth_fit.py synth/synth_trunc_even.npz scratch 7500 0.01 trunc_even_lag10 10 2>&1 | grep -v "^#="
$P -u synth_fit.py synth/synth_trunc_odd.npz scratch 7500 0.01 trunc_odd_lag10 10 2>&1 | grep -v "^#="
echo ALLDONE
