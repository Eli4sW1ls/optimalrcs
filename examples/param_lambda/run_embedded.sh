#!/usr/bin/env bash
# Embedding test of one embedded system (configs/<system>.toml with an [embed] table):
#   ./run_embedded.sh z_emb          embed -> screen -> fit --lean -> evaluate -> export
#   ./run_embedded.sh cc_emb_np      the same, with an NPNEq fit on the embedded CVs first (fit-qnp-embedded)
# Then: python embed_report.py <systems...>
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/elias/anaconda3/envs/test/bin/python}
LOGS=../figures/param_lambda/logs
mkdir -p "$LOGS"
for s in "$@"; do
    if grep -q '^qnp_fit_args' "configs/$s.toml"; then
        echo "$(date +%T) fit-qnp-embedded $s"
        "$PY" -u run.py fit-qnp-embedded "$s" > "$LOGS/qnp_$s.log" 2>&1
    fi
    for step in embed screen fit evaluate export; do
        echo "$(date +%T) $step $s"
        extra=""; [ "$step" = fit ] && extra="--lean"
        # export times the one-frame forward pass: single-threaded BLAS, as in the sampler
        if [ "$step" = export ]; then export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; fi
        "$PY" -u run.py "$step" "$s" $extra > "$LOGS/${step}_$s.log" 2>&1
    done
done
echo "$(date +%T) done"
