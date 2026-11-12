#!/usr/bin/env bash
# Regenerate every number of docs/param_lambda_report.md from the configs (fixed seeds).
#
#   ./run_all.sh                    all systems: z_br cc_br flat1d bumps1d
#   ./run_all.sh z_br               one system
#   PY=/path/to/python ./run_all.sh
#
# Needs: numpy, scipy, scikit-learn, torch (CUDA optional), cupy (q_np fits and the TIS plots), tistools,
# and the toytis runs named in the configs. Long: launch it as its own unit, e.g.
#   systemd-run --user --unit=pl-all --collect -p MemoryMax=12G -p MemorySwapMax=0 \
#       --working-directory=$PWD ./run_all.sh
# Logs: <system folder>/logs/steps.log (common.out_dir, below examples/figures_by_potential/) and examples/figures_by_potential/_shared/param_lambda/logs/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/elias/anaconda3/envs/test/bin/python}
LOGS=../figures_by_potential/_shared/param_lambda/logs
mkdir -p "$LOGS"
SYSTEMS=${*:-"z_br cc_br flat1d bumps1d"}
for s in $SYSTEMS; do
    # systems without a saved q_np get an NPNEq fit first (load_tis_data_cupy.py, seeded)
    if [ "$s" != "z_br" ]; then
        "$PY" -u run.py fit-qnp "$s" > "$LOGS/qnp_$s.log" 2>&1
    fi
    for step in extract screen fit evaluate export; do
        echo "$(date +%T) $step $s"
        "$PY" -u run.py "$step" "$s" > "$LOGS/${step}_$s.log" 2>&1
    done
done
"$PY" -u run.py report
"$PY" -m pytest tests -q
