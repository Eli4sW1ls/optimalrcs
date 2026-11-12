#!/usr/bin/env bash
# Structured copy of examples/figures, sorted by potential, then by dynamics and kind of run.
#
#   bash reorganize_figures.sh            # build figures_by_potential/ next to figures/
#   bash reorganize_figures.sh dry        # only write the manifest (figures_by_potential/MANIFEST.tsv) and list
#                                         # what would go to _unsorted/; nothing is copied
#   bash reorganize_figures.sh links      # after figures/ is deleted: rebuild figures/ as symlinks into the new
#                                         # tree, so notebooks and scripts with the old paths keep working
#
# Files are hard-linked (COPY=link, default): no extra disk space, and deleting figures/ later leaves the new
# tree intact. COPY=copy makes real copies (needed if the two folders are on different filesystems).
# Run it on each machine: folders that only exist there (gitignored data) are picked up the same way; a folder
# missing on a machine is skipped. Anything not in the mapping below goes to _unsorted/ with its old path, so
# nothing is lost. MANIFEST.tsv lists every file as  old path <tab> new path.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-copy}
SRC=figures
DST=figures_by_potential
CP="cp -alf"; [ "${COPY:-link}" = copy ] && CP="cp -af"
shopt -s nullglob

if [ "$MODE" = links ]; then
    [ -f "$DST/MANIFEST.tsv" ] || { echo "no $DST/MANIFEST.tsv"; exit 1; }
    [ -e "$SRC" ] && { echo "$SRC/ still exists: delete it first (or move it away)"; exit 1; }
    while IFS=$'\t' read -r old new; do
        [ -e "$DST/$new" ] || continue
        mkdir -p "$SRC/$(dirname "$old")"
        [ -e "$SRC/$old" ] || ln -sr "$DST/$new" "$SRC/$old"
    done < "$DST/MANIFEST.tsv"
    echo "$SRC/ rebuilt as symlinks into $DST/ ($(find "$SRC" -type l | wc -l) links); new outputs written to old paths land in real folders under $SRC/"
    exit 0
fi

mkdir -p "$DST"
MAN="$DST/MANIFEST.tsv"
: > "$MAN"
missing=()

# put OLD NEW: the whole folder figures/OLD as figures_by_potential/NEW
put() {
    local s="$SRC/$1" d="$DST/$2"
    if [ ! -d "$s" ]; then missing+=("$1"); return 0; fi
    find "$s" -type f -not -path '*/__pycache__/*' -not -path '*/.ipynb_checkpoints/*' -printf "$1/%P\t$2/%P\n" >> "$MAN"
    if [ "$MODE" != dry ]; then mkdir -p "$d"; $CP "$s/." "$d/"; fi
}

# putf NEW PATTERN...: the files figures/PATTERN (globs) into figures_by_potential/NEW
putf() {
    local d="$1"; shift
    local pat f rel
    for pat in "$@"; do
        for f in $SRC/$pat; do
            [ -f "$f" ] || continue
            rel=${f#"$SRC/"}
            printf '%s\t%s\n' "$rel" "$d/$(basename "$f")" >> "$MAN"
            if [ "$MODE" != dry ]; then mkdir -p "$DST/$d"; $CP "$f" "$DST/$d/"; fi
        done
    done
}

# ---------------------------------------------------------------------------------------------- 1D flat
F=1d_flat
putf $F/exact                          'exact_committors/brownian1d/*flat*' 'exact_committors/brownian1d/brownian1d_exact.*' \
                                       exact_committors/brownian1d_exact.py exact_committors/br1d_run.log
put  sim_istar0br10intf                $F/brownian_staple/first_fits
put  staple_compare/sim_istar0br10intf_p1000_s0  $F/brownian_staple/staple_compare_p1000_s0
putf $F/brownian_staple/crossing_check crossing_check/run_1d.py
putf $F/brownian_staple/overview_plots 'overview_plots/flat_*'
put  param_lambda/flat1d               $F/brownian_staple/param_lambda
putf $F/brownian_long_md/npq           'longmd_npq/*flat*' 'longmd_npq/npq_longmd.*' longmd_npq/run.log
putf $F/brownian_long_md/plots         'longmd_plots/flat_*' longmd_plots/longmd_plots.py longmd_plots/run.log
put  lgv_check                         $F/langevin_staple/lgv_check

# ---------------------------------------------------------------------------------------------- 1D two bumps
T=1d_two_bumps
putf $T/exact                          'exact_committors/brownian1d/*two_bumps*' 'exact_committors/brownian1d/brownian1d_exact.*' \
                                       exact_committors/brownian1d_exact.py exact_committors/br1d_run.log
put  param_lambda/bumps1d              $T/brownian_staple/param_lambda
putf $T/brownian_long_md/npq           'longmd_npq/*two_bumps*' 'longmd_npq/npq_longmd.*' longmd_npq/run.log
putf $T/brownian_long_md/plots         'longmd_plots/two_bumps_*' longmd_plots/longmd_plots.py longmd_plots/run.log

# ---------------------------------------------------------------------------------------------- 1D cos well
W=1d_cos_well
put  exact_committors/well             $W/exact
putf $W/exact                          exact_committors/well_exact.py exact_committors/well_run.log
put  langevin_seeds                    $W/langevin_md/seeds
put  langevin_seeds_dt5e-4             $W/langevin_md/seeds_dt5e-4
put  langevin_hist_seeds               $W/langevin_md/history_seeds
put  langevin_hist01                   $W/langevin_md/history01
put  langevin_dt5e-4                   $W/langevin_md/dt5e-4_check
put  well_md_fit                       $W/langevin_md/fits
putf $W/langevin_md/overview_plots     'overview_plots/md_well_*'
put  well_check                        $W/langevin_staple/checks
put  nb_well_staple                    $W/langevin_staple/notebook
putf $W/langevin_staple/overview_plots 'overview_plots/wellpath_*'

# ---------------------------------------------------------------------------------------------- 2D Z potential
Z=2d_zpot
put  exact_committors/zpot             $Z/exact
putf $Z/exact                          exact_committors/zpot_exact.py exact_committors/zpot_run.log
put  z_brownian                        $Z/brownian_g10/reference_and_analyses
put  nb_zbrownian_md                   $Z/brownian_g10/md
put  istarz_br_g10                     $Z/brownian_g10/staple/original
put  istarz_br_g10_diag                $Z/brownian_g10/staple/original_diag
put  istarz_br_g10_reversed            $Z/brownian_g10/staple/reversed
put  istarz_test                       $Z/brownian_g10/staple/subset_test
put  nb_brownian_staple                $Z/brownian_g10/staple/notebook
put  staple_compare/sim_istarz_br_g10_p1000_s12  $Z/brownian_g10/staple/staple_compare_p1000_s12
put  selection_test                    $Z/brownian_g10/staple/selection_test
put  param_lambda/z_br                 $Z/brownian_g10/staple/param_lambda/z_br
put  param_lambda/z_emb                $Z/brownian_g10/staple/param_lambda/z_emb
put  nb_zpot_short_traj                $Z/brownian_g10/short_trajectories
putf $Z/brownian_g10/checks/crossing_check  crossing_check/run_z.py 'crossing_check/z_check.*'
putf $Z/brownian_g10/checks/reference_check 'reference_check/z_*' reference_check/check_reference.py
put  poisson_trap                      $Z/brownian_g10/checks/poisson_trap
put  zq_weights                        $Z/brownian_g10/checks/zq_weights
putf $Z/brownian_g10/overview_plots    'overview_plots/zmd*' 'overview_plots/zstaple*'
put  zlgv_fd                           $Z/langevin_g2/fd_velocity
put  zlgv_history                      $Z/langevin_g2/history
put  nb_zpot_short_traj_langevin       $Z/langevin_g2/short_trajectories
putf $Z/langevin_g2/overview_plots     'overview_plots/zlgv*'

# ---------------------------------------------------------------------------------------------- 2D curved channel
C=2d_curved_channel
put  exact_committors/cc               $C/exact
putf $C/exact                          exact_committors/cc_exact.py exact_committors/cc_run.log
put  toy2d_check                       $C/toy2d_check
put  nb_toy2d_staple/sim_istar_cc_br   $C/brownian_g10/staple
put  nb_toy2d_staple/test_cc_br        $C/brownian_g10/staple_test
put  staple_compare/sim_istar_cc_br_p2000_s0  $C/brownian_g10/staple_compare_p2000_s0
for d in $SRC/nb_toy2d_md/CurvedChannelDoubleWell/brownian_*;      do put "${d#"$SRC/"}" $C/brownian_g10/md/$(basename "$d"); done
for d in $SRC/nb_toy2d_md/CurvedChannelDoubleWell/TEST_brownian_*; do put "${d#"$SRC/"}" $C/brownian_g10/md_test/$(basename "$d"); done
putf $C/brownian_g10/checks/crossing_check  crossing_check/run_cc.py
putf $C/brownian_g10/checks/reference_check 'reference_check/cc_*' reference_check/check_reference.py
put  param_lambda/cc_br                $C/brownian_g10/param_lambda/cc_br
put  param_lambda/cc_emb               $C/brownian_g10/param_lambda/cc_emb
put  param_lambda/cc_emb_np            $C/brownian_g10/param_lambda/cc_emb_np
put  nb_toy2d_staple/sim_istar_cc_lgv  $C/langevin_g2/staple
put  nb_toy2d_staple/test_cc_lgv       $C/langevin_g2/staple_test
for d in $SRC/nb_toy2d_md/CurvedChannelDoubleWell/langevin_*;      do put "${d#"$SRC/"}" $C/langevin_g2/md/$(basename "$d"); done
for d in $SRC/nb_toy2d_md/CurvedChannelDoubleWell/TEST_langevin_*; do put "${d#"$SRC/"}" $C/langevin_g2/md_test/$(basename "$d"); done

# ---------------------------------------------------------------------------------------------- shared
S=_shared
putf $S/exact_committors               exact_committors/exact_tools.py
putf $S/overview_plots_scripts         overview_plots/make_original_plots.py overview_plots/regen_all.py overview_plots/regen.log
put  param_lambda/embedding            $S/param_lambda/embedding
put  param_lambda/logs                 $S/param_lambda/logs
put  stopped_zq                        $S/stopped_zq

# ---------------------------------------------------------------------------------------------- the rest
LEFT="$DST/.leftover.txt"
comm -23 <(cd "$SRC" && find . -type f -not -path '*/__pycache__/*' -not -path '*/.ipynb_checkpoints/*' | sed 's#^\./##' | sort) \
         <(cut -f1 "$MAN" | sort -u) > "$LEFT"
while IFS= read -r rel; do
    printf '%s\t%s\n' "$rel" "_unsorted/$rel" >> "$MAN"
    if [ "$MODE" != dry ]; then mkdir -p "$DST/_unsorted/$(dirname "$rel")"; $CP "$SRC/$rel" "$DST/_unsorted/$rel"; fi
done < "$LEFT"
sort -o "$MAN" "$MAN"

echo "manifest: $MAN ($(wc -l < "$MAN") entries)"
[ ${#missing[@]} -gt 0 ] && echo "not on this machine (skipped): ${missing[*]}"
if [ -s "$LEFT" ]; then
    echo "to _unsorted/ ($(wc -l < "$LEFT") files), by folder:"
    sed 's#/.*##' "$LEFT" | sort | uniq -c | sed 's/^/   /'
else
    echo "nothing left for _unsorted/"
fi
rm -f "$LEFT"
[ "$MODE" = dry ] && { echo "dry run: nothing copied"; exit 0; }
find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------------------------- READMEs
cat > "$DST/README.md" <<'EOF'
# Figures and results, by potential

A structured copy of `examples/figures/`, made by `examples/reorganize_figures.sh`. Sorted by **potential**, then by
**dynamics** (overdamped `brownian_g<gamma>` or `langevin_g<gamma>`), then by kind of run (`staple` = staple/iSTAR TIS path
data, `md` = plain MD, `exact` = reference committors, `checks` = method checks).

| Folder | Potential | Runs |
|---|---|---|
| `1d_flat/` | flat 1D potential | staple `sim_istar0br10intf` (overdamped), staple `sim_istar0lgv15intf` (Langevin), long overdamped MD |
| `1d_two_bumps/` | two cosine bumps of 1 kT on [-0.1, 0.3] | staple `sim_istar2cosbr_2108` (param_lambda), long overdamped MD |
| `1d_cos_well/` | toytis `cos_well.CosWellWalls` | Langevin MD `md_istarwell2_100M`, Langevin staple `sim_istarwell2rv` |
| `2d_zpot/` | toytis `sjoelbak_angle` (Z potential, Lx = 0.3, 45° ridge) | overdamped γ = 10 (`sim_istarz_br_g10`, MD `md_zbr_g10_s*`), Langevin γ = 2 (`sim_istarz_2603`) |
| `2d_curved_channel/` | toytis `curved_channel.CurvedChannelDoubleWell` (defaults) | overdamped γ = 10 (`sim_istar_cc_br`), Langevin γ = 2 (`sim_istar_cc_lgv`), MD |
| `_shared/` | several | tools and results that cover more than one potential |
| `_unsorted/` | ? | files the script had no rule for, under their old path |

Each potential folder has its own README with the old folder of every subfolder.

**Old paths.** `examples/figures/` is gone: notebooks, scripts and docs use the paths in this tree. `MANIFEST.tsv`
(per machine, not in git) lists every file as `old path (under figures/)` and `new path (under figures_by_potential/)`,
for finding where an old path went.

**Git.** As in `figures/`, numpy files (`*.np*`) are not tracked except under `*/exact/`; see `.gitignore`. The
workstation and the laptop therefore hold different files in the same folders; run the script on each machine.
EOF

cat > "$DST/1d_flat/README.md" <<'EOF'
# 1D flat potential

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/brownian1d` (flat files), `brownian1d_exact.py`, `br1d_run.log` | exact committors of the flat runs |
| `brownian_staple/first_fits/` | `sim_istar0br10intf` | first fits on the overdamped staple run `sim_istar0br10intf` (γ = 1, kT = 1, dt = 2e-4; A x ≤ -0.1, B x ≥ 0.8) |
| `brownian_staple/staple_compare_p1000_s0/` | `staple_compare/sim_istar0br10intf_p1000_s0` | `staple_data_choices.ipynb`: weights, mirrored pieces, cut, piece length |
| `brownian_staple/crossing_check/` | `crossing_check/run_1d.py` | MSM crossing check |
| `brownian_staple/overview_plots/` | `overview_plots/flat_*` | plots for `docs/results_overview.md` |
| `brownian_staple/param_lambda/` | `param_lambda/flat1d` | parametric λ models |
| `brownian_long_md/npq/`, `plots/` | `longmd_npq`, `longmd_plots` (flat files) | NPq on the long overdamped MD (A λ ≤ -0.1, B λ ≥ 0.3) |
| `langevin_staple/lgv_check/` | `lgv_check` | Langevin staple run `sim_istar0lgv15intf` (γ = 1, kT = 1, m = 1, dt = 2e-4; A -0.1, B 1.3): lag, basis and velocity checks |
EOF

cat > "$DST/1d_two_bumps/README.md" <<'EOF'
# 1D two bumps (two cosine bumps of 1 kT on [-0.1, 0.3]), overdamped

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/brownian1d` (two_bumps files) | exact committor |
| `brownian_staple/param_lambda/` | `param_lambda/bumps1d` | parametric λ models on the staple run `sim_istar2cosbr_2108` |
| `brownian_long_md/npq/`, `plots/` | `longmd_npq`, `longmd_plots` (two_bumps files) | NPq on the long overdamped MD (A λ ≤ -0.1, B λ ≥ 0.3) |

The long-MD scripts (`npq_longmd.py`, `longmd_plots.py`) and logs cover the flat run as well; they are in both folders.
EOF

cat > "$DST/1d_cos_well/README.md" <<'EOF'
# 1D cos well (toytis `potentials/cos_well.py`, `CosWellWalls`), Langevin

A: λ ≤ -0.35, B: λ ≥ 0.35. Long MD `md_istarwell2_100M` (dt = 0.005, positions and velocities); staple run with recorded
velocities `sim_istarwell2rv`.

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/well`, `well_exact.py`, `well_run.log` | exact q(x, v) for the Euler and BBK integrators |
| `langevin_md/seeds/` | `langevin_seeds` | `base_documented_1d_langevin.ipynb`, per fit seed |
| `langevin_md/seeds_dt5e-4/` | `langevin_seeds_dt5e-4` | the same at dt = 5e-4 |
| `langevin_md/history_seeds/` | `langevin_hist_seeds` | fits with a history CV (y(t-d), y(t)) |
| `langevin_md/history01/` | `langevin_hist01` | history 0,1 fit checks (gamma, envelope) |
| `langevin_md/dt5e-4_check/` | `langevin_dt5e-4` | MD committor check at dt = 5e-4 |
| `langevin_md/fits/` | `well_md_fit` | longer fits on 6M MD frames (energy CV, lagged (x, v)) |
| `langevin_md/overview_plots/` | `overview_plots/md_well_*` | plots for `docs/results_overview.md` |
| `langevin_staple/checks/` | `well_check` | fits on the staple paths: recorded vs finite-difference velocity, mirrored pieces, lag; brute-force reference |
| `langevin_staple/notebook/` | `nb_well_staple` | staple notebook fits |
| `langevin_staple/overview_plots/` | `overview_plots/wellpath_*` | plots for `docs/results_overview.md` |
EOF

cat > "$DST/2d_zpot/README.md" <<'EOF'
# 2D Z potential (toytis `sjoelbak_angle`, `RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)`)

y is the order parameter, x the transverse coordinate; A: y ≤ 0.05, B: y ≥ 0.85 (interfaces 0.05, 0.13, ..., 0.85).

## brownian_g10: overdamped, γ = 10, kT = 0.05, dt = 0.002 (staple `sim_istarz_br_g10`, MD `md_zbr_g10_s1..16`)

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `reference_and_analyses/` | `z_brownian` | finite-volume reference committor (`committor_reference_h*.npz`, read by many notebooks), engine shooting, MD and synthetic-data tests, staple splits |
| `md/` | `nb_zbrownian_md` | fits on the long MD |
| `staple/original/` | `istarz_br_g10` | original staple fit (3000 paths/ensemble, forward pieces) |
| `staple/original_diag/` | `istarz_br_g10_diag` | the same with the ridge-normal CV y + x |
| `staple/reversed/` | `istarz_br_g10_reversed` | forward + mirrored pieces, continued fits, even/odd halves, lag 50 |
| `staple/subset_test/` | `istarz_test` | first tests on a subset of the run |
| `staple/notebook/` | `nb_brownian_staple` | `base_documented_2d_brownian_staple.ipynb` (per seed, new fixed-length trajectories) |
| `staple/staple_compare_p1000_s12/` | `staple_compare/sim_istarz_br_g10_p1000_s12` | `staple_data_choices.ipynb` |
| `staple/selection_test/` | `selection_test` | balancing the TIS data over regions (path selection, blind starts) |
| `staple/param_lambda/` | `param_lambda/z_br`, `z_emb` | parametric λ models |
| `short_trajectories/` | `nb_zpot_short_traj` | `zpot_short_trajectories.ipynb`: short trajectories from path data, fixed/geometric/turn stopping |
| `checks/crossing_check/` | `crossing_check/run_z.py`, `z_check.*` | MSM crossing check |
| `checks/reference_check/` | `reference_check/z_*` | reference committor check |
| `checks/poisson_trap/`, `checks/zq_weights/` | `poisson_trap`, `zq_weights` | Z_q trap and weight tests |
| `overview_plots/` | `overview_plots/zmd*`, `zstaple*` | plots for `docs/results_overview.md` |

## langevin_g2: Langevin, γ = 2, kT = 0.05, dt = 0.002 (staple `sim_istarz_2603`, positions only)

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `fd_velocity/` | `zlgv_fd` | finite-difference velocities as CVs, lag 1 and 4 |
| `history/` | `zlgv_history` | history CVs |
| `short_trajectories/` | `nb_zpot_short_traj_langevin` | `zpot_short_trajectories_langevin.ipynb` |
| `overview_plots/` | `overview_plots/zlgv*` | plots for `docs/results_overview.md` |

`exact/` (`exact_committors/zpot`, `zpot_exact.py`): exact committor of the staple dynamics.
EOF

cat > "$DST/2d_curved_channel/README.md" <<'EOF'
# 2D curved channel (toytis `curved_channel.CurvedChannelDoubleWell`)

Defaults: Vb = 0.2, eps = 0, y0 = 0.5, w = 0.4, xA = -0.15, xB = 0.15, b = 0.2, k = 15, a = 0. y is the order parameter;
interfaces 0.15, 0.22, ..., 0.85 (A: y ≤ 0.15, B: y ≥ 0.85); kT = 0.05, dt = 0.005.

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/cc`, `cc_exact.py`, `cc_run.log` | exact committor of the overdamped dynamics |
| `toy2d_check/` | `toy2d_check` | potential and committor check |
| `brownian_g10/staple/` | `nb_toy2d_staple/sim_istar_cc_br` | `toy2d_staple_analysis.ipynb` on `sim_istar_cc_br` (γ = 10) |
| `brownian_g10/staple_test/` | `nb_toy2d_staple/test_cc_br` | the same on the test run |
| `brownian_g10/staple_compare_p2000_s0/` | `staple_compare/sim_istar_cc_br_p2000_s0` | `staple_data_choices.ipynb` |
| `brownian_g10/md/`, `md_test/` | `nb_toy2d_md/CurvedChannelDoubleWell/(TEST_)brownian_*` | `toy2d_md.ipynb`: MD and fits |
| `brownian_g10/checks/` | `crossing_check/run_cc.py`, `reference_check/cc_*` | MSM crossing check, reference check |
| `brownian_g10/param_lambda/` | `param_lambda/cc_br`, `cc_emb`, `cc_emb_np` | parametric λ models |
| `langevin_g2/staple/`, `staple_test/` | `nb_toy2d_staple/sim_istar_cc_lgv`, `test_cc_lgv` | `toy2d_staple_analysis-lgv.ipynb` (γ = 2) |
| `langevin_g2/md/`, `md_test/` | `nb_toy2d_md/CurvedChannelDoubleWell/(TEST_)langevin_*` | `toy2d_md-lgv.ipynb` |
EOF

cat > "$DST/_shared/README.md" <<'EOF'
# Shared: tools and results across potentials

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact_committors/` | `exact_committors/exact_tools.py` | helpers of the `*_exact.py` scripts (those are in each potential's `exact/`) |
| `overview_plots_scripts/` | `overview_plots/make_original_plots.py`, `regen_all.py` | make the `overview_plots` of all potentials |
| `param_lambda/embedding/`, `logs/` | `param_lambda/embedding`, `logs` | parametric λ: embedding comparison and logs of all systems |
| `stopped_zq/` | `stopped_zq` | `rerun_stopped_zq.py`: stopped Z_q redraws of earlier fits (if made on this machine) |
EOF

if [ -d "$DST/_unsorted" ]; then
    cat > "$DST/_unsorted/README.md" <<'EOF'
# Unsorted

Files of `figures/` that `reorganize_figures.sh` has no rule for, under their old path (folders that only exist on this
machine, or new ones). Add a rule to the script and run it again, or move them by hand.
EOF
fi

echo "done: $DST/ (READMEs in every potential folder); add to .gitignore if not there yet:"
cat <<'EOF'
    examples/figures_by_potential/MANIFEST.tsv
    examples/figures_by_potential/**/*.np*
    examples/figures_by_potential/**/*.pyc
    examples/figures_by_potential/2d_zpot/brownian_g10/staple/subset_test/sim_istarz_subset/
    !examples/figures_by_potential/*/exact/**/*np*
EOF
