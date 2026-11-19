# Order-parameter learning loop: Step 0 plan (2026-11-16)

Findings from reading the notes, the `param_lambda` code, toytis, infretis (the APPTIS checkout, which has StapleTIS)
and inftools, and the plan for `examples/param_lambda/loop/`. Nothing has been implemented yet.

## 1. What each backend can do

### toytis (`/mnt/.../PyRETIS3/toytis`, in-process, pure Python)

| | |
|---|---|
| ensembles | `simtype` ∈ {retis, repptis, i*}: RETIS [0⁻], [0⁺], [i⁺] with full A→A/B paths (`simulation.py:345-389`); REPPTIS [i±]; staple [0*], [i*]. So toytis **can** run RETIS full-path ensembles. |
| order parameter | `toytis_adapter.install` rebinds `ensemble.OrderX/OrderParameter` to `LearnedOrderX`, which returns `[λ, cv_0, …]`. λ is evaluated **every MD step** (`moves.py:731`, `pgmoves.py:724`), never recomputed for shooting points or swapped segments (stored `path.orders` are reused). |
| per-frame record | `order.txt`: time, λ, CVs. toytis appends velocities only when `len(orders) == dim`, which `[λ, cvs]` breaks; for velocity-aware runs (M5) the adapter will append the velocities as further order columns itself and the loader will read them by the stored column spec. |
| N cycles | `Simulation(settings).run()` with `max_cycles`; or `sim.do_shooting_moves()` / `sim.do_swap_moves()` per cycle from our driver (they increment `sim.cycle`). |
| restart | `restart.pkl` (settings ignored except `max_cycles`; global `np.random` state restored). |
| initial paths | **no API**: `create_initial_path` runs `kick_retis` / `kick_star` (MD from a hand-placed kick point) or a synthetic `linspace` path for RETIS [i⁺] (`ensemble.py:871-889`, flagged `dummy`). The backend will patch `Ensemble.create_initial_path` (module attribute, as the adapter does) to install our `Path` objects (`staridx` set after construction; `meta = [ptype, dir, status, sh_op]`) and write the usual `ld` record. |
| MD steps | `ens.engine.mdsteps` per ensemble (pickled). Per cycle = difference of the sum. |
| lineage on disk | none: `pathensemble.txt` has no parent id, shooting index or coordinates. Only the `ld` line marks the kick/dummy path. |

### infretis (checkout `/mnt/.../APPTIS/infretis`, must be first on `sys.path`; the installed 2025.4 copy has no StapleTIS)

| | |
|---|---|
| order parameter | `[orderparameter] class/module` (absolute module path), `calculate(system)` with `system.pos`, `system.vel` (sign-flipped on backward segments), returns a list; order.txt = frame index, λ, every further value (`formatter.py:297-356`). Evaluated **once per saved subcycle** (`turtlemdengine.py:247-268`) and once at the shooting point. Workers are **spawned**; every worker builds `min(n_ens, workers)` engines, each with its own order-parameter object (`asyncrunner.py:210-218`, `factory.py:95-102`). The `infretis_orderp.py` docstring's "built once, inherited by fork" is wrong for this checkout. The driver must guard `if __name__ == "__main__"`. |
| modes | `[simulation] mode = "staple"` + `tis_set.staple = true` (windows [i] = (λ_{i−1}, λ_i, λ_{i+1}), [0⁺] = (λ0, λ0, λ1)); `mode = "pp"` for REPPTIS; else RETIS. |
| N cycles | `[simulation] steps`; in-process `infretis.bin.internalrun(toml)` (what inftools' `run_infretis_ext` calls) or `infretisrun -i`. |
| restart | `restart.toml` (`[current]`: active, traj_num, cstep, frac, rng_state, locked, tsubcycles). A toml with new interfaces and new `load/0..n-1` must have **no** `[current]` table and no `restart.toml` next to it (`setup.py:115-151`); `shooting_moves` must have ≥ n_interfaces entries, `workers ≤ n_interfaces − 1`. |
| load paths | `load/<n>/{order.txt, traj.txt, accepted/*}`; order parameters are **never recomputed** on load (`path.py:420-426`). `inft recalculate_traj` writes `order_rec.txt` through MDAnalysis and does not touch `order.txt`. Our relabel writes order.txt from the recorded CV columns (extension of `infretis_orderp.relabel`). |
| MD steps | `current.tsubcycles` × `[engine] subcycles` (saved frames, including the first frame of each propagation). Staple `print_shooted` omits the count; RETIS logs it. |
| lineage | `traj.txt` (file, frame index, vel sign) and `accepted/` files hard-linked from the parent (same inode across `load/*/accepted/`); `generated` tuple (move, shooting-point order, index); parent path number only in `sim.log` and, for staples, the `ensemble`/`mc_move` columns of `infretis_data.txt`. Shooting moves re-emit the shooting point as frame 0 of the new files, so inheritance of *frames* happens only through swaps and the `ld` paths. `delete_old = true` destroys the evidence: loop runs set it false. |

### inftools `infinit`

`inftools/tistools/initial_paths.py:170-265`, helpers in `inftools/misc/infinit_helper.py`. Monolithic loop, but every step
is an importable function: `generate_zero_paths` (one [0⁻] + one [0⁺] path from `initial_conf`, plain dynamics at the
run temperature, no steering), `run_infretis_ext(steps)` (in-process), `update_toml_interfaces` (binless WHAM
P_cross from the `maxop` column of `infretis_data_*.txt` / `combo_*`, interfaces at equal local probability pL,
λ_B never moved), `update_toml`, `initial_path_from_iretis` (relabels paths to the highest ensemble whose interface is
below the path's max of **stored** `order.txt`; never cuts), `update_actives_toml` (symlinks `load/<new> → load/<old>`,
resets `[current]`), `rename_file(restart.toml)`. It stops only when `steps_per_iter` is exhausted: **there is no
"reached B" criterion**; the front just approaches the fixed λ_B.

**λ switch insertion point:** between `run_infretis_ext` and `update_toml_interfaces`. The controller skips
`update_toml_interfaces` entirely (its P_cross is in the old λ and it silently merges `combo_*` files), rewrites
`order.txt` of the paths it keeps with the new λ, places interfaces itself (`interfaces.py`), writes
`[orderparameter]` and the interfaces into the toml, then calls the stock tail (`update_toml`, `initial_path_from_iretis`,
`update_actives_toml`, `rename_file`).

## 2. Valid initial paths from stored paths (`seed_paths`)

Common: the new λ is evaluated on the recorded CV columns of every stored path; states are exact (λ = s outside
(λ_A, λ_B)), so start/end-in-state conditions survive any λ switch.

| ensemble | rule |
|---|---|
| RETIS [0⁻] | stored [0⁻] paths stay valid (λ = s inside A, λ_A unchanged). |
| RETIS [i⁺] | whole stored A-started paths, assigned to the highest free ensemble with λ_i < max λ (as `initial_path_from_iretis`); missing lower ensembles get a copy of a higher one. No cutting. |
| REPPTIS [i±] | cut from any stored path: a segment that starts with a frame outside the window (L or R), stays inside, crosses M, and ends with the first frame outside again. |
| staple [i*] | cut from full A→A/B paths or from stored staples: a window segment crossing M as the core, then the stored path is followed outward on both sides until the sampler's turn rule (`turn_detected`, replayed on λ with the new interface set) or A/B. If the stored path ends before the turn completes, the candidate is dropped. [0*] rejects RMR; both turns and the core are validated with the sampler's own code (toytis `pg_check_path`/`validate_staple`; infretis `StaplePath.check_turns`/`get_sh_region`, called in-process as `infretis_data.staple_core` already does). |
| fallback | if no stored path yields a valid path for an ensemble: toytis runs its kick with a **bounded** retry count (wrapper raises `SeedError(ensemble)`); infretis copies the nearest valid path (what infinit does) and reports it. |

infretis seeds are written as `load/<n>/` with `traj.txt` rows pointing (hard links) at the parent's `accepted/` files
and the frame index range of the cut, plus the relabelled `order.txt`. toytis seeds are `Path` objects with
`staridx` and `meta` set, installed through the patched `create_initial_path`.

## 3. Shooting regions, single-step crossings, turns, unbounded loops

| sampler / ensemble | shooting region | no frame in region | one step over two interfaces | turn in one step |
|---|---|---|---|---|
| toytis RETIS/REPPTIS (`moves.py:61`) | any interior frame | cannot happen (paths ≥ 3 frames) | not interpolated; validity by min/max range | n/a |
| toytis staple (`pgmoves.py:62-81`) | frames with L ≤ λ ≤ R inside `staridx` | prints, then `np.random.choice([])` → **ValueError, run dies** | `pg_check_path` returns `***` if no frame strictly inside (L, R); `validate_staple` counts interfaces between extremes, so a jump over two is still a turn | `turn_detected` fires on that step |
| infretis RETIS (`path.py:152`) | interior frames | n/a | stop tests only against L/R; crossing by range | n/a |
| infretis staple (`staple_path.py:405-438`) | `get_sh_region`: borders of the window segment that crosses M (first and last frame inside; `(0,0)` if invalid) | `ValueError` → fallback `get_pp_path` returns None → `AttributeError` → worker exception kills the run (no retry) | same as RETIS; `_check_start_turn` masks all interfaces between consecutive frames | handled (`orders` length 3, "back across M" test). If the extension start point lies strictly inside (L, R), `turn_detected` raises. |

Loops that can retry forever: only toytis `kick_retis` (`moves.py:412`) and `kick_star` (`pgmoves.py:530`), both
`while True` with `continue` on every failure. infretis has no unbounded retry in `tis.py`; its hang mode is the
busy-wait in `as_completed` when a worker dies (`asyncrunner.py:243-257`). Other sharp edges: toytis `pg_swap`
mutates `staridx`/`meta` of shared Path objects; infretis `staple_extender` overwrites `FTX/BTX` with `ACC`
(`tis.py:2678-2679`), so over-length staples are accepted (left as is, reported in the interface-health table).

## 4. Conflicts between the prompt, the notes and the code

1. **Mirrored pieces for RETIS.** `tis_data.load_order_file` and `infretis_data.load_infretis_data` add the reversed
   piece for every ensemble, including RETIS [0⁻]/[i⁺] (where it always ends in A). Guard to add: reversed pieces only
   for ensembles whose start and end conditions are the same set (staple, REPPTIS); RETIS → refuse with a message.
2. **"infinit until B is reached"** has no counterpart in inftools. The controller defines it: pass 0 ends when at
   least `n_B` (default 20) accepted paths end in B (λ ≥ λ_B), otherwise it keeps advancing with infinit's own
   placement in the seed λ.
3. **Velocities in toytis `order.txt`** are written only when the order parameter returns exactly `dim` values.
   With `[λ, cvs]` the adapter must append them itself (M5 only; a config flag).
4. **`infretis_orderp.py` assumes forked workers**; the checkout spawns. The model is loaded once per engine per
   worker (a 64×64 MLP: harmless), module paths must be absolute, the driver needs a `__main__` guard.
5. **infretis staple shooting weight (notes §15):** no toml option adds the `min(1, w_new/w_old)` factor; making the
   sampler consistent needs a core edit. Plan: M4 first measures the bias on a flat 1D turtlemd staple run with the
   exact committor (the test §15 proposes); if it is significant, stop and ask before touching core.
6. **toytis adapter kick placement is 2D-only** (`other = 1 - s`); the 3D gate potential needs
   `notebooks/nd_patches.py` composed with the adapter (documented to compose) and a kick target in the extra
   coordinates. Extension of the adapter, not of toytis.
7. **The learned-λ test runs on Z used 15–16 interfaces placed without the spacing gate** (natural coordinate +
   ratio cap). They are still usable as pass-k data for placement and as the "before" row of the interface-health
   table; the production runs of M1 get re-placed interfaces.
8. **Lineage in toytis is not recorded**, so constraint 7 is implemented in the adapter: a registry of the
   non-physical frames (kick points, `linspace` dummy frames) by exact position value; a path is `tainted` while any
   of its frames matches (shooting points are reused by reference, so the hand-placed point survives in children with
   the same value). infretis: `ld` paths, and inode/basename inheritance through `accepted/`.
9. **The Brownian staggered-channel staple run** of the figures folder lives only on the workstation; M3 makes its
   own.

## 5. Design decisions (to confirm)

- **Package layout** `examples/param_lambda/loop/`: `backends/{base,toytis,infretis}.py`, `pool.py`, `teacher.py`,
  `student.py`, `interfaces.py`, `diagnostics.py`, `seed.py` (path cutting, shared by both backends), `manifest.py`,
  `controller.py`, `report.py`; `run.py loop <config>` dispatches to `controller`. Configs `configs/loop_*.toml`.
- **Pool format**: per pass `pool/pass_<k>/frames.npz` in the `data.read_run` layout (cv, piece, path, t, w, A, B,
  ids, cycles, ensembles, t*, mirrored) plus `lineage` per path, `lambda_version`, `interfaces`, column spec; a
  manifest JSON with config, seeds, md5 of the λ export, interfaces and data. Pieces are cut with the λ and
  interfaces of their own pass and never re-cut.
- **Teacher**: `optimalrcs-cupy` `CommittorNE` on the selected pool (capped frames), warm-started from the stored q of
  old frames and the current model's m(x) on new frames (the notebooks' continued fits show `r_traj` continuation
  works); fixed iteration budget per pass; provisional B = {s ≥ s_front} when fewer than `n_B` paths reach B; optional
  nested fit below an isosurface. CV screening (within/between-path variance, shuffle test) before the fit.
- **Student**: `models.MLP` 64×64, β = 10, with a new `init_params` option for the warm start (an addition, not a
  rewrite). Export through `steps.build_export` plus the new fields. `lambda_runtime` gets a format version:
  `ramp_kind = "smin"` (smooth minimum at A, B and the provisional front, replacing the multiplicative ramps) and
  the trust-region blend (`t(ood)`, `h(s)` table); old exports keep loading unchanged. Constraint 2 stays exact by the
  same clamp as today, with a test.
- **Interfaces**: `place_interfaces(model, pool_pass, cfg)` = natural-coordinate plan (`natural.Increments`,
  `staple_run.cap_ratio`) → length budget (`natural.predict_lengths`, may only add) → greedy thinning under the
  spacing checks (skip both directions, gap, distance from A/B, shooting-region occupancy per sampler, single-step
  turns, kick/seed points) → unverified flags → conflict report. All checks are recomputed on each pass's own paths
  (`check_interfaces`) and the runtime counters (empty shooting regions, skipped interfaces) come from the adapters.
- **Pass size**: toytis 2000 cycles per pass (Z: ≈ 13M MD steps, ≈ 5 min single-core); infretis the equivalent in
  `steps`. Production runs 10⁵ cycles as the existing training runs, launched with `systemd-run --user`.
- **Rates**: toytis staple runs through the tistools iSTAR MSM (`crossing_check.istar_msm`: P_cross) and the [0⁻]/[0*]
  flux; exact rates from the exact committors (`toy2d.exact_committor_brownian` TPT rate; the long-MD transition
  counts as a cross-check). Errors from MC-cycle blocks.
- **GPU**: the per-pass NPNEq teacher on 2D systems (≤ 5M frames, a few thousand warm-started iterations) fits the
  laptop GPU. Workstation jobs, with commands handed over when reached: M3 gate potential (3 CVs, long NPNEq) and M5
  (x, v) fits at lag Δ; the 3D exact committor if wanted.

## 6. Milestone M1 in detail (first implementation step after confirmation)

1. `loop/pool.py` loader for toytis runs (reuses `data.read_run`; adds ensemble windows and per-step λ for the checks).
2. `loop/interfaces.py`: `place_interfaces`, `check_interfaces`, the health table.
3. `loop/backends/toytis.py`: `install_order`, `run_pass` (wraps `staple_run`), `md_steps`, runtime counters,
   bounded kick retries.
4. Z and cc: exported `m3-fast-nat` / `m3-nat` λ → placement on the existing 2000-cycle learned-λ runs and the
   training-run frames → production staple runs (λ frozen), seed-λ control at equal MD steps → crossing probability,
   rate, MSM level check, health table. One deliberately failing set (two interfaces one step apart near A on cc,
   q ≤ 3·10⁻³ levels) → acceptance, path statistics, rate.
5. Report section in `docs/op_loop_report.md` (auto-generated).

## 7. Revisions after review (2026-11-16)

- **Order of work: infretis first.** M4 (infretis backend, infinit as pass 0, λ switch between iterations, loaders)
  moves to the front, on turtlemd toy systems. toytis milestones (M1–M3) are postponed; the adapter, export and
  placement tools stay usable by hand so the loop can be applied manually. Items 4.6 (3D kick placement) and 4.8
  (toytis lineage) are postponed with toytis. 4.9 (Brownian staggered channels) runs on the workstation when M3 comes.
- **Manual workflow (deliverable of the first step):** `python infretis_orderp.py relabel`, `interfaces.py` placement
  and checks, `seed.py` path cutting and a `loop/controller.py` whose passes can also be started one at a time from
  the command line, each writing its manifest.
- **CVs.** toytis: positions, and velocities when a config asks (single-particle nD systems need nothing else).
  infretis: the configurable CV pool `[λ, s, cv_1, …]` of the order-parameter module.
- **Mirrored pieces:** never for RETIS ensembles (guard in both loaders).
- **Pass size:** 1000 cycles per pass for the first tests (config value).
- **Rates for infretis runs:** the inftools staple analysis (`inftools.istar`: weight matrices, MSM, P_cross, rates)
  and `inft wham` for RETIS; MC-cycle blocks for the errors.
- **Lineage** = for every stored path, which earlier path and which move produced each of its frames, so a path that
  still carries a frame of an initial load path (kick point, synthetic path, steered or high-temperature frame) can be
  excluded from training (constraint 7). infretis records it (`traj.txt`, hard-linked `accepted/` files, the
  generating move); toytis records nothing, so it was going to be reconstructed in the adapter (postponed).
- **Refill after thinning (2026-11-16).** On Z the spacing gate removed every planned level on the ridge between
  the basin and the plateau, leaving one ensemble over q 0.007 -> 0.645. The exact committor shows the ridge is
  crossed in about 10 saved frames, so no lambda can hold interfaces there; the placement margin had also removed
  one level the gate itself accepts. `interfaces.place` now ends with `interfaces.refill`: for interior neighbour
  pairs over `max_ratio`, insert the best-splitting candidate that passes the placement limits (with the margin;
  against the bare gate, refill undid the thinning and the set failed verification three times), or leave the
  pair if none does; end pairs use the effective state levels of `interfaces.state_levels` (mean h at the first
  frames past the state). Details and numbers in `op_loop_report.md`; test `test_refill_splits_wide_pairs_only_where_the_gate_allows`.
- **Second infretis core change (authorized 2026-11-17): a staple path with no shooting point is a rejected move.**
  A 50k-cycle production on Z died at cycle 23 069: `staple_sh` picked a path in [1*] whose shooting region was empty
  (`{0: (1, 0)}`, length 15); `get_shooting_point` raised, the fallback `get_pp_path` returned `None`, and
  `prepare_shooting_point(None, ...)` raised `AttributeError` in the worker, which stopped the scheduler (a plain
  restart reproduced it: random state and path are restored). `core/tis.py: staple_sh` now catches the failure of the
  fallback and returns `False, trial_path, "NSG"` (the status wire fencing already uses for "no frames to shoot from"),
  with a one-frame trial path as after a failed kick (`run_md` reads its order extrema). Staying in the old path is a
  valid move, so the sampled distribution is untouched. Tests: `test/staple/core/test_no_shooting_point.py` (fail on the
  old code, pass on the new); staple (244) and tis/core/repex (22) suites unchanged. The alternative of zero weights for
  such (path, ensemble) pairs in `calc_cv_vector` was measured on the data and changes P_cross and the rate by -0.7% and
  -0.6% (dropped weight 0.006-0.05% per ensemble); not used, since it changes the ensemble measure and the swap
  bookkeeping for a path invalid in its own ensemble is untested.
- **The gate cannot guarantee zero empty-region paths in a long run.** Accepted paths with an empty shooting region occur
  at about 1 in 1600 in [1*] (7 (path, ensemble) pairs among 13 691 accepted paths of the refilled Z run); a 2000-cycle
  verification sees about one and passed with none by chance. The gate's check stays (a high rate is a sign of badly
  spaced interfaces, e.g. a window 0.0115 wide next to A), but production correctness no longer depends on it being zero.

- **M3 on infretis + ASE (2026-11-17).** The robustness systems run on the infretis backend (toytis stays postponed);
  configs `loop_zu`, `loop_zw`, `loop_stag`.
  - *Native embedding* `zu_br`: Z plus a simulated w in an uncoupled double well (dV 3 kT, w0 0.1: the w dynamics of the
    gate system, so only the coupling differs; `ase_calcs.ToytisPotential(base_dim, w_well)`). Committor and rate of Z.
  - *Gate* `zw_br`: toytis `zpot_w.ZChannelsW` (dV_w 3 kT). Exact committor and rate from a 3D grid solve
    (`loop/exact_zw.py`, the finite-volume TPT method of `toy2d.exact_committor_brownian` with a third axis, Jacobi-CG):
    k_AB = 2.115e-5 per time unit at h 0.004 (2.123e-5 at h 0.008 and 0.012); it matches the six shooting points of
    `3d_zpot_w/README.md` and the GPU MD scan (nAB / walker steps = 2.10e-5 +- 2.4%). A minute on one core, so not a
    workstation job after all.
  - *Two channels* `stag_br`: the staggered-channel landscape, overdamped (gamma 1, kT 0.25, dt 0.005), state CV x.
    Exact: `loop/exact_stag.py` (k_AB 1.148e-4 per time unit, half the flux per channel, q at x = 0: 0.997 in the
    upper channel, 0.003 in the lower). One channel in the initial data: `[pass0] exclude_box` drops every pass-0 path
    that enters the upper channel from the pool (training and seed candidates); the later passes keep what they sample.
  - *CV screening* (`loop/screen.py`, every pass when the system has more than two CVs): the student is trained on all
    CVs and once without each candidate; a CV is kept when leaving it out raises the held-out Dirichlet energy
    (path-weighted mean squared increment of q along the validation pieces, the committor's variational objective; no
    labels) by more than `screen.tol` (10%). The first version used the validation SD(Z_q) and failed on loop_zw pass 1:
    leaving out x or w changed it by -1%, so it dropped both, while the energy rose by 49% and 64% and the RMS error
    against the exact committor from 0.054 to 0.078. The kept set is the input of the student only; the teacher
    keeps every CV (`screen.teacher = "all"`), since a CV the teacher never sees cannot shape the labels and could not
    come back in a later screening. Labels alone cannot separate a coupled from a path-labelling slow CV
    (`param_lambda_embedding.md`, finding 4), hence the dynamical test.
  - *GPU*: the fits of a pass run under `loop/gpu.py` (a file lock shared by the loops, then a wait while any other
    process holds a GPU context).
- **M3 decisions (2026-11-17, evening).**
  - *Fixed CV sets*: `[screen] enabled = false` for loop_zu and loop_zw (teacher and student on y, x, w). The 5-fold
    held-out Dirichlet energy kept x and w on the gate system (+36%, +41%, 5/5 folds) but also kept the uncoupled w
    of the embedding (+24%, 4/5 folds): with the path-ensemble weights it is not the equilibrium Dirichlet form, so the
    committor need not minimize it, and w is not path-constant there (share 0.31). Automatic selection (E7) needs a
    different test later. loop_zw pass 1 ran with lambda(y, x) from the earlier screening (valid data, kept).
  - *IPT change reverted*: a guard in `staple_sh` / `staple_swap` against the loop_zu pass-1 crash
    (`treat_output`: "Ensemble 9 has invalid pptype RML") was built on a synthetic B -> B path that the extender cannot
    produce (it stops at the first turn); a replay of the crash state for 2771 cycles did not reproduce the crash, so
    the real path is unknown. infretis is back at its HEAD (with the NSG change). If the crash recurs, the trial path
    has to be captured first (replay wrapper in the session scratchpad: dump `picked` on a `treat_output` exception).
  - *loop_stag seeding*: the lower-channel pass-0 paths never get past x ~ -0.74, so the ensembles placed above the
    data front could not be seeded. `loop/artificial_paths.py` makes full A -> B paths through the lower channel
    (overdamped dynamics of the same potential at 2 x kT, 4 substeps per frame, accepted only if every frame in the
    channel section is in the lower channel and no frame step exceeds 0.3); all tainted (training never sees them or
    their descendants), appended after every stored path in `seed.build_seeds`, MD steps counted (3.0M walker steps).
    In the future infinit should provide such paths. The pass-0 exclusion box is now the upper channel past its barrier
    top (the first box, any upper-channel entry, left 22.6k frames and a provisional B still in the strip).
- **loop_stag restarted without the exclusion box (2026-11-17, evening).** The pass-0 paths never entered the lower
  channel (1880 paths: 139551 frames past x = -0.9 at y > 0, none at y < 0), although plain dynamics from A enters
  both about equally (941 / 1059 of 2000 walkers): A cuts the strip next to the channel mouths, RETIS paths end in A
  and are shot from old ones, so pass 0 stays on the side of infinit's initial path (upper). The box had removed the
  only transitions pass 0 had (hence the provisional front and the artificial lower-channel seeds). Now the initial
  data are the upper channel by the sampler's own dynamics; E4 asks whether the lower channel ever enters and how far
  the rate is off without it (exact: 2x the one-channel rate). The mixed attempt is in loop_stag/attempt_artificial_lower/.
  Fixes found on the way stay: `passes.learn` passes the teacher's provisional B to the export (it was always None),
  and the blend's seed table ends at the states (it held its last value beyond the data).
- **E7 screening and seed controls (2026-11-18).**
  - *Split-half test failed* (`loop/experiments/w_split.py`): the teacher on two disjoint halves of the pool, the
    well difference of its bin-mean q compared between halves. Correlation gate 0.59 / 0.37 (MC-block halves /
    even-odd passes), embedding 0.00 / 0.68. A single (y, x, w) teacher fit already gives the uncoupled w a well
    difference of 0.044 (gate 0.070; exact 0.0075 / 0.106). The embedding pool has 50 paths that reach B over 12
    passes, clustered by lineage, so every split of the data keeps the same few w-labelled outcomes on both sides.
    Conclusion: no data-only test on these pools can separate the two; the per-piece conditioning is impossible too
    (28% / 63% of the frames lie in pieces that spend more than 20% in the other well).
  - *Shooting test works* (`loop/experiments/w_shoot.py`): 400 transition-region points (teacher q 0.2-0.8), short
    overdamped shots with w set to each well, paired statistic S = mean d1 . d2 of the well difference of the mean
    (y, x) displacement (two independent halves of the shots; E S = 0 exactly for an uncoupled w). Gate z 14-15,
    embedding z <= 1.0 (4 seeds, 5.1M walker steps, seconds); at a quarter of the budget 4.9-5.2 against <= 1.5.
    Not yet wired into the loop (it needs a backend call for short shots from given configurations, and setting w
    is a toy-system shortcut).
  - *loop_zu_xy*: the embedding loop with teacher and student on (y, x) (`[screen] cvs`, new option in
    `passes.run_passes`), from loop_zu's pass 0 with the labels removed.
  - *Seed-lambda controls* (20 000 cycles, same pool): gate 1.61x (rel. error 0.70), embedding 5.75x (0.35), two
    channels **15.6x (0.44)**. With lambda = x on the staggered channels, shooting cannot move paths between channels;
    ensembles 3-6 hold mostly lower-channel paths (before their barrier), ensembles 7-11 only upper-channel ones (past
    theirs), so the crossing probability chains the two and is about 30x too high. `seed_control.py` now takes the
    interfaces from `[seed_control] interfaces` or the system's `seed_interfaces`, resumes an interrupted run and
    uses `rates.reference_rate`.
- **Frame-budget bug behind the embedding failure (2026-11-18).** `loop_zu_xy` (no w) failed like loop_zu (0.17x,
  7 interfaces, teacher RMS 0.16-0.24). Cause: `pool.select(max_frames)` and `teacher.select_frames` thinned whole
  paths uniformly, and `teacher.front` counted B paths in the thinned view: pass 1 kept 14 of 43, fell below
  n_b = 20 and fitted to a provisional B at y >= 0.372, from which neither loop recovered. Fix: two strata (paths
  that reach B, at most half the budget; the rest fills it), every kept piece weighted by its inverse inclusion
  probability (`View.incl`, also in the piece weights), and `front` sums these factors (an estimate of the pool's
  count; pass 1 now 42). Test `test_frame_budget_keeps_b_paths_and_reweights_the_rest`. Reruns from loop_zu's pass 0:
  `loop_zu_b` (y, x, w) and `loop_zu_xy_b` (y, x).
  - *loop_zu_b* (fixed budget, w kept): production **1.85x** (rel. error 0.51), 10 interfaces, 13.6M MD to
    initialize and 17.6M in production, teacher RMS vs exact 0.056 at pass 7, student logit RMSE 0.33, 325 B paths in
    7 passes (loop_zu: 127). The embedding works once the budget is fixed; the uncoupled w is not what broke loop_zu.
    One realization each.
  - *loop_zu_xy_b* crashed at pass 1, cycle 340: infretis `repex_staple.treat_output` raised `Ensemble 8 has invalid
    pptype RML`. Interfaces [..., 0.818, 0.8366, B = 0.85]; refill added 0.8366, and it passed every placement check
    including dist_b. Path 233, accepted by `staple_sh` in the last ensemble, starts in B (0.853), drops to 0.768
    (below 0.818) and ends at 0.8373 just above 0.8366: an RML path, while `treat_output` allows only RMR in the
    last ensemble. Correction: 233 was not the crashing path (its turns are at 9 and 7, so it never reached the
    check). The second crash was path 320 (41 frames: B -> 0.812 -> B). Both of its turns are at B, but its label is
    RML, because `staple_sh` labels the shot segment before the extension (the forward part stopped at lambda_7 and
    the extender carried it back into B). Figure: `loop_zu_xy_b/pass_1/rml_paths.png`. The resumed run crashed
    again. **infretis edit (authorized 2026-11-18):** `repex_staple.treat_output` now accepts every path type in the
    last ensemble. When both turns are at B (index n - 1), one end (coin flip, as before for RMR) gets the turn of the
    first interface above the path's minimum: n - 2 for an RMR path, which stays above L, and n - 3 for a path that
    went below L, such as these "RML" ones (Elias: they turn one interface further back). Both turns at an interior
    interface: no offsets. Only the ptype string in infretis_data.txt changes. The weights use `get_pptype` per
    ensemble, and inftools (`istar.reading.parse_ptype_direction`) reads only the order of the two numbers, which
    the coin flip randomizes either way. Between the edits, passes 5-7 and the production of loop_zu_xy_b ran with a
    version that wrote n - 3 for every RMR path (e.g. 186 rows `10RMR12`/`12RMR10` in production). That leaves the
    rate unchanged, and none of loop_zu_xy_b's runs met a both-at-B non-RMR path again. Crash log kept in
    `loop_zu_xy_b/pass_1/infretis_crash_RML.out`; figure `loop_zu_xy_b/pass_1/rml_paths.png`.
  - *loop_zu_xy_b* (fixed budget, no w): production **1.08x** (rel. error 0.36), 13 interfaces, 12.9M MD to
    initialize and 12.7M in production, teacher RMS 0.086 at pass 7, 237 B paths in 7 passes. Equal to loop_zu_b
    within errors; steadier interface sets without w.
- **M5 started on the 1D cos well (2026-11-18).** System `well_lgv` (ase_systems): toytis CosWellWalls, kT 1,
  gamma 1, m 1, dt 0.005, A x <= -0.35, B x >= 0.35, ASE `Langevin` (Vanden-Eijnden-Ciccotti), CVs (x, v); config
  `loop_well_lgv`. Pieces:
  - *Exact committor of the ASE chain* (`loop/exact_well_lgv.py`): `exact_tools.vec_step` equals ase's step to
    round-off (test `test_vec_step_matches_ase`); the new position depends on one rotated normal, as the 1D
    underdamped grid solver needs. 701 x 501 grid (laptop GPU; the 2801 x 1001 level needs ~6 GB): 200k-shot check
    chi^2 13.1 / 10, grid change 351 -> 701 RMS 2.8e-5. ASE vs toytis BBK committor: RMS 8e-6 (max 2.6e-3 at A/B).
  - *Reference rate* (`loop/md_well_lgv.py`, GPU walkers of the same step): k_AB = 0.33703 +- 0.00022 per time unit
    (k = n(A->B) / time last in A; 3.45M transitions, k_BA 0.33719), one transition per ~593 steps.
  - *Velocity CVs in infretis*: `infretis_orderp` flags the order parameter velocity dependent when a CV source is
    (infretis `Velocity`), so `Path.reverse` recomputes reversed paths; `_ensure_phase_point` reads v from the
    trajectory frame with the sign of vel_rev. Pass 0 uses `sh` (wire fencing reverses paths). No mirrored pieces
    when the CVs hold velocities (`infretis_backend.has_velocity_cvs`). Smoke run: recorded v matches the
    finite-difference velocity in every stored path (median corr 0.999, no sign flips), including backward segments.
  - *Teacher lag* (`[teacher] lag`, `teacher.stopped_targets` / `_npneq_lag`): the loop's teacher had always fitted
    at lag 1 (`lags` in the config was unused). Stopped windows (t, min(t + lag, tau_t)); at lag 1 equal to the
    library update (test). loop_well_lgv uses lag 4 (staple paths with recorded v: lag 4 RMS 0.015 vs lag 1 0.019).
  - Runs on the workstation: the loop, the seed-lambda control as the long lambda = x infretis run, the long MD.
- **Results and workflow review (2026-11-18, evening).**
  - *loop_well_lgv* (workstation): production 1.008x the MD rate (rel. error 0.061; P_cross 0.641 +- 1.8%, the rest
    is flux), 8 interfaces, 0.95M MD to initialize, 1.67M in production. Student logit RMSE plateaus at 0.63-0.76
    from pass 3. Long MD (4096 x 2M steps, 50 walkers saved): k_AB 0.33709 +- 0.00014, and the MD committor matches
    q_exact (`loop/plot_well_lgv.py`). The velocity-CV machinery works. The system is not a rare event (plain MD at
    equal cost: about 2% error), so it cannot show a gain in efficiency.
  - *M3 replicates (seed 100)*: loop_zu_b_s2 1.50x (0.36), loop_zw_s2 0.65x (0.51), both within errors of seed 0.
    loop_zu_xy_b_s2: student RMSE 2.55 at pass 1 (seed 0: 1.24, same pass 0), 1.37 at pass 7, freeze refused (both
    verification rounds: skip_dn 3.7% in ensemble 5), no production.
  - *Workflow findings*: (1) the stop test never fires: all 7 loops ran to max_passes, because "interfaces changed" is
    true at every pass; (2) the student RMSE stops improving by pass 3 in 5 of 7 loops, and later passes sometimes
    make it worse; (3) the freeze takes the last pass, not the best one; (4) pass 0 is about half the initialization
    cost on the Z systems (6.9M of 11-18M); (5) the teacher always runs to its iteration cap (6000 on Z, 4000 on
    well_lgv), and no convergence history is saved; (6) between seeds the learned lambda varies more than the rate.
  - Proposed changes (not yet decided):
    - Stop rule: compare interface sets with a tolerance in q (or drop that condition) and keep the stability and
      prospective tests; until then use max_passes 4.
    - Freeze the best pass by an internal score (prospective SD(Z_q) or stability), not the last one. Whether those
      scores track the exact RMSE has to be checked on the 7 finished loops first.
    - Teacher iterations: refit one stored pass at 2x and 4x max_iter and compare against exact; save the history.
    - M5 efficiency: a higher-barrier variant of the well (rate about 1e-4 to 1e-5 per time unit), same machinery.
- **Offline checks: passes, teacher iterations, student budget (2026-11-19).** Scripts in
  `examples/param_lambda/loop/experiments/` (`extend_loop.py`, `teacher_iters.py`, `student_variants.py`); results
  under `op_loop/<loop>/experiments/`.
  - *Longer loops* (copies continued from pass 7 to 12; logit RMSE vs exact per pass, 8-12):
    - loop_zw_x12: 0.737, 0.659, 0.529, 0.503, 0.503. Still falling about 0.09 per pass at pass 7; levels off at
      0.50 from pass 10. 7 passes are too few here. Froze pass 12 (verification 1 OK), production prepared, not run.
    - loop_zu_xy_b_s2_x12: 0.954, 0.608, 1.093, 1.192, 1.228 (exact spread 0.09, 0.16, 0.41, 0.38, 0.54). Best at
      pass 9, then it drifts away; the passes that got worse were the ones flagged internally (stability 0.33 at pass
      10, spacing checks failing on the pass's own paths at 10-12). Froze pass 12 (verification 2 OK; 9 interfaces),
      i.e. a worse lambda than pass 9. Production prepared, not run.
    - So: no fixed pass count fits both. A pass whose own internal checks fail should not replace the previous lambda.
  - *Teacher iterations* (loop_zu_b, loop_zw; 8 chunks of 3000, each warm-started from the previous one):
    - Warm start (pass-7 labels) and cold start (linear ramp) agree from 6000 iterations on (RMS 0.019 vs 0.021 at
      15000, 0.103 vs 0.101 at 24000): within one pass budget the start is forgotten.
    - The result swings with the seed and not with the iteration count (RMS 0.02-0.12, logit RMSE 0.43-0.75 on
      loop_zu_b). On loop_zw, refitting the pass-7 labels makes them worse (logit 0.89 to 1.2-1.8).
    - More iterations do not help; 6000 is not too few.
  - *Teacher ensemble* (6 independent 6000-iteration fits from the same warm start, mean of the logits):
    - loop_zu_b: members 0.39-0.69, mean of 6: 0.273 (q RMS 0.026); better than every member and than the start (0.406).
    - loop_zw: members 0.60-0.89, mean of 6: 0.678 (q RMS 0.070, members 0.049-0.095); not monotone in the number of
      members (0.95 at 5). Logit averaging is sensitive to members that go to the clip limits; a median or q-space
      mean has not been tried yet.
  - *Student budget and architecture* (loop_zu_b pass-7 labels, 3 CVs y, x, w; teacher vs exact 0.278):

    | variant | exact logit RMSE | vs teacher | train s |
    |---|---|---|---|
    | [64,64], 24 epochs (loop) | 0.327 | 0.154 | (lock wait) |
    | [64,64], 96 epochs | 0.326 | 0.143 | 245 |
    | [128,128], 96 epochs | 0.324 | 0.138 | 150 |
    | [64,64,64], 96 epochs | 0.325 | 0.138 | 445 |

    Neither more epochs nor a larger network help: the student is limited by its labels, not by its capacity.
    w256_e96 was OOM-killed at 4 GB, and the warm variant and the 4-5 CV loops (loop_zw, loop_zu_xy_b) did not run yet.
- **Pass stop on accepted paths (`loop_zw_paths`, 2026-11-19).** `[run] pass_stop = "paths"` (passes.py
  `_extend_until_sampled`): a learning pass runs 1000 cycles, then blocks of 250 until every ensemble holds 100
  accepted paths after the 20% burn-in (unweighted: the weights only set the distribution of the same data); hard
  stop at 200 x n_ens accepted paths, safety cap 6000 cycles. Same pass 0 as loop_zw (`extend_loop --pass0-only`).
  - Pass 1 ran under a first version (stop at 2000 accepted paths in all: 3500 cycles); passes 2-5 stopped on target
    at 2500, 2750, 3250, 3250 cycles (1505-1901 paths), never at the hard stop. The bottleneck was always a middle
    ensemble (2 or 6), at the end ensemble 6; the first ensembles and those near B fill 2x faster.
  - Logit RMSE vs exact against the MD steps of the data the lambda was fitted on (pass 0 included):
    loop_zw_paths 1.51 (4.23M), 1.26 (5.80M), 0.835 (7.19M), 0.629 (8.95M), 0.571 (10.92M); loop_zw (1000-cycle
    passes) 0.976 (7.41M), 0.737 (8.68M), 0.659 (9.32M), 0.529 (10.01M), 0.503 (10.66M). Ahead up to about 9M,
    slightly behind at the end; within the teacher's fit-to-fit noise (about +-0.1-0.15). Same quality with 5
    teacher fits instead of 11. Total 12.9M MD steps (loop_zw_x12: 13.4M).
  - Froze pass 5 (verification 1 OK, 14 interfaces). Production (20 000 cycles, 12.8M MD steps): **1.19x exact**,
    rel. error 0.54 (P_cross plateau 0.46). loop_zw_x12 (pass 12, 14 interfaces): 0.94x, 0.69 at 13.0M.
    loop_zu_xy_b_s2_x12 (pass 12, 9 interfaces after verification round 2 removed three): 0.37x, 0.62 at 23.4M;
    its running rate falls from about 1.5x at 5M to 0.17x at 13M and climbs back to 0.37x, not converged. Its set
    has two wide gaps in q (0.00072 -> 0.043 and 0.043 -> 0.46), where the P_cross variance comes from. All four zw
    productions (1.14x, 0.65x, 0.94x, 1.19x) agree with exact within their errors; more passes or the paths rule
    did not reduce the production error at equal production MD.
  - Figures: `python -m loop.plots <loop>` now also writes `paths*.png` (the frozen lambda, exact isocommittors and
    two reactive + two LMR production paths). `_read_runav` cut fixed-width columns wrongly once MFPT > 1e6 (rows
    dropped, running rate truncated): fixed, all pcross_convergence.png regenerated (2026-11-19).
  - Storage (2026-11-19): every finished production / verification run and every pass run except pass 0 and each
    loop's last two passes archived to <run>/archive.h5 and load/ pruned to the active paths (`loop.archive_runs`).
    Readers take the archive through `infretis_backend.order_h5` whenever its check passed (a pruned load/ still
    exists, so its presence no longer means it is complete). Seeds skip pruned paths (no traj.txt).
  - Bug found and fixed on the way: the cost of an extended pass was recorded after its first block only (passes 1-2
    corrected from restart.toml; originals *.bak_costfix).
