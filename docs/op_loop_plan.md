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
