#!/usr/bin/env bash
# StapleTIS run (infinite swap, ASE overdamped Langevin). Needs the APPTIS checkouts of ASE (OverdampedLangevin),
# infretis (StapleTIS) and inftools in front of the installed packages, and the toytis checkout for ToytisPotential.
APPTIS=${APPTIS:-/home/staff7/fbsskr/Notebooks/Students/elias}
export TOYTIS=${TOYTIS:-/home/staff7/fbsskr/Notebooks/Students/elias/toytis}
export PYTHONPATH=$APPTIS/ase:$APPTIS/infretis:$APPTIS/inftools:$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash clean.sh
cp -r load_copy load
infretisrun -i infretis.toml
