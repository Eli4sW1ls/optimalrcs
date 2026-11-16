#!/usr/bin/env bash
# StapleTIS run (infinite swap, ASE overdamped Langevin). Needs the APPTIS checkouts of ASE (OverdampedLangevin),
# infretis (StapleTIS) and inftools in front of the installed packages, and the toytis checkout for ToytisPotential.
APPTIS=${APPTIS:-/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS}
export TOYTIS=${TOYTIS:-/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis}
export PYTHONPATH=$APPTIS/ase:$APPTIS/infretis:$APPTIS/inftools:$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash clean.sh
cp -r load_copy load
infretisrun -i infretis.toml
