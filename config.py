"""Config file with hyperparameter setting"""

#
# Created on Mon Aug 8 2022
#
# The Corintis License (Corintis)
# Copyright (c) 2022 Athanasios Boutsikakis
#
import os

from firedrake import COMM_WORLD

# Directories for saving files
OUTDIR = "./"
MESHDIR = "meshes/"
os.makedirs(OUTDIR + MESHDIR, exist_ok=True)
RESULTDIR = "results/"
os.makedirs(OUTDIR + RESULTDIR, exist_ok=True)
PHIDIR = "POMs/"
os.makedirs(OUTDIR + PHIDIR, exist_ok=True)
