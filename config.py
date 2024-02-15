"""Config file with hyperparameter setting"""

#
# Created on Mon Aug 8 2022
#
# The Corintis License (Corintis)
# Copyright (c) 2022 Athanasios Boutsikakis
#
import os

from colorama import Fore, Style
from firedrake import COMM_WORLD
from firedrake.petsc import PETSc

rank = COMM_WORLD.rank


def Print(message: str, color: str = Fore.WHITE):
    """Print function that prints only on rank 0 with color

    Args:
        message (str): message to be printed
        color (str, optional): color of the message. Defaults to Fore.WHITE.
    """
    PETSc.Sys.Print(f"{color}{message}{Style.RESET_ALL}")


# Directories for saving files
OUTDIR = "./"
MESHDIR = "meshes/"
os.makedirs(OUTDIR + MESHDIR, exist_ok=True)
RESULTDIR = "results/"
os.makedirs(OUTDIR + RESULTDIR, exist_ok=True)
PHIDIR = "POMs/"
os.makedirs(OUTDIR + PHIDIR, exist_ok=True)
