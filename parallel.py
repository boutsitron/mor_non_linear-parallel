"""Parallel utilities for PETSc matrices and vectors"""

#
# Created on Mon 19 2022
#
# The Corintis License (Corintis)
# Copyright (c) 2022 Athanasios Boutsikakis
#
#
from colorama import Fore, Style
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size

# --------------------------------------------
# Parallel print functions
# --------------------------------------------


def Print(message: str, color: str = Fore.WHITE):
    """Print function that prints only on rank 0 with color

    Args:
        message (str): message to be printed
        color (str, optional): color of the message. Defaults to Fore.WHITE.
    """
    PETSc.Sys.Print(f"{color}{message}{Style.RESET_ALL}")


def print_matrix_partitioning(mat, name="", values=False):  # sourcery skip: move-assign
    """Prints partitioning information of a PETSc MPI matrix.

    Args:
        mat (PETSc Mat): The PETSc MPI matrix.
        name (str): Optional name for the matrix for better identification in printout.
        values (bool): Toggle for printing the local values of the vector. Defaults to False.
    """
    # Get the local ownership range for rows
    local_rows_start, local_rows_end = mat.getOwnershipRange()
    # Collect all the local ownership ranges and local rows in the root process
    ownership_ranges_rows = COMM_WORLD.gather(
        (local_rows_start, local_rows_end), root=0
    )

    # Initialize an empty list to hold local row values
    local_rows = []
    for i in range(local_rows_start, local_rows_end):
        cols, row_data = mat.getRow(i)
        local_rows.append((i, list(zip(cols, row_data))))
    all_local_rows = COMM_WORLD.gather(local_rows, root=0)

    Print(
        f"{Fore.YELLOW}MATRIX {mat.getType()} {name} [{mat.getSize()[0]}x{mat.getSize()[1]}]{Style.RESET_ALL}"
    )
    if mat.isAssembled():
        Print(f"{Fore.GREEN}Assembled{Style.RESET_ALL}")
    else:
        Print(f"{Fore.RED}Not Assembled{Style.RESET_ALL}")
    Print("")
    Print(f"Partitioning for matrix {name}:")
    for i, ((start, end), local_rows) in enumerate(
        zip(ownership_ranges_rows, all_local_rows)
    ):
        Print(f"  Rank {i}: {end-start} rows [{start}, {end})")
        if values:
            for row_idx, row_data in local_rows:
                Print(f"    Row {row_idx}: {row_data}")


# --------------------------------------------
# PETSc matrices
# --------------------------------------------


def create_petsc_matrix(input_array, partition_like=None, sparse=True):
    """Create a PETSc matrix from an input_array

    Args:
        input_array (np array): Input array
        partition_like (PETSc mat, optional): Petsc matrix. Defaults to None.
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        PETSc mat: PETSc mpi matrix
    """
    # Check if input_array is 1D and reshape if necessary
    assert len(input_array.shape) == 2, "Input array should be 2-dimensional"
    global_rows, global_cols = input_array.shape

    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        # No parallelization in the columns, set local_cols = None to parallelize
        size = ((local_rows, global_rows), (None, global_cols))
    else:
        size = ((None, global_rows), (None, global_cols))

    # Create a sparse or dense matrix based on the 'sparse' argument
    if sparse:
        matrix = PETSc.Mat().createAIJ(size=size, comm=COMM_WORLD)
    else:
        matrix = PETSc.Mat().createDense(size=size, comm=COMM_WORLD)
    matrix.setUp()

    local_rows_start, local_rows_end = matrix.getOwnershipRange()

    for counter, i in enumerate(range(local_rows_start, local_rows_end)):
        # Calculate the correct row in the array for the current process
        row_in_array = counter + local_rows_start
        matrix.setValues(
            i, range(global_cols), input_array[row_in_array, :], addv=False
        )

    # Assembly the matrix to compute the final structure
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix
