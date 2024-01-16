import os
import itertools
import numpy as np
from firedrake import *
import time


import sys
import os

# Get the directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (or another appropriate directory) to sys.path
sys.path.append(os.path.join(script_dir, ".."))

# Import MORProjector from the mor package
from mor.morprojector import MORProjector

resultdir = "mor_results"


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(f"[{self.name}]")
            print(f"Elapsed: {time.time() - self.tstart}")


def create_directories():
    output_dir = f"{script_dir}/{resultdir}/poisson_mixed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def setup_poisson_problem():
    mesh = UnitSquareMesh(20, 20)
    Sigma = FunctionSpace(mesh, "RT", 2)
    V = FunctionSpace(mesh, "DG", 1)
    W = Sigma * V
    return mesh, W


def test_parametric_poisson():
    mesh, W = setup_poisson_problem()
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    morproj = MORProjector()  # Initialize MORProjector
    output_dir = create_directories()
    output_pvd = File(f"{output_dir}/potential_flux.pvd")

    x, y = SpatialCoordinate(mesh)
    f1_range = np.arange(1, 1.6, 0.2)
    f2_range = np.arange(1, 1.6, 0.2)

    parameter_counter = 0
    for f1, f2 in itertools.product(f1_range, f2_range):
        f = sin(f1 * pi * x) * cos(f2 * pi * y)

        n = FacetNormal(mesh)
        a = dot(sigma, tau) * dx + div(tau) * u * dx + div(sigma) * v * dx
        L = -f * v * dx + Constant(0.0) * dot(tau, n) * ds

        w_sol = Function(W)

        solve(a == L, w_sol)

        # Split the mixed solution
        sigma_save, u_save = w_sol.split()
        sigma_save.rename("flux")
        u_save.rename("potential")

        output_pvd.write(sigma_save, u_save, time=parameter_counter)

        # Take a snapshot for each solution
        morproj.take_snapshot(w_sol)

        parameter_counter += 1

    # Solve another problem using FOM
    new_f1, new_f2 = 1.65, 1.65  # New parameters for the test problem
    new_f = sin(new_f1 * pi * x) * cos(new_f2 * pi * y)
    new_L = -new_f * v * dx + Constant(0.0) * dot(tau, n) * ds

    # Full solve
    full_solve_sol = Function(W)
    solve(a == new_L, full_solve_sol)

    # After collecting all snapshots, compute the basis
    n_basis = min(50, parameter_counter)  # For example, up to 50 basis vectors
    morproj.compute_basis(n_basis, "L2")
    basis_mat = morproj.get_basis_mat()

    print(basis_mat.getSizes())

    # Define appctx for the custom preconditioner
    appctx = {"projection_mat": basis_mat}

    # Define the solver parameters for the ROM solve with a MOR preconditioner
    solver_parameters_rom = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "mor.morpc.MORPC",
    }

    # ROM solve
    rom_solve_sol = Function(W)
    prob_rom = LinearVariationalProblem(
        a,
        new_L,
        rom_solve_sol,
    )
    solver_rom = LinearVariationalSolver(
        prob_rom,
        appctx=appctx,
        solver_parameters=solver_parameters_rom,
    )

    with Timer("Reduced solve"):
        solver_rom.solve()

    # Calculate the error between FOM and ROM solutions
    error_norm = errornorm(full_solve_sol, rom_solve_sol)
    print("Error between FOM and ROM solutions:", error_norm)

    # Calculate the error between FOM and ROM solutions
    error_norm = errornorm(full_solve_sol, rom_solve_sol)
    print("Error between FOM and ROM solutions:", error_norm)

    # Split the solutions into components
    full_solve_sol_sigma, full_solve_sol_u = full_solve_sol.split()
    rom_solve_sol_sigma, rom_solve_sol_u = rom_solve_sol.split()

    sigma_diff = Function(W.sub(0))  # Create a new Function in the appropriate subspace
    sigma_diff.assign(full_solve_sol_sigma - rom_solve_sol_sigma)
    sigma_diff.rename("sigma diff")

    u_diff = Function(W.sub(1))  # Create a new Function in the appropriate subspace
    u_diff.assign(full_solve_sol_u - rom_solve_sol_u)
    u_diff.rename("u diff")

    full_solve_sol_sigma.rename("FOM sigma")
    full_solve_sol_u.rename("FOM u")
    rom_solve_sol_sigma.rename("ROM sigma")
    rom_solve_sol_u.rename("ROM u")

    # Write each component separately
    output_pvd_rom = File(f"{output_dir}/mixed_poisson_solution.pvd")
    output_pvd_rom.write(
        full_solve_sol_u,
        rom_solve_sol_u,
        full_solve_sol_sigma,
        rom_solve_sol_sigma,
        sigma_diff,
        u_diff,
    )

    # Assert that the error is below a certain threshold
    assert error_norm < 1e-5, f"Error norm is {error_norm}"
