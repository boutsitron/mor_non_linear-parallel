"""Test file for Newton-Raphson iterations"""

import firedrake as fd
from colorama import Fore
from firedrake import div, dot, ds, dx, grad, inner

import config
from mor.morpc import Print
from utils.create_domain import create_domain


def create_neumann_bcs(inlets, boundary_conditions, F, v):

    W = F.arguments()[0].function_space()

    # Boundary conditions (see create_domain.py for explanation)
    for key in inlets.keys():
        sign = inlets[key][0]
        component = inlets[key][1]
        F -= sign * v[component] * fd.Constant(1.0) * ds(key)
    bcs_walls = fd.DirichletBC(
        W.sub(0), fd.Constant((0.0, 0.0)), boundary_conditions["WALLS"]
    )
    bcs = [bcs_walls]

    return bcs, F


def define_problem(Re, inlets, outlets, meshfile):

    boundary_conditions = {
        "INLET": tuple(inlets.keys()),
        "OUTLET": tuple(outlets),
        "WALLS": 3,
    }
    mesh = fd.Mesh(config.MESHDIR + meshfile)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    Q = fd.FunctionSpace(mesh, "CG", 1)
    W = V * Q
    Print(f"W # DOFS: {W.dim()}")
    up, vq = fd.Function(W), fd.TestFunction(W)
    u, p = fd.split(up)
    v, q = fd.split(vq)
    # physical problem
    F = (
        1 / Re * inner(grad(u), grad(v)) * dx  # diffusion term
        + inner(dot(grad(u), u), v) * dx  # advection term
        - p * div(v) * dx  # pressure gradient
        + div(u) * q * dx  # mass conservation
    )

    bcs, F = create_neumann_bcs(inlets, boundary_conditions, F, v)

    return F, up, bcs


def solve_ns(F, up, bcs, iter):

    solver_parameters_direct = {
        "snes_monitor": None,
        "ksp_type": "preonly",
        "mat_type": "aij",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 200,
        "mat_mumps_icntl_24": 1,
    }
    problem = fd.NonlinearVariationalProblem(F, up, bcs=bcs)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters_direct
    )
    solver.solve()
    u_plot, p_plot = up.subfunctions
    u_plot.rename("velocity")
    p_plot.rename("pressure")
    stokes_pvd = fd.File(f"{config.RESULTDIR}/navier_stokes.pvd")
    stokes_pvd.write(u_plot, p_plot, time=iter)


if __name__ == "__main__":

    meshfile = "test.msh"

    branches_positions = [
        (0, 1.2, 0, 1.45),
        (0, 0.1, 0, 0.4),
        (0.95, 0, 1.2, 0),
        (2, 1.2, 2, 1.45),
        (2, 0.5, 2, 0.7),
    ]
    lx = ly = 2
    buffer_length = 0.4

    inlet_positions = [branches_positions[0]]
    outlet_positions = [
        branches_positions[1],
        branches_positions[2],
        branches_positions[3],
        branches_positions[4],
    ]

    inlets, outlets, DESIGN, _ = create_domain(
        inlet_positions,
        outlet_positions,
        buffer_length=0.4,
        lx=lx,
        ly=ly,
        meshfile=meshfile,
        DIVISIONS=100,
    )

    for i in range(10, 200, 10):

        Re = fd.Constant(i)

        Print(f"{Fore.YELLOW}Solving for Re = {i}{Fore.RESET}")

        F, up, flow_bcs = define_problem(Re, inlets, outlets, meshfile)
        solve_ns(F, up, flow_bcs, iter=i)
        Print("")
