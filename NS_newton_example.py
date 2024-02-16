"""Test file for Newton-Raphson iterations"""

import firedrake as fd
import numpy as np
from colorama import Fore
from firedrake import div, dot, ds, dx, grad, inner

from create_domain import create_domain
from mor.morprojector import MORProjector
from parallel import Print, create_petsc_matrix, print_matrix_partitioning

meshdir = "meshes/"
resultdir = "results/"


class Problem:
    r"""General class for a problem to import in thermal-fluid.py"""

    def __init__(
        self,
        Re: float,
        inlet_positions: list = None,
        outlet_positions: list = None,
        buffer_length: float = 0.4,
        lx: float = 2,
        ly: float = 2,
        DIVISIONS: int = 100,
        meshfile: str = "test.msh",
    ):
        r"""
        Args:
            Re (float): Reynolds number
            inlet_positions (list, optional): list with the positions of the inlets (of their projection onto the design
            domain, then they are shifted by buffer_length). Defaults to None
            outlet_positions (list, optional): list with the positions of the outlets (of their projection onto the
            design domain, then they are shifted by buffer_length). Defaults to None
            buffer_length (float, optional): length of the branches popping out of the design domain. Defaults to None
            lx (float, optional): x-length of the design domain. Defaults to None
            ly (float, optional): y-length of the design domain. Defaults to None
            DIVISIONS (int, optional): number of divisions in the mesh. Defaults to 200
            meshfile (str, optional): .msh file where the domain is saved or the mesh is going to be saved. Defaults to "test.msh"
        """
        # Call the constructor of the parent class (if needed)
        self.solver_parameters_direct = {
            "snes_monitor": None,
            "ksp_type": "preonly",
            "mat_type": "aij",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_24": 1,
        }
        inlets, outlets, _, _ = create_domain(
            inlet_positions=inlet_positions,
            outlet_positions=outlet_positions,
            buffer_length=buffer_length,
            lx=lx,
            ly=ly,
            meshfile=meshfile,
            DIVISIONS=DIVISIONS,
        )
        self.current_solution = None
        self.mesh = fd.Mesh(meshdir + meshfile)
        self.Re = fd.Constant(Re)
        self.F, self.up = self.set_flow_formulation()
        self.flow_bcs = self.set_neumann_bcs(inlets, outlets)
        self.I_FOM = 5
        self.flow_problem, self.flow_solver = self.set_flow_problem()
        self.reduced_flow_solver = self.set_reduced_flow_solver()

    def set_neumann_bcs(self, inlets: dict, outlets: dict) -> tuple:
        """
        Create Neumann boundary conditions for the Navier-Stokes problem.

        Args:
            inlets (dict): Dictionary containing inlet information.
            outlets (dict): Dictionary containing outlet information.

        Returns:
            tuple: Tuple containing the boundary conditions and the modified UFL form.
        """
        boundary_conditions = {
            "INLET": tuple(inlets.keys()),
            "OUTLET": tuple(outlets),
            "WALLS": 3,
        }
        W = self.F.arguments()[0].function_space()

        # Boundary conditions (see create_domain.py for explanation)
        for key in inlets.keys():
            sign = inlets[key][0]
            component = inlets[key][1]
            self.F -= sign * self.v[component] * fd.Constant(1.0) * ds(key)
        bcs_walls = fd.DirichletBC(
            W.sub(0), fd.Constant((0.0, 0.0)), boundary_conditions["WALLS"]
        )
        bcs = [bcs_walls]

        return bcs

    def set_flow_formulation(self) -> tuple:
        """
        Set the flow formulation for the Navier-Stokes problem.

        Returns:
            tuple: A tuple containing the variational form and the solution function
        """
        V = fd.VectorFunctionSpace(self.mesh, "CG", 2)
        Q = fd.FunctionSpace(self.mesh, "CG", 1)
        W = V * Q
        Print(f"W # DOFS: {W.dim()}")
        up, vq = fd.Function(W), fd.TestFunction(W)
        self.u, self.p = fd.split(up)
        self.v, self.q = fd.split(vq)
        # physical problem

        F = (
            1 / self.Re * inner(grad(self.u), grad(self.v)) * dx  # diffusion term
            + inner(dot(grad(self.u), self.u), self.v) * dx  # advection term
            - self.p * div(self.v) * dx  # pressure gradient
            + div(self.u) * self.q * dx  # mass conservation
        )

        return F, up

    def set_flow_problem(self):

        flow_problem = fd.NonlinearVariationalProblem(
            self.F, self.up, bcs=self.flow_bcs
        )
        flow_solver = fd.NonlinearVariationalSolver(
            flow_problem, solver_parameters=self.solver_parameters_direct
        )
        if self.current_solution is not None:
            flow_solver._problem.u.assign(self.current_solution)

        return flow_problem, flow_solver

    def set_reduced_flow_solver(self):

        m = self.up.function_space().dim()
        k = 1

        self.PPhi = create_petsc_matrix(np.zeros((m, k)))
        appctx = {"projection_mat": self.PPhi}

        solver_parameters_rom = {
            "snes_monitor": None,
            "snes_type": "newtonls",
            "mat_type": "matfree",
            "pc_type": "python",
            "pc_python_type": "mor.morpc.MORPC",
            "snes_linesearch_type": "bt",
            "snes_linesearch_monitor": None,
            "snes_linesearch_max_it": 10,
            "snes_linesearch_minlambda": 1e-4,
            # "snes_atol": 1e-7,
            # "snes_linesearch_damping": 1.0,
            # "ksp_monitor": None,
        }

        # Create a NonlinearVariationalSolver with the ROM preconditioner
        reduced_flow_solver = fd.NonlinearVariationalSolver(
            self.flow_problem,
            appctx=appctx,
            solver_parameters=solver_parameters_rom,
        )

        return reduced_flow_solver


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

    problem = Problem(
        Re=1,
        inlet_positions=inlet_positions,
        outlet_positions=outlet_positions,
        buffer_length=buffer_length,
        lx=lx,
        ly=ly,
        meshfile=meshfile,
    )

    morproj = MORProjector()  # Initialize MORProjector

    problem.current_solution = None
    parameter_counter = 1
    for i in range(1, 100, 5):

        Print(
            f"{Fore.YELLOW}Try: #{parameter_counter:d}: Solving for Re = {i}{Fore.RESET}"
        )

        # Take a snapshot for each solution
        morproj.take_snapshot(problem.up)
        # After collecting the snapshots, compute the basis on-the-fly
        morproj.compute_basis(parameter_counter, "L2")
        # Reassign the new Reynolds number
        problem.Re.assign(i)

        if parameter_counter >= problem.I_FOM:
            problem.PPhi = morproj.get_basis_mat()
            print_matrix_partitioning(problem.PPhi, "PPhi")
            problem.reduced_flow_solver.snes.setAppCtx({"projection_mat": problem.PPhi})
            problem.reduced_flow_solver.snes.appctx["projection_mat"] = problem.PPhi
            problem.reduced_flow_solver.solve()
        else:
            problem.flow_solver.solve()
        problem.current_solution = problem.up

        u_plot, p_plot = problem.up.subfunctions
        u_plot.rename("velocity")
        p_plot.rename("pressure")
        stokes_pvd = fd.File(f"{resultdir}/navier_stokes.pvd")
        stokes_pvd.write(u_plot, p_plot, time=iter)

        parameter_counter += 1

        Print("")
