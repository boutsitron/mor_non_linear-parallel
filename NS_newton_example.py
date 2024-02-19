"""Test file for Newton-Raphson iterations"""

import firedrake as fd
import numpy as np
from colorama import Fore
from firedrake import div, dot, ds, dx, grad, inner

from mor.morprojector import MORProjector
from parallel import Print, create_petsc_matrix

meshdir = "meshes/"
resultdir = "results/"


class Problem:
    r"""General class for a problem to import in thermal-fluid.py"""

    def __init__(
        self,
        Re: float,
        DIVISIONS: int = 100,
    ):
        r"""
        Args:
            Re (float): Reynolds number
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
        self.current_solution = None
        self.mesh = fd.UnitSquareMesh(DIVISIONS, DIVISIONS)
        self.Re = fd.Constant(Re)
        self.F, self.up = self.set_flow_formulation()
        self.flow_bcs = self.set_neumann_bcs()
        self.I_FOM = 20
        self.flow_problem, self.flow_solver = self.set_flow_problem()
        self.reduced_flow_solver = self.set_reduced_flow_solver()

    def set_neumann_bcs(self) -> tuple:
        """
        Create Neumann boundary conditions for the Navier-Stokes problem.

        Args:
            inlets (dict): Dictionary containing inlet information.
            outlets (dict): Dictionary containing outlet information.

        Returns:
            tuple: Tuple containing the boundary conditions and the modified UFL form.
        """
        boundary_conditions = {
            "INLET": 1,
            "OUTLET": 2,
            "WALLS": (3, 4),
        }
        W = self.F.arguments()[0].function_space()

        self.F -= self.v[0] * fd.Constant(1.0) * ds(boundary_conditions["INLET"])
        bcs_walls = fd.DirichletBC(
            W.sub(0), fd.Constant((0.0, 0.0)), boundary_conditions["WALLS"]
        )
        return [bcs_walls]

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
        self.appctx = {"projection_mat": self.PPhi}

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
            "ksp_monitor": None,
        }

        return fd.NonlinearVariationalSolver(
            self.flow_problem,
            appctx=self.appctx,
            solver_parameters=solver_parameters_rom,
        )


if __name__ == "__main__":

    problem = Problem(
        Re=1,
        DIVISIONS=200,
    )

    morproj = MORProjector()  # Initialize MORProjector

    problem.current_solution = None
    for parameter_counter, i in enumerate(range(1, 2000, 10), start=1):

        problem.Re.assign(i)

        if parameter_counter >= problem.I_FOM:
            Print(
                f"{Fore.RED}Try: #{parameter_counter:d}: Solving ROM for Re = {i}{Fore.RESET}"
            )
            problem.appctx["projection_mat"] = morproj.get_basis_mat()
            problem.reduced_flow_solver.solve()
        else:
            Print(
                f"{Fore.YELLOW}Try: #{parameter_counter:d}: Solving FOM for Re = {i}{Fore.RESET}"
            )
            problem.flow_solver.solve()

        # Take a snapshot for each solution
        morproj.take_snapshot(problem.up)
        # After collecting the snapshots, compute the basis on-the-fly
        if parameter_counter == problem.I_FOM - 1:
            Print(
                f"{Fore.GREEN}Computing basis for {parameter_counter:d} snapshots{Fore.RESET}"
            )
            morproj.compute_basis(parameter_counter, "L2")
        # Reassign the new Reynolds number

        problem.current_solution = problem.up

        u_plot, p_plot = problem.up.subfunctions
        u_plot.rename("velocity")
        p_plot.rename("pressure")
        stokes_pvd = fd.File(f"{resultdir}/navier_stokes.pvd")
        stokes_pvd.write(u_plot, p_plot, time=parameter_counter)

        Print("")
