from firedrake import *
import firedrake as fd
import sys
import os

# Get the directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (or another appropriate directory) to sys.path
sys.path.append(os.path.join(script_dir, ".."))

from mor import *

from mor.morprojector import MORProjector
import time

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
    output_dir = f"{script_dir}/{resultdir}/poisson"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


N = 256

mesh = UnitSquareMesh(N, N)

V = FunctionSpace(mesh, "CG", 2)

u = Function(V)
v = TestFunction(V)

x = SpatialCoordinate(mesh)
f = Function(V)
f.interpolate(sin(x[0] * pi) * sin(2 * x[1] * pi))

R = inner(grad(u), grad(v)) * dx - f * v * dx

bcs = [DirichletBC(V, Constant(2.0), (1,))]

# Full solve
with Timer("Full solve"):
    solve(
        R == 0, u, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
    )

full_solve_sol = Function(V)
full_solve_sol.assign(u)
full_solve_sol.rename("FOM solution")


morproj = MORProjector()
morproj.take_snapshot(u)
morproj.compute_basis(1, "L2")
basis_mat = morproj.get_basis_mat()

print(basis_mat.getSizes())

appctx = {"projection_mat": basis_mat}

prob = NonlinearVariationalProblem(R, u, bcs=bcs)
solver = NonlinearVariationalSolver(
    prob,
    appctx=appctx,
    solver_parameters={
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "mor.morpc.MORPC",
    },
)

with Timer("Reduced solve"):
    solver.solve()

u.rename("ROM solution")

output_dir = create_directories()

output_pvd_rom = File(f"{output_dir}/poisson_solution.pvd")
output_pvd_rom.write(full_solve_sol, u)

error_norm = errornorm(full_solve_sol, u)
print(error_norm)

# Assert that the error is below a certain threshold
assert error_norm < 1e-5  # You can adjust the threshold as needed
