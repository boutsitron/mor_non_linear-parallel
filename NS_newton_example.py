"""Test file for Newton-Raphson iterations"""

import firedrake as fd
from firedrake import div, dot, dx, grad, inner

import config
from utils.create_domain import create_domain

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

boundary_conditions = {
    "INLET": tuple(inlets.keys()),
    "OUTLET": tuple(outlets),
    "WALLS": 3,
}
mesh = fd.Mesh(config.MESHDIR + meshfile)
V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)
W = V * Q
print(f"W # DOFS: {W.dim()}")
up, vq = fd.Function(W), fd.TestFunction(W)
u, p = fd.split(up)
v, q = fd.split(vq)
# physical problem
Re = fd.Constant(100.0)
F = (
    inner(grad(u), grad(v)) * dx  # diffusion term
    + Re * inner(dot(grad(u), u), v) * dx  # advection term
    - p * div(v) * dx  # pressure gradient
    + div(u) * q * dx  # mass conservation
)

# _, y = fd.SpatialCoordinate(W.ufl_domain())
# Extracting Y_INLET_START and Y_INLET_END from the first inlet position
Y_INLET_START = inlet_positions[0][1]  # The second element of the first tuple
Y_INLET_END = inlet_positions[0][3]  # The fourth element of the first tuple

# Calculating Y_INLET_RADIUS and Y_INLET_COORD based on the inlet's vertical extent
Y_INLET_RADIUS = (Y_INLET_END - Y_INLET_START) / 2.0  # Half of the spanwise width
Y_INLET_COORD = Y_INLET_START + Y_INLET_RADIUS  # Middle of the inlet

# Now, you can define the inflow profile using these calculated values
_, y = fd.SpatialCoordinate(mesh)  # Ensure mesh is defined and accessible
inflow = fd.as_vector(
    [
        1.0 - (((y - Y_INLET_COORD) / Y_INLET_RADIUS) ** 2),
        0.0,
    ]
)

bc1 = fd.DirichletBC(W.sub(0), inflow, boundary_conditions["INLET"])
bc2 = fd.DirichletBC(W.sub(0), fd.Constant((0.0, 0.0)), boundary_conditions["WALLS"])
bcs = [bc1, bc2]

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
u_plot.rename("Firedrake non-linear solver velocity")
p_plot.rename("Firedrake non-linear solver pressure")
stokes_pvd = fd.File(f"{config.RESULTDIR}/navier_stokes_firedrake.pvd")
stokes_pvd.write(u_plot, p_plot)

F_res = 1.0
up.assign(fd.zero())
for bc in bcs:
    bc.apply(up)
while F_res > 1e-8:
    Frhs = fd.assemble(-F, bcs=fd.homogenize(bcs))
    Jacobian = fd.assemble(
        fd.derivative(F, up), bcs=fd.homogenize(bcs)
    )  # Gateaux derivative dF/dup
    solver = fd.LinearSolver(Jacobian, solver_parameters=solver_parameters_direct)
    du = fd.Function(W)
    solver.solve(du, Frhs)
    up += du
    F_vec = fd.assemble(F, bcs=fd.homogenize(bcs))
    with F_vec.dat.vec_ro as F_petsc:
        F_res = F_petsc.norm()
    print(f"F_res: {F_res}")

u_plot, p_plot = up.subfunctions
u_plot.rename("Custom Newton-Raphson solver velocity")
p_plot.rename("Custom Newton-Raphson solver pressure")
stokes_pvd = fd.File(f"{config.RESULTDIR}/navier_stokes_NR.pvd")
stokes_pvd.write(u_plot, p_plot)
