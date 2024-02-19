from __future__ import absolute_import, print_function

import numpy as np
from colorama import Fore, Style
from firedrake import PCBase, assemble
from firedrake.assemble import allocate_matrix
from firedrake.petsc import PETSc

from parallel import create_petsc_matrix

tab = "    "


def Print(message: str, color: str = Fore.WHITE):
    """Print function that prints only on rank 0 with color

    Args:
        message (str): message to be printed
        color (str, optional): color of the message. Defaults to Fore.WHITE.
    """
    PETSc.Sys.Print(f"{color}{message}{Style.RESET_ALL}")


class MORPC(PCBase):
    def initialize(self, pc):
        _, K = pc.getOperators()
        self.ctx = K.getPythonContext()
        self.prefix = pc.getOptionsPrefix()

        self.assembleK()

        # Reassign K the Jacobian matrix
        K = self.K.M.handle

        # Get the POD modes which is the projection matrix
        self.Z = self.ctx.appctx.get("projection_mat", None)
        assert self.Z != None

        # Project the Jacobian onto the reduced basis
        Kp = K.ptap(self.Z)

        m, k = Kp.getSize()
        assert m == k, "Projected matrix is not square!"

        KpInv = PETSc.KSP().create()
        KpInv.setOperators(Kp)
        KpInv.setType("preonly")
        KpInv.setConvergenceHistory()
        KpInvPC = KpInv.getPC()
        KpInvPC.setFactorSolverType("lu")  # mumps
        # KpInv.setUp()
        self.KpInv = KpInv
        Print(f"{tab} Setting reduced system with: {KpInv.getType()}", Fore.MAGENTA)

        # Create reduced space vectors
        self.Xp, _ = self.Z.createVecs()
        self.Yp, _ = self.Z.createVecs()

    def update(self, pc):

        Print("Updating the PC", Fore.CYAN)

        _, K = pc.getOperators()
        self.ctx = K.getPythonContext()

        # Reassemble the Jacobian matrix
        self.assembleK()

        # Reassign K with the newly assembled matrix
        K = self.K.M.handle

        # Project the Jacobian onto the reduced basis
        Kp = K.ptap(self.Z)

        # Update the inverse of the projected Jacobian
        self.KpInv.setOperators(Kp)
        self.KpInv.setUp()

        # Print("")

    def apply(self, pc, X, Y):
        """Apply the preconditioner

        Args:
            pc (PETSc.PC): PETSc preconditioner object
            X (PETSc.Vec): residual vector (input) in the full space
            Y (PETSc.Vec): solution increment vector (output) in the full space
        """
        self.Z = self.ctx.appctx.get("projection_mat", None)
        _, k = self.Z.getSize()

        # Project X and Y into the projected space
        # self.Z.multTranspose(Y, self.Yp)
        self.Z.multTranspose(X, self.Xp)

        # Solve in the reduced space for the solution increment in the reduced space
        self.KpInv.solve(self.Xp, self.Yp)  # Yp = KpInv * Xp

        Print(
            f"{tab} Linear truncated system [{k}x{k}] converged in {self.KpInv.getIterationNumber()} iterations.",
            Fore.MAGENTA,
        )

        # Project x back to original space
        # self.Z.mult(self.Xp, X)  # X [mx1] = Phi [mxk] * Xp [kx1]
        self.Z.mult(self.Yp, Y)  # Y [mx1] = Phi [mxk] * Yp [kx1]

    def applyTranspose(self, pc, X, Y):
        pass

    def assembleK(self):
        ctx = self.ctx
        mat_type = PETSc.Options().getString(f"{self.prefix}assembled_mat_type", "aij")

        self.K = allocate_matrix(
            ctx.a,
            bcs=ctx.row_bcs,
            form_compiler_parameters=ctx.fc_params,
            mat_type=mat_type,
        )

        # Encapsulate the assembly process in a method
        self._assemble_K()

    def _assemble_K(self):
        # Perform the assembly
        assemble(
            self.ctx.a,
            tensor=self.K,
            bcs=self.ctx.row_bcs,
            form_compiler_parameters=self.ctx.fc_params,
        )

        self.mat_type = self.K.mat_type

        # Check if the assembled matrix is symmetric
        # is_symmetric = self.K.petscmat.isSymmetric()
