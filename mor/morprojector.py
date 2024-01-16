from petsc4py import PETSc
import firedrake
from firedrake import TrialFunction, TestFunction, assemble, dx, inner, grad
import numpy as np
import scipy.sparse as sp


def petsc2sp(A):
    """Creates scipy sparse matrix/numpy array from a PETSc matrix/vector.

    :arg A: PETSc matrix/vector A
    :returns: Scipy sparse matrix/numpy array
    """
    if A.Type == PETSc.Vec.Type:
        return A.array.reshape(1, A.local_size)
    if A.getInfo()["nz_used"] == 0.0:
        return sp.csr_matrix(A.size, shape=A.local_size)
    Asp = A.getValuesCSR()[::-1]
    return sp.csr_matrix(Asp, shape=A.local_size)


class MORProjector(object):
    """Model Order Reduction Projector

    The main object to handle storage and projection using basis functions.

    :arg default_type: (optional) type for the assumed vector or matrix.
    """

    def __init__(self, default_type="petsc"):
        self.default_type = default_type
        self.snaps = []
        self.snap_mat = None
        self.n_basis = 0
        self.basis_mat = None

    def take_snapshot(self, u):
        """Store a snapshot

        :arg u: a :class:`firedrake.Function`
        """
        self.snaps.append(u.copy(deepcopy=True))

    def snapshots_to_matrix(self):
        """Convert the snapshots to a matrix suitable for mixed function spaces."""
        if not self.snaps:
            raise ValueError("No snapshots available to form a matrix.")

        # Determine the total number of DoFs for the mixed function space
        total_dofs = sum(fs.dof_count for fs in self.snaps[0].function_space())

        # Initialize the snapshot matrix
        self.snap_mat = np.zeros((total_dofs, len(self.snaps)))

        # Fill the matrix with data from each snapshot
        for col, snap in enumerate(self.snaps):
            # Start index for filling the snapshot data
            start_idx = 0

            for component in snap.split():  # Split the mixed space solution
                with component.dat.vec_ro as vec:
                    # Number of DoFs for this component
                    num_dofs = vec.getSize()

                    # Fill the corresponding section of the column
                    self.snap_mat[start_idx : start_idx + num_dofs, col] = vec.array

                    # Update the start index for the next component
                    start_idx += num_dofs

        return total_dofs

    def compute_basis(
        self, n_basis, inner_product="L2", time_scaling=False, delta_t=None
    ):
        """

        :arg n_basis: Number of basis.
        :arg inner_product: Type of inner product (L2 or H1).
        :arg time_scaling: Use time scaling.
        :arg delta_t: :class:`numpy.ndarray` with used timesteps to scale.
        :return: Estimated error.
        """
        # Build inner product matrix
        V = self.snaps[-1].function_space()
        if inner_product == "L2":
            ip_form = inner(TrialFunction(V), TestFunction(V)) * dx
        elif inner_product == "H1":
            ip_form = (
                inner(TrialFunction(V), TestFunction(V)) * dx
                + inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx
            )

        ip_mat = assemble(ip_form, mat_type="aij").M.handle

        M = self.snapshots_to_matrix()

        # This matrix is symmetric positive semidefinite
        corr_mat = np.matmul(
            self.snap_mat.transpose(), petsc2sp(ip_mat).dot(self.snap_mat)
        )

        # Build time scaling diagonal matrix
        if time_scaling is True and delta_t is not None:
            D = np.zeros((len(self.snaps), len(self.snaps)))
            for i in range(len(self.snaps)):
                D[i, i] = np.sqrt(delta_t[i])
            D[0, 0] = np.sqrt(delta_t[0] / 2.0)
            D[-1, -1] = np.sqrt(delta_t[-1] / 2.0)

            # D'MD
            corr_mat = np.matmul(D.transpose(), np.matmul(corr_mat, D))

        self.n_basis = n_basis

        # Compute eigenvalues, all real and non negative
        w, v = np.linalg.eigh(corr_mat)

        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]

        # Skip negative entries
        idx_neg = np.argwhere(w < 0)
        if len(idx_neg) > 0:
            # Reduce number of basis to min(n_basis, first_negative_eigenvalue)
            n_basis = np.minimum(n_basis, idx_neg[0][0])

        # Calculate the total DoFs for the mixed function space
        total_dofs = sum(fs.dof_count for fs in V)

        psi_mat = np.zeros((total_dofs, n_basis))

        for i in range(n_basis):
            psi_mat[:, i] = self.snap_mat.dot(v[:, i]) / np.sqrt(w[i])

        ratio = np.sum(w[:n_basis]) / np.sum(w[:M])

        self.basis_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.basis_mat.setType("dense")
        self.basis_mat.setSizes([total_dofs, n_basis])
        self.basis_mat.setUp()

        self.basis_mat.setValues(
            range(total_dofs),
            range(n_basis),
            psi_mat.reshape((total_dofs, n_basis)),
        )

        self.basis_mat.assemble()

        return ratio

    def get_basis_mat(self):
        """Return the basis mat"""
        assert self.basis_mat is not None
        return self.basis_mat

    def project_operator(self, oper, oper_type="petsc"):
        """Project an operator.

        :param oper: :class:`firedrake.matrix.Matrix`.
        :param oper_type: Type of returned operator.
        :return: Projected operator.
        """
        A = oper.M.handle if type(oper) is firedrake.matrix.Matrix else oper
        Ap = A.PtAP(self.basis_mat)

        if oper_type == "petsc":
            return Ap
        elif oper_type == "scipy":
            return petsc2sp(Ap).todense()

    def project_function(self, f, func_type="petsc"):
        """Project a function from full space to reduced space.

        :param f: :class:`firedrake.Function`.
        :param func_type: Type of returned function.
        :return: Projected function.
        """
        fp, _ = self.basis_mat.createVecs()
        if type(f) is firedrake.Function:
            with f.dat.vec as vec:
                self.basis_mat.multTranspose(vec, fp)
        else:
            self.basis_mat.multTranspose(f, fp)

        if func_type == "petsc":
            return fp
        elif func_type == "scipy":
            return fp.array

    def recover_function(self, fp, func_type="petsc"):
        """Recover project function from reduced space to full space.

        :param fp: Function of type func_type.
        :param func_type: Type of the input function.
        :return: Recovered function.
        """
        if func_type == "scipy":
            fp_petsc, _ = self.basis_mat.createVecs()
            fp_petsc.setValues(range(fp.shape[0]), fp)

        f = firedrake.Function(self.snaps[-1].function_space())
        with f.dat.vec as vec:
            if func_type == "scipy":
                self.basis_mat.mult(fp_petsc, vec)
            else:
                self.basis_mat.mult(fp, vec)

        return f
