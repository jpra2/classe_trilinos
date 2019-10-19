from PyTrilinos import Epetra, AztecOO, EpetraExt
import numpy as np
import scipy.sparse as sp


class TrilInit:

    def __init__(self, p=1):
        self.comm  = Epetra.PyComm()

    def solve_linear_problem(self, A, b):
        comm = self.comm
        n = len(b)
        std_map = Epetra.Map(n, 0, comm)
        x = Epetra.Vector(std_map)
        b2 = Epetra.Vector(std_map)
        b2[:] = b[:]
        A2 = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        indices = sp.find(A)
        A2.InsertGlobalValues(indices[0], indices[1], indices[2])
        linearProblem = Epetra.LinearProblem(A2, x, b2)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-12)
        x = np.array(x)

        return x
