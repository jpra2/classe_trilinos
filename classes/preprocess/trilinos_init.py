from PyTrilinos import Epetra, AztecOO, EpetraExt, Teuchos
import numpy as np
import scipy.sparse as sp


class PyTrilWrap:

    def __init__(self, p=1):
        self.__comm  = Epetra.PyComm()
        self.__params = dict()
        self.set_parameters()
        help(Teuchos)
        import pdb; pdb.set_trace()

    def solve_linear_problem(self, A, b, its=1000, tolerance=1e-10):
        '''
        resolve o problema Ax = b
        input:
            A: matriz quadrada do scipy
            b = termo fonte
        output:
            res: informa se o residuo foi menor que a tolerancia
            x: vetor resposta
        '''
        comm = self.comm
        n = len(b)
        std_map = Epetra.Map(n, 0, comm)
        x = Epetra.Vector(std_map)
        b2 = Epetra.Vector(std_map)
        b2[:] = b[:]
        A2 = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        indices = sp.find(A)
        A2.InsertGlobalValues(indices[0], indices[1], indices[2])
        irr = A2.FillComplete()
        linearProblem = Epetra.LinearProblem(A2, x, b2)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.SetParameters(self.__params)
        solver.Iterate(its, tolerance)
        x = np.array(x)
        res = solver.ScaledResidual() < tolerance
        return x, res

    def set_parameters(self, params=None):
        if params:
            pass
        else:
            params = {'Solver': 'GMRES',
                      'Precond': 'Jacobi'}

        self.__params.update(params)
