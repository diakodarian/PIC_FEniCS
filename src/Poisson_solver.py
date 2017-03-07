# Poisson solver with periodic and Dirichlet boundary conditions
#
#     - div grad u(x, y, z) = f(x, y, z)
#
from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import numpy as np
from dolfin import *


class PoissonSolverPeriodic:

    def __init__(self, V):

        self.solver = PETScKrylovSolver('gmres', 'hypre_amg')
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000

        self.V = V

        phi = TrialFunction(V)
        phi_ = TestFunction(V)

        self.a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
        A = assemble(self.a)

        self.solver.set_operator(A)
        self.phi_ = phi_

        phi = Function(V)
        null_vec = Vector(phi.vector())
        V.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0/null_vec.norm("l2")

        self.null_space = VectorSpaceBasis([null_vec])
        as_backend_type(A).set_nullspace(self.null_space)

    def solve(self, rho, object_bcs = None):
        L = rho*self.phi_*dx
        if object_bcs is None:
            b = assemble(L)
        else:
            A, b = assemble_system(self.a, L, object_bcs)

        self.null_space.orthogonalize(b)

        phi = Function(self.V)
    	self.solver.solve(phi.vector(), b)

    	return phi

class PoissonSolverDirichlet:

    def __init__(self, V, bcs):

        self.solver = PETScKrylovSolver('gmres', 'hypre_amg')
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000

        self.V = V
        self.bcs = bcs

        phi = TrialFunction(V)
        phi_ = TestFunction(V)

        a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
        self.A = assemble(a)
        [bc.apply(self.A) for bc in self.bcs]

        self.phi_ = phi_

    def solve(self, rho, object_bcs = None):

        L = rho*self.phi_*dx
        b = assemble(L)
        [bc.apply(b) for bc in self.bcs]

        phi = Function(self.V)
        if object_bcs is None:
            self.solver.solve(self.A, phi.vector(), b)
        else:
            [bc.apply(self.A) for bc in object_bcs]
            [bc.apply(b) for bc in object_bcs]

            self.solver.solve(self.A, phi.vector(), b)

        return phi

def electric_field(phi, constr = None):
    """ This function calculates the gradiant of the electric potential, which
    is the electric field:

            E = -\del\varPhi

    Args:
          phi   : The electric potential.
          constr: constrained_domain

    returns:
          E: The electric field.
    """
    V = phi.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'CG', degree, constrained_domain=constr)
    return project(-1*grad(phi), W)

if __name__=='__main__':

    from boundary_conditions import *
    import sys
    sys.path.insert(0, '/home/diako/Documents/FEniCS/demos')

    from mark_object import *

    object_type = None
    object_info = []
    n_components = 0

    def test_periodic_solver():

        solver = PETScKrylovSolver('cg', 'hypre_amg')

        solver.parameters["absolute_tolerance"] = 1e-14
        solver.parameters["relative_tolerance"] = 1e-12
        solver.parameters["maximum_iterations"] = 1000
        # solver.parameters["monitor_convergence"] = True
        # solver.parameters["convergence_norm_type"] = "true"
        # for item in solver.parameters.items(): print(item)
        # solver.set_reuse_preconditioner(True)


        # mesh = Mesh("demos/mesh/rectangle_periodic.xml")
        Lx = 2*DOLFIN_PI
    	Ly = 2*DOLFIN_PI
    	Nx = 256
    	Ny = 256
    	mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),Nx,Ny)

        d = mesh.geometry().dim()
        L = np.empty(2*d)
        for i in range(d):
            l_min = mesh.coordinates()[:,i].min()
            l_max = mesh.coordinates()[:,i].max()
            L[i] = l_min
            L[d+i] = l_max


        PBC = periodic_bcs(mesh, L)
        V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)

        class Source(Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])

        class Exact(Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])

        f = Source(degree=2)
        phi_e = Exact(degree=2)

        poisson = PoissonSolverPeriodic(V)
        phi = poisson.solve(f)

        # phi = periodic_solver(f, V, solver)

        error_l2 = errornorm(phi_e, phi, "L2")
        print("l2 norm: ", error_l2)

        vertex_values_phi_e = phi_e.compute_vertex_values(mesh)
        vertex_values_phi = phi.compute_vertex_values(mesh)

        error_max = np.max(vertex_values_phi_e - \
                            vertex_values_phi)
        tol = 1E-9
        msg = 'error_max = %g' %error_max
        print(msg)
        assert error_max < tol , msg

        plot(phi, interactive=True)
        plot(phi_e, mesh=mesh, interactive=True)


    def test_dirichlet_solver():
        mesh = Mesh("demos/mesh/rectangle.xml")
        V = FunctionSpace(mesh, "P", 1)
        d = mesh.geometry().dim()

        L = np.empty(2*d)
        for i in range(d):
            l_min = mesh.coordinates()[:,i].min()
            l_max = mesh.coordinates()[:,i].max()
            L[i] = l_min
            L[d+i] = l_max

        u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
        f = Constant(-6.0)

        facet_f = mark_boundaries(mesh, L, object_type, object_info, n_components)
        plot(facet_f, interactive=True)

        bcs = dirichlet_bcs(V, facet_f, n_components, phi0 = u_D)

        # phi = dirichlet_solver(f, V, bcs, bc_object=None)
        poisson = PoissonSolverDirichlet(V, bcs)
        phi = poisson.solve(f)

        error_l2 = errornorm(u_D, phi, "L2")
        print("l2 norm: ", error_l2)

        vertex_values_u_D = u_D.compute_vertex_values(mesh)
        vertex_values_phi = phi.compute_vertex_values(mesh)

        error_max = np.max(vertex_values_u_D - \
                            vertex_values_phi)
        tol = 1E-10
        msg = 'error_max = %g' %error_max
        assert error_max < tol , msg

        plot(phi, interactive=True)

    test_periodic_solver()
