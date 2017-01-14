# Poisson solver with periodic and Dirichlet boundary conditions
#
#     - div grad u(x, y, z) = f(x, y, z)
#
from __future__ import print_function
import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()

def periodic_solver(f, V):

    # Create Krylov solver
    solver = PETScKrylovSolver('cg', 'hypre_amg')#'gmres', 'ilu')

    solver.parameters["absolute_tolerance"] = 1e-14
    solver.parameters["relative_tolerance"] = 1e-12
    solver.parameters["maximum_iterations"] = 1000
    #solver.parameters["monitor_convergence"] = True
    solver.parameters["convergence_norm_type"] = "true"
    #for item in solver.parameters.items(): print(item)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    A, b = assemble_system(a, L)
    phi = Function(V)

    solver.set_operator(A)

    # Create vector that spans the null space and normalize
    null_vec = Vector(phi.vector())
    V.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")

    # Create null space basis object and attach to PETSc matrix
    null_space = VectorSpaceBasis([null_vec])
    as_backend_type(A).set_nullspace(null_space)

    # Orthogonalize RHS vector b with respect to the null space
    null_space.orthogonalize(b);

    # Solve
    solver.solve(phi.vector(), b)
    return phi

def dirichlet_solver(f, V, bc):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    phi = Function(V)
    solve(a == L, phi, bc)
    return phi

def E_field(phi, VV):
    # Compute gradient
    V = phi.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'CG', degree)

    grad_phi = Function(VV)
    grad_phi = project(-1*grad(phi), VV)
    e_field = project(grad_phi, W)
    return e_field

def test_periodic_solver():
    divs = [[50, 50], [10, 10, 10]]
    l_unit = [[0., 0., 1., 1.], [0., 0., 0., 1., 1., 1.]]
    l_hyper = [[-1., -1, 1., 1.],[-1., -1., -1., 1., 1., 1.]]

    mesh_type = ["UnitHyperCube", "HyperCube"]
    tol = 1E-6

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")
            if i == 0:
                L = l_unit[j]
                mesh = UnitHyperCube(divisions)
            if i == 1:
                L = l_hyper[j]
                mesh = HyperCube(L, divisions)

            V, VV, V_g = periodic_bcs(mesh, L)

            if j == 0:
                class Source(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*(8.0*DOLFIN_PI*DOLFIN_PI)

                class Exact(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])

            if j == 1:
                class Source(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*cos(2.0*DOLFIN_PI*x[2])*(12.0*DOLFIN_PI*DOLFIN_PI)

                class Exact(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*cos(2.0*DOLFIN_PI*x[2])

            f = Source(degree=2)
            phi_e = Exact(degree=2)
            phi = periodic_solver(f, V)

            #error_l2 = errornorm(phi_e, phi, "L2")
            #print("l2 norm: ", error_l2)

            vertex_values_phi_e = phi_e.compute_vertex_values(mesh)
            vertex_values_phi = phi.compute_vertex_values(mesh)

            error_max = np.max(vertex_values_phi_e - \
                                vertex_values_phi)

            msg = 'error_max = %g' %error_max
            print("error_max: ", error_max)
            #assert error_max < tol , msg

            #plot(phi, interactive=True)
            #plot(phi_e, mesh=mesh, interactive=True)

            E = E_field(phi, V_g)
            plot(E, interactive=True)

def test_dirichlet_solver():
    divs = [[20,20], [20,20, 20]]
    L = [[-1., -1, 2., 1.], [-1., -1, 0, 2., 1., 1.]]
    mesh_type = ["UnitHyperCube", "HyperCube"]
    tol = 1E-10
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    f = Constant(-6.0)

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")
            if i == 0:
                mesh = UnitHyperCube(divisions)
            elif i == 1:
                mesh = HyperCube(L[j], divisions)
            bc, V, VV, V_g = dirichlet_bcs(u_D, mesh)
            phi = dirichlet_solver(f, V, bc)

            # error_l2 = errornorm(u_D, phi, "L2")
            # print("l2 norm: ", error_l2)

            vertex_values_u_D = u_D.compute_vertex_values(mesh)
            vertex_values_phi = phi.compute_vertex_values(mesh)

            error_max = np.max(vertex_values_u_D - \
                                vertex_values_phi)

            msg = 'error_max = %g' %error_max
            assert error_max < tol , msg

            plot(phi, interactive=True)
            #plot(u_D, mesh=mesh, interactive=True)

            # The gradient:
            E = E_field(phi, V_g)
            plot(E, interactive=True)
            # flux_u_x, flux_u_y = E.split(deepcopy=True)

            # Exact flux expressions
            # u_e = lambda x, y: 1 + x**2 + 2*y**2
            # flux_x_exact = lambda x, y: -2*x
            # flux_y_exact = lambda x, y: -4*y
            #
            # # Compute error in flux
            # coor = phi.function_space().mesh().coordinates()
            # for i, value in enumerate(flux_u_x.compute_vertex_values()):
            #     print('vertex %d, x = %s, -p*u_x = %g, error = %g' %
            #           (i, tuple(coor[i]), value, flux_x_exact(*coor[i])))
            # for i, value in enumerate(flux_u_y.compute_vertex_values()):
            #     print('vertex %d, x = %s, -p*u_y = %g, error = %g' %
            #           (i, tuple(coor[i]), value, flux_y_exact(*coor[i])))

def test_D_solver():
    divs = [[3,3], [5,5], [10,10]]
    L = [[-1., -1, 2., 1.]]
    mesh_type = ["UnitHyperCube", "HyperCube"]
    tol = 1E-10
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    f = Constant(-6.0)

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        h = []
        E = []
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")
            if i == 0:
                mesh = UnitHyperCube(divisions)
            elif i == 1:
                mesh = HyperCube(L[0], divisions)
            bc, V, VV, V_g = dirichlet_bcs(u_D, mesh)
            phi = dirichlet_solver(f, V, bc)
            E.append(errornorm(u_D, phi, "l2"))
            h.append(1./divisions[0])
        from math import log as ln
        for i in range(1, len(E)):
            r = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
            print("h =%10.2E E =%10.2E r =%.2f" %(h[i], E[i], r))


def test_p_solver():
    divs = [[3,3], [5,5], [10,10], [20,20], [30,30], [40,40], [50,50], [100,100]]
    l_unit = [[0., 0., 1., 1.]]
    l_hyper = [[-1., -1, 1., 1.]]

    mesh_type = ["UnitHyperCube", "HyperCube"]
    tol = 1E-10

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        h = []
        E1 = []
        E2 = []
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")

            if i == 0:
                L = l_unit[0]
                mesh = UnitHyperCube(divisions)
            if i == 1:
                L = l_hyper[0]
                mesh = HyperCube(L, divisions)

            V, VV, V_g = periodic_bcs(mesh, L)

            if j == 0:
                class Source(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*(8.0*DOLFIN_PI*DOLFIN_PI)

                class Exact(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])

                class Exact_E(Expression):
                    def eval(self, values, x):
                        values[0] = -2.0*DOLFIN_PI*cos(2.0*DOLFIN_PI*x[0])
                        values[1] = -2.0*DOLFIN_PI*cos(2.0*DOLFIN_PI*x[1])

            f = Source(degree=4)
            phi_e = Exact(degree=4)
            E_e = Exact_E(degree=4)

            phi = periodic_solver(f, V)
            E = E_field(phi, V_g)

            E1.append(errornorm(phi_e, phi, "l2"))
            E2.append(errornorm(E_e, E, "l2"))
            h.append(1./divisions[0])
        from math import log as ln
        for i in range(1, len(E1)):
            r1 = ln(E1[i]/E1[i-1])/ln(h[i]/h[i-1])
            print("h =%10.2E E1 =%10.2E r1 =%.2f" %(h[i], E1[i], r1))
            r2 = ln(E2[i]/E2[i-1])/ln(h[i]/h[i-1])
            print("h =%10.2E E1 =%10.2E r2 =%.2f" %(h[i], E2[i], r2))
if __name__ == '__main__':
    import time
    from mesh_types import *
    from boundary_conditions import *

    #test_periodic_solver()
    #test_dirichlet_solver()

    test_p_solver()
    #test_D_solver()
