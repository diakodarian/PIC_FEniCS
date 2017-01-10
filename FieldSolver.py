# Poisson solver with periodic boundary conditions
#
#     - div grad u(x, y, z) = f(x, y, z)
#
# on the unit square with periodic boundary conditions
import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()

def periodic_solver(f, mesh, V, V_e):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    A, b = assemble_system(a, L)
    uh = Function(V)

    # Create Krylov solver
    solver = PETScKrylovSolver('cg', 'hypre_amg')
    solver.set_operator(A)

    # Create vector that spans the null space and normalize
    null_vec = Vector(uh.vector())
    V.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")

    # Create null space basis object and attach to PETSc matrix
    null_space = VectorSpaceBasis([null_vec])
    as_backend_type(A).set_nullspace(null_space)

    # Orthogonalize RHS vector b with respect to the null space (this
    # gurantees a solution exists)
    null_space.orthogonalize(b);

    # Solve
    solver.solve(uh.vector(), b)

    # Compute gradient
    method1 = False
    if method1:
        v = TestFunction(V_e)
        w = TrialFunction(V_e)

        a = inner(w, v)*dx
        L = inner(grad(uh), v)*dx
        A, b = assemble_system(a, L)
        grad_u = Function(V_e)

        # Create Krylov solver
        solver = PETScKrylovSolver("cg")
        solver.set_operator(A)

        # Create vector that spans the null space and normalize
        null_vec = Vector(grad_u.vector())
        V.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0/null_vec.norm("l2")

        # Create null space basis object and attach to PETSc matrix
        null_space = VectorSpaceBasis([null_vec])
        as_backend_type(A).set_nullspace(null_space)

        # Orthogonalize RHS vector b with respect to the null space (this
        # gurantees a solution exists)
        null_space.orthogonalize(b);

        # Solve
        solver.solve(-1*grad_u.vector(), b)
    else:
        grad_u = Function(V_e)
        grad_u = project(-1*grad(uh), V_e)

    #grad_u_x, grad_u_y = grad_u.split(deepcopy=True)  # extract component

    return uh, grad_u

def dirichlet_solver(f, V, V_g, bc):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    A, b = assemble_system(a, L, bc)
    # Compute solution
    u = Function(V)
    solver = KrylovSolver(A, "cg", "petsc_amg")
    solver.solve(u.vector(), b)

    # Compute the gradient
    grad_u = Function(V_g)
    grad_u = project(-1*grad(u), V_g)
    return u, grad_u

if __name__ == '__main__':
    import time

    def run_periodic_solver():
        L = [-1., 1.,-1., 1.]
        # The mesh
        #mesh = UnitSquareMesh(10,10)#RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
        mesh = RectangleMesh(Point(L[0],L[2]), Point(L[1],L[3]), 20, 20)
        class Source(Expression):
            def eval(self, values, x):
                values[0] = sin(2.0*DOLFIN_PI*x[0]) #+ sin(2.0*DOLFIN_PI*x[1])

        f = Source(degree=2)
        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(SubDomain):

            def __init__(self, L):
                dolfin.SubDomain.__init__(self)
                self.Lx_left = L[0]
                self.Lx_right = L[1]
                self.Ly_left = L[2]
                self.Ly_right = L[3]
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
                return bool((near(x[0], self.Lx_left) or near(x[1], self.Ly_left)) and
                       (not((near(x[0], self.Lx_left) and near(x[1], self.Ly_right)) or
                            (near(x[0], self.Lx_right) and near(x[1], self.Ly_left)))) and on_boundary)

            def map(self, x, y):
                if near(x[0],  self.Lx_right) and near(x[1], self.Ly_right):
                    y[0] = x[0] - (self.Lx_right - self.Lx_left)
                    y[1] = x[1] - (self.Ly_right - self.Ly_left)
                elif near(x[0],  self.Lx_right):
                    y[0] = x[0] - (self.Lx_right - self.Lx_left)
                    y[1] = x[1]
                else:   # near(x[1], 1)
                    y[0] = x[0]
                    y[1] = x[1] - (self.Ly_right - self.Ly_left)

        # Create boundary and finite element
        PBC = PeriodicBoundary(L)
        V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
        V_g = VectorFunctionSpace(mesh, 'DG', 0, constrained_domain=PBC)

        #ff = project(f, V)
        phi, E = periodic_solver(f, mesh, V, V_g)

        #plot(ff,  interactive=True)
        plot(phi, interactive=True)
        plot(E, interactive=True)
        grad_u_x, grad_u_y = E.split(deepcopy=True)  # extract components

        u_array = phi.vector().array()
        grad_u_x_array = grad_u_x.vector().array()
        grad_u_y_array = grad_u_y.vector().array()
        coor = mesh.coordinates()
        for i in range(len(u_array)):
            x, y = coor[i]
            print 'Node (%.3f,%.3f): u = %.4f (%9.2e), '\
                  'grad(u)_x = %.4f  (%9.2e), grad(u)_y = %.4f  (%9.2e)' % \
                  (x, y, u_array[i], (-1.*sin(2.0*DOLFIN_PI*x))/(4.0*DOLFIN_PI*DOLFIN_PI) - u_array[i],
                   grad_u_x_array[i], cos(2.0*DOLFIN_PI*x)/(2.0*DOLFIN_PI) - grad_u_x_array[i],
                   grad_u_y_array[i], 0 - grad_u_y_array[i])
                #   (x, y, u_array[i], (sin(4.0*DOLFIN_PI*x) + sin(4.0*DOLFIN_PI*y))/(16.0*DOLFIN_PI*DOLFIN_PI) - u_array[i],
                #    grad_u_x_array[i], cos(4.0*DOLFIN_PI*x)/(4.0*DOLFIN_PI) - grad_u_x_array[i],
                #    grad_u_y_array[i], cos(4.0*DOLFIN_PI*y)/(4.0*DOLFIN_PI) - grad_u_y_array[i])

    def run_Dirichlet_solver():
        # Source term
        class Source(Expression):
            def eval(self, values, x):
                dx = x[0] - 0.5
                dy = x[1] - 0.5
                values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) \
                            + 1.0*exp(-(dx*dx + dy*dy)/0.02)

        class Source1(Expression):
            def eval(self, values, x):
                values[0] = x[0]*exp(-pow(pow(2*(x[0]-0.5),2)+pow(2*(x[1]-0.5),2),2))

        # Create mesh and finite element
        mesh = UnitSquareMesh(200, 200)
        V = FunctionSpace(mesh, "CG", 1)
        V_g = VectorFunctionSpace(mesh, 'CG', 1)

        # Create Dirichlet boundary condition
        u0 = Constant(0.0)
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u0, boundary)

        f = Source1(degree=1)
        phi, E = dirichlet_solver(f, V, V_g, bc)

        File("data/phi.pvd") << phi
        File("data/partitions.pvd") << CellFunction("size_t", mesh, rank)

    # tic = time.time()
    # run_Dirichlet_solver()
    # toc = time.time()
    # if rank == 0:
    #     seconds = toc-tic
    #     print "Total run time: %d seconds " %seconds
    #     m, s = divmod(seconds, 60)
    #     h, m = divmod(m, 60)
    #     print "Total run time: %d:%02d:%02d" % (h, m, s)
    run_periodic_solver()
