# Poisson solver with periodic boundary conditions
#
#     - div grad u(x, y, z) = f(x, y, z)
#
# on the unit square with periodic boundary conditions
import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

def periodic_solver(f, mesh, V, V_g):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    A, b = assemble_system(a, L)
    uh = Function(V)

    # Create Krylov solver
    solver = PETScKrylovSolver("cg")
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
        v = TestFunction(V_g)
        w = TrialFunction(V_g)

        a = inner(w, v)*dx
        L = inner(grad(uh), v)*dx
        A, b = assemble_system(a, L)
        grad_u = Function(V_g)

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
        solver.solve(grad_u.vector(), b)
    else:
        grad_u = project(grad(uh), V_g)

    #grad_u_x, grad_u_y = grad_u.split(deepcopy=True)  # extract component

    return uh, grad_u

def dirichlet_solver(f, V, V_g, bc):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Compute the gradient
    grad_u = Function(V_g)
    grad_u = project(grad(u), V_g)
    return u, grad_u
if __name__ == '__main__':

    def run_periodic_solver():
        # The mesh
        mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)

        class Source(Expression):
            def eval(self, values, x):
                values[0] = sin(4.0*DOLFIN_PI*x[0]) + sin(4.0*DOLFIN_PI*x[1])

        f = Source(degree=1)
        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
                return bool((near(x[0], 0) or near(x[1], 0)) and
                       (not((near(x[0], 0) and near(x[1], 1)) or
                            (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

            def map(self, x, y):
                if near(x[0], 1) and near(x[1], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1] - 1.
                elif near(x[0], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1]
                else:   # near(x[1], 1)
                    y[0] = x[0]
                    y[1] = x[1] - 1.

        # Create boundary and finite element
        PBC = PeriodicBoundary()
        V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
        V_g = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=PBC)

        phi, E = periodic_solver(f, mesh, V, V_g)

        grad_u_x, grad_u_y = E.split(deepcopy=True)  # extract components

        u_array = phi.vector().array()
        grad_u_x_array = grad_u_x.vector().array()
        grad_u_y_array = grad_u_y.vector().array()
        coor = mesh.coordinates()
        for i in range(len(u_array)):
            x, y = coor[i]
            print 'Node (%.3f,%.3f): u = %.4f (%9.2e), '\
                  'grad(u)_x = %.4f  (%9.2e), grad(u)_y = %.4f  (%9.2e)' % \
                  (x, y, u_array[i], (sin(4.0*DOLFIN_PI*x) + sin(4.0*DOLFIN_PI*y))/(16.0*DOLFIN_PI*DOLFIN_PI) - u_array[i],
                   grad_u_x_array[i], cos(4.0*DOLFIN_PI*x)/(4.0*DOLFIN_PI) - grad_u_x_array[i],
                   grad_u_y_array[i], cos(4.0*DOLFIN_PI*y)/(4.0*DOLFIN_PI) - grad_u_y_array[i])

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
        mesh = UnitSquareMesh(32, 32)
        V = FunctionSpace(mesh, "CG", 1)
        V_g = VectorFunctionSpace(mesh, 'CG', 1)

        # Create Dirichlet boundary condition
        u0 = Constant(0.0)
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u0, boundary)

        f = Source1(degree=1)
        phi, E = dirichlet_solver(f, V, V_g, bc)
        plot(phi)
        plot(phi.function_space().mesh())
        plot(E)
        interactive()

    run_Dirichlet_solver()
    #run_periodic_solver()
