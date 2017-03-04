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

def periodic_solver(f, V, solver, bc=None):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx

    if bc == None:
        A, b = assemble_system(a, L)
    else:
        A, b = assemble_system(a, L, bc)
        # bc.apply(A)
        # bc.apply(b)

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

def dirichlet_solver(f, V, bcs, bc_object=None):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    phi = Function(V)
    if bc_object == None:
        solve(a == L, phi, bcs)
    else:
        for i in range(len(bc_object)):
            bcs.append(bc_object[i])

        A = assemble(a)
        b = assemble(L)

        [bc.apply(A) for bc in bcs]
        [bc.apply(b) for bc in bcs]

        solve(A, phi.vector(), b, 'cg', 'hypre_amg')
    return phi

def electric_field(phi, VV):
    # Compute gradient
    V = phi.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'CG', degree)

    grad_phi = Function(W)
    grad_phi = project(-1*grad(phi), W)
    e_field = grad_phi#project(grad_phi, W)
    return e_field
