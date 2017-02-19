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

def E_field(phi, VV):
    # Compute gradient
    V = phi.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'CG', degree)

    grad_phi = Function(W)
    grad_phi = project(-1*grad(phi), W)
    e_field = grad_phi#project(grad_phi, W)
    return e_field

def test_periodic_solver():

    solver = PETScKrylovSolver('gmres', 'hypre_amg')#, 'hypre_amg')#'gmres', 'ilu')

    solver.parameters["absolute_tolerance"] = 1e-14
    solver.parameters["relative_tolerance"] = 1e-12
    solver.parameters["maximum_iterations"] = 1000
    #solver.parameters["monitor_convergence"] = True
    solver.parameters["convergence_norm_type"] = "true"
    #solver.parameters['preconditioner']['reuse'] = True
    #solver.parameters['preconditioner']['structure'] = 'same'
    #for item in solver.parameters.items(): print(item)
    solver.set_reuse_preconditioner(True)

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
            phi = periodic_solver(f, V, solver)

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
    d = 2              # Space dimension
    l1 = 0.            # Start position x-axis
    l2 = 2.*np.pi      # End position x-axis
    w1 = 0.            # Start position y-axis
    w2 = 2.*np.pi      # End position y-axis
    h1 = 0.            # Start position z-axis
    h2 = 2.*np.pi      # End position z-axis
    mesh = Mesh("mesh/circle.xml")
    #-------------------------------------------------------------------------------
    #                       Mark facets
    #-------------------------------------------------------------------------------
    facet_f = FacetFunction('size_t', mesh, 0)
    DomainBoundary().mark(facet_f, 1)

    boundary_l1 = 'near((x[0]-l1), 0, tol)'
    boundary_l2 = 'near((x[0]-l2), 0, tol)'
    boundary_w1 = 'near((x[1]-w1), 0, tol)'
    boundary_w2 = 'near((x[1]-w2), 0, tol)'
    boundary_l1 = CompiledSubDomain(boundary_l1, l1=l1, tol=1E-8)
    boundary_l2 = CompiledSubDomain(boundary_l2, l2=l2, tol=1E-8)
    boundary_w1 = CompiledSubDomain(boundary_w1, w1=w1, tol=1E-8)
    boundary_w2 = CompiledSubDomain(boundary_w2, w2=w2, tol=1E-8)
    boundary_l1.mark(facet_f, 2)
    boundary_l2.mark(facet_f, 3)
    boundary_w1.mark(facet_f, 4)
    boundary_w2.mark(facet_f, 5)

    vd = np.array([1.5, 0., 0.])  # Drift velocity of the plasma particles

    B0 = np.array([0., 0., 0.01])     # Uniform background magnetic field
    B0_2 = np.linalg.norm(B0)         # Norm of the B-field

    E0 = -B0_2*np.cross(vd, B0)       # Induced background electric field

    V = FunctionSpace(mesh, "CG", 1)
    VV = VectorFunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, 'DG', 0)

    phi_0 = 'x[0]*Ex + x[1]*Ey'
    phi_l1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
    phi_l2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
    phi_w1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
    phi_w2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])

    bc0 = DirichletBC(V, phi_l1, boundary_l1)
    bc1 = DirichletBC(V, phi_l2, boundary_l2)
    bc2 = DirichletBC(V, phi_w1, boundary_w1)
    bc3 = DirichletBC(V, phi_w2, boundary_w2)

    bcs_Dirichlet = [bc0, bc1, bc2, bc3]
    c = Constant(0.0)
    bc_object = DirichletBC(V, c, facet_f, 1)
    f = Function(V)
    f = Constant(0.0)
    phi = dirichlet_solver(f, V, bcs_Dirichlet, bc_object)
    E = E_field(phi, W)

    coor = phi.function_space().mesh().coordinates()
    phi_vert = phi.compute_vertex_values()
    for i, value in enumerate(coor):
        print(i, "  ", value, "  ", phi_vert[i])

    plot(phi, interactive=True)

def test_D_solver():
    divs = [[3,3], [5,5], [7,7], [10,10]]
    L = [[0., 0., 1., 1.]]
    print_norm = True
    print_max_error = True

    mesh_type = ["UnitHyperCube"]#, "HyperCube"]
    tol = 1E-10
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=4)
    E_ex = Expression('-2*x[0]', degree=4)
    E_ey = Expression('-4*x[1]', degree=4)
    f = Constant(-6.0)

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        h = []
        E1 = []
        E2 = []
        E3 = []
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")
            if i == 0:
                mesh = UnitHyperCube(divisions)
            elif i == 1:
                mesh = HyperCube(L[0], divisions)
            bc, V, VV, V_g = dirichlet_bcs(u_D, mesh)
            phi = dirichlet_solver(f, V, bc)

            E = E_field(phi, V_g)
            E_x, E_y = E.split(deepcopy=True)
            if print_norm:
                E1.append(errornorm(u_D, phi, "l2"))
                E2.append(errornorm(E_ex, E_x, "l2"))
                E3.append(errornorm(E_ey, E_y, "l2"))
                h.append(1./divisions[0])
            if print_max_error:
                coor = phi.function_space().mesh().coordinates()
                error = np.zeros(len(coor))
                for i, value in enumerate(E_x.compute_vertex_values()):
                    error[i] = np.abs(value - E_ex(*coor[i]))
                print("Max error: ", error.max())
                print("Min error: ", error.min())
        if print_norm:
            from math import log as ln
            for i in range(1, len(E1)):
                r1 = ln(E1[i]/E1[i-1])/ln(h[i]/h[i-1])
                print("h =%10.2E E1 =%10.2E r1 =%.2f" %(h[i], E1[i], r1))
            for i in range(1, len(E2)):
                r2 = ln(E2[i]/E2[i-1])/ln(h[i]/h[i-1])
                print("h =%10.2E E2 =%10.2E r2 =%.2f" %(h[i], E2[i], r2))
            for i in range(1,len(E3)):
                r3 = ln(E3[i]/E3[i-1])/ln(h[i]/h[i-1])
                print("h =%10.2E E3 =%10.2E r3 =%.2f" %(h[i], E3[i], r3))


def test_p_solver():

    solver = PETScKrylovSolver('gmres', 'hypre_amg')#, 'hypre_amg')#'gmres', 'ilu')

    solver.parameters["absolute_tolerance"] = 1e-14
    solver.parameters["relative_tolerance"] = 1e-12
    solver.parameters["maximum_iterations"] = 1000
    #solver.parameters["monitor_convergence"] = True
    solver.parameters["convergence_norm_type"] = "true"
    #solver.parameters['preconditioner']['reuse'] = True
    #solver.parameters['preconditioner']['structure'] = 'same'
    #for item in solver.parameters.items(): print(item)
    solver.set_reuse_preconditioner(True)

    divs = [[3,3], [5,5], [10,10], [11,11], [12,12], [13,13], [15,15]]
    l_unit = [[0., 0., 1., 1.]]
    l_hyper = [[-1., -1, 1., 1.]]
    print_norm = False
    print_max_error = True

    mesh_type = ["UnitHyperCube"]#, "HyperCube"]
    tol = 1E-10

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        h = []
        E1 = []
        E2 = []
        E3 = []
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

                class Exact_Ex(Expression):
                    def eval(self, values, x):
                        values[0] = -2.0*DOLFIN_PI*cos(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])
                class Exact_Ey(Expression):
                    def eval(self, values, x):
                        values[0] = -2.0*DOLFIN_PI*sin(2.0*DOLFIN_PI*x[0])*cos(2.0*DOLFIN_PI*x[1])

            f = Source(degree=4)
            phi_e = Exact(degree=4)
            E_ex = Exact_Ex(degree=4)
            E_ey = Exact_Ey(degree=4)

            phi = periodic_solver(f, V, solver)
            E = E_field(phi, V_g)
            E_x, E_y = E.split(deepcopy=True)

            # plot(E_x, title = 'Ex', interactive=True)
            # plot(E_x, title = 'E_ex', interactive=True)

            if print_norm:
                E1.append(errornorm(phi_e, phi, "l2"))
                E2.append(errornorm(E_ex, E_x, "l2"))
                E3.append(errornorm(E_ey, E_y, "l2"))
                h.append(1./divisions[0])
            if print_max_error:
                coor = phi.function_space().mesh().coordinates()
                error = np.zeros(len(coor))
                for i, value in enumerate(E_x.compute_vertex_values()):
                    error[i] = np.abs(value - E_ex(*coor[i]))
                print("Max error: ", error.max())
                print("Min error: ", error.min())
        if print_norm:
            from math import log as ln
            for i in range(1, len(E1)):
                r1 = ln(E1[i]/E1[i-1])/ln(h[i]/h[i-1])
                print("h =%10.2E E1 =%10.2E r1 =%.2f" %(h[i], E1[i], r1))
            for i in range(1, len(E2)):
                r2 = ln(E2[i]/E2[i-1])/ln(h[i]/h[i-1])
                print("h =%10.2E E2 =%10.2E r2 =%.2f" %(h[i], E2[i], r2))
            for i in range(1,len(E3)):
                r3 = ln(E3[i]/E3[i-1])/ln(h[i]/h[i-1])
                print("h =%10.2E E3 =%10.2E r3 =%.2f" %(h[i], E3[i], r3))

def test_periodic_object_solver():

    solver = PETScKrylovSolver('gmres', 'hypre_amg')#, 'hypre_amg')#'gmres', 'ilu')

    solver.parameters["absolute_tolerance"] = 1e-14
    solver.parameters["relative_tolerance"] = 1e-12
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["monitor_convergence"] = True
    solver.parameters["convergence_norm_type"] = "true"
    #solver.parameters['preconditioner']['reuse'] = True
    #solver.parameters['preconditioner']['structure'] = 'same'
    #for item in solver.parameters.items(): print(item)
    solver.set_reuse_preconditioner(True)

    l1 = 0.
    l2 = 2.*np.pi
    w1 = 0.
    w2 = 2.*np.pi

    L = [l1, w1, l2, w2]
    meshes = [Mesh("mesh/circle.xml")]#,Mesh("mesh4.xml"),Mesh("mesh5.xml")]
    h = []
    E = []
    vol = l2*w2
    for i in range(len(meshes)):
        mesh = meshes[i]

        from pylab import show, triplot
        coords = mesh.coordinates()
        triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
        show()
        n_cells = mesh.num_cells()
        a_averg = vol/n_cells
        l_averg = np.sqrt(2.*a_averg)

        facet_f = FacetFunction('size_t', mesh, 0)
        DomainBoundary().mark(facet_f, 1)
        square_boundary = 'near(x[0]*(x[0]-l2), 0, tol) || near(x[1]*(x[1]-w2), 0, tol)'
        square_boundary = CompiledSubDomain(square_boundary, l2=l2, w2=w2, tol=1E-8)
        square_boundary.mark(facet_f, 2)
        plot(facet_f, interactive=True)
        #sys.exit()
        V, VV, W = periodic_bcs(mesh, L)
        class Source(Expression):
            def eval(self, values, x):
                values[0] = 2.*sin(x[0])*sin(x[1])

        class RhoBCs(Expression):
            def eval(self, values, x):
                tol = 1E-8
                if near((x[0]-l2/2.)*(x[0]-l2/2.)+(x[1]-w2/2.)*(x[1]-w2/2.), 0.25, tol):
                    values[0] = sin(x[0])*sin(x[1])
                else:
                    values[0] = 0.

        class Exact(Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])*sin(x[1])

        f = RhoBCs(degree=4)
        rho = Source(degree=4)
        phi_e = Exact(degree=4)

        bc = DirichletBC(V, f, facet_f, 1)
        #print bc.get_boundary_values()
        phi = periodic_solver(rho, V, solver, bc)

        # E.append(errornorm(phi_e, phi, "l2"))
        # h.append(l_averg)

    # from math import log as ln
    # for i in range(1, len(E)):
    #     r = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
    #     print("h =%10.2E E =%10.2E r =%.2f" %(h[i], E[i], r))

    plot(phi, interactive=True)

if __name__ == '__main__':
    import time
    from mesh_types import *
    from boundary_conditions import *

    #test_periodic_solver()
    test_dirichlet_solver()

    # test_p_solver()
    #test_D_solver()

    test_periodic_object_solver()
