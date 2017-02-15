# __authors__ = ('Mikael Mortensen <mikaem@math.uio.no>',
#                'Miroslav Kuchta <mirok@math.uio.no>')
# __date__ = '2014-19-11'
# __copyright__ = 'Copyright (C) 2011' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
'''
This module contains functionality for Lagrangian tracking of particles with
DOLFIN
'''
from __future__ import print_function
import dolfin as df
import numpy as np
import copy
from mpi4py import MPI as pyMPI
from collections import defaultdict
from itertools import product
from particle_distribution import *
#from particle_injection import inject_particles
import sys

# Disable printing
__DEBUG__ = False

comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Particle:
    __slots__ = ['position', 'velocity', 'properties']
    'Lagrangian particle with position, velocity and some other passive properties.'
    def __init__(self, x, v):
        self.position = x
        self.velocity = v
        self.properties = {}

    def send(self, dest):
        'Send particle to dest.'
        comm.Send(self.position, dest=dest)
        comm.Send(self.velocity, dest=dest)
        comm.send(self.properties, dest=dest)

    def recv(self, source):
        'Receive info of a new particle sent from source.'
        comm.Recv(self.position, source=source)
        comm.Recv(self.velocity, source=source)
        self.properties = comm.recv(source=source)


class CellWithParticles(df.Cell):
    'Dolfin cell with list of particles that it contains.'
    def __init__(self, mesh, cell_id, particle, velocity):
        # Initialize parent -- create Cell with id on mesh
        df.Cell.__init__(self, mesh, cell_id)
        # Make an empty list of particles that I carry
        self.particles = []
        self.__add__(particle, velocity)
        # Make the cell aware of its neighbors; neighbor cells are cells
        # connected to this one by vertices
        tdim = mesh.topology().dim()

        neighbors = sum((vertex.entities(tdim).tolist() for vertex in df.vertices(self)), [])
        neighbors = set(neighbors) - set([cell_id])   # Remove self
        self.neighbors = map(lambda neighbor_index: df.Cell(mesh, neighbor_index),
                         neighbors)

    def __add__(self, particle, velocity):
        'Add single particle to cell.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
            self.particles.append(particle)
            return self
        else:
            return self.__add__(Particle(particle, velocity), velocity)

    def __len__(self):
        'Number of particles in cell.'
        return len(self.particles)


class CellParticleMap(dict):
    'Dictionary of cells with particles.'
    def __add__(self, ins):
        '''
        Add ins to map:
            ins is either (mesh, cell_id, particle, velocity) or
                          (mesh, cell_id, particle, velocity, particle_properties)
        '''
        assert isinstance(ins, tuple) and len(ins) in (4, 5)
        # If the cell_id is in map add the particle
        if ins[1] in self:
            #self[ins[1]] += ins[2]
            self[ins[1]].__add__(ins[2], ins[3])
        # Other wise create new cell
        else:
            self[ins[1]] = CellWithParticles(ins[0], ins[1], ins[2], ins[3])
        # With particle_properties, update properties of the last added particle
        if len(ins) == 5:
            self[ins[1]].particles[-1].properties.update(ins[4])

        return self

    def pop(self, cell_id, i):
        'Remove i-th particle from the list of particles in cell with cell_id.'
        # Note that we don't check for cell_id being a key or cell containg
        # at least i particles.
        particle = self[cell_id].particles.pop(i)

        # If the cell is empty remove it from map
        if len(self[cell_id]) == 0:
            del self[cell_id]

        return particle

    def total_number_of_particles(self):
        'Total number of particles in all cells of the map.'
        return sum(map(len, self.itervalues()))


class LagrangianParticles:
    'Particles moved by the velocity field in V.'
    def __init__(self, V, object_type, object_info, B_field, count_e, count_i,
                mu_e, mu_i, sigma_e, sigma_i, w, q_e, q_i, m_e, m_i, dt):
        self.__debug = __DEBUG__

        self.V = V
        self.mesh = V.mesh()
        # self.mesh.init(2, 2)  # Cell-cell connectivity for neighbors of cell
        self.tree = self.mesh.bounding_box_tree()  # Tree for isection comput.

        # The object
        self.object_type = object_type
        self.object_info = object_info
        self.B_field = B_field
        self.count_e = count_e
        self.count_i = count_i
        self.mu_e = mu_e
        self.mu_i = mu_i
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.w = w
        self.q_e = q_e
        self.q_i = q_i
        self.m_e = m_e
        self.m_i = m_i
        self.dt = dt
        # Allocate some variables used to look up the velocity
        # Velocity is computed as U_i*basis_i where i is the dimension of
        # element function space, U are coefficients and basis_i are element
        # function space basis functions. For interpolation in cell it is
        # advantageous to compute the resctriction once for cell and only
        # update basis_i(x) depending on x, i.e. particle where we make
        # interpolation. This updaea mounts to computing the basis matrix
        self.dim = self.mesh.topology().dim()
        self.mesh.init(0, self.dim)  # Vertex-to-cell connectivity for neighb. comput

        self.element = V.dolfin_element()
        self.num_tensor_entries = 1
        for i in range(self.element.value_rank()):
            self.num_tensor_entries *= self.element.value_dimension(i)
        # For VectorFunctionSpace CG1 this is 3
        self.coefficients = np.zeros(self.element.space_dimension())
        # For VectorFunctionSpace CG1 this is 3x3
        self.basis_matrix = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))

        # Allocate a dictionary to hold all particles
        self.particle_map = CellParticleMap()

        # Allocate some MPI stuff
        self.num_processes = comm.Get_size()
        self.myrank = comm.Get_rank()
        self.all_processes = range(self.num_processes)
        self.other_processes = range(self.num_processes)
        self.other_processes.remove(self.myrank)
        self.my_escaped_particles = np.zeros(1, dtype='I')
        self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
        # Dummy particle for receiving/sending at [0, 0, ...]
        self.particle0 = Particle(np.zeros(self.mesh.geometry().dim()),
                                  np.zeros(self.mesh.geometry().dim()))


    def __iter__(self):
        '''Iterate over all particles.'''
        for cwp in self.particle_map.itervalues():
            for particle in cwp.particles:
                yield particle


    def add_particles(self, list_of_particles, list_of_velocities, properties_d=None):
        '''Add particles and search for their home on all processors.
           Note that list_of_particles must be same on all processes. Further
           every len(properties[property]) must equal len(list_of_particles).
        '''
        if properties_d is not None:
            n = len(list_of_particles)
            assert all(len(sub_list) == n
                       for sub_list in properties_d.itervalues())
            # Dictionary that will be used to feed properties of single
            # particles
            properties = properties_d.keys()
            particle_properties = dict((key, 0) for key in properties)
            has_properties = True
        else:
            has_properties = False

        pmap = self.particle_map
        my_found = np.zeros(len(list_of_particles), 'I')
        all_found = np.zeros(len(list_of_particles), 'I')
        for i, particle in enumerate(list_of_particles):
            c = self.locate(particle, list_of_velocities[i])
            if not (c == -1 or c == __UINT32_MAX__):
                my_found[i] = True
                if not has_properties:
                    pmap += self.mesh, c, particle, list_of_velocities[i]
                else:
                    # Get values of properties for this particle
                    for key in properties:
                        particle_properties[key] = properties_d[key][i]
                    pmap += self.mesh, c, particle, list_of_velocities[i], particle_properties
        # All particles must be found on some process
        comm.Reduce(my_found, all_found, root=0)

        if self.myrank == 0:
            missing = np.where(all_found == 0)[0]
            n_missing = len(missing)

            assert n_missing == 0,\
                '%d particles are not located in mesh' % n_missing

            # Print particle info
            if self.__debug:
                for i in missing:
                    print('Missing', list_of_particles[i].position)

                n_duplicit = len(np.where(all_found > 1)[0])
                print('There are %d duplicit particles' % n_duplicit)

    def barycentric_Interpolation(self, v, p):
        eps = 1e-14
        d = v[0]*(v[3]-v[5]) - \
            v[2]*(v[1]-v[5]) + \
            v[4]*(v[1]-v[3])
        if abs(d) > eps:
            dx = p[0]*(v[3]-v[5]) - \
                 v[2]*(p[1]-v[5]) + \
                 v[4]*(p[1]-v[3])
            dy = v[0]*(p[1]-v[5]) - \
                 p[0]*(v[1]-v[5]) + \
                 v[4]*(v[1]-p[1])
            dz = v[0]*(v[3]-p[1]) - \
                 v[2]*(v[1]-p[1]) + \
                 p[0]*(v[1]-v[3])
            return (dx/d, dy/d, dz/d)

        raise ValueError('Singular system, no solution.')

    def potential_energy(self, phi):
        e_p = 0.0
        for cwp in self.particle_map.itervalues():
            phi_coefficients = np.zeros(phi.function_space().dolfin_element().space_dimension())
            phi_basis_matrix = np.zeros(phi.function_space().dolfin_element().space_dimension())
            phi.restrict(phi_coefficients,
                       phi.function_space().dolfin_element(),
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            for particle in cwp.particles:
                x = particle.position
                # Compute velocity at position x
                phi.function_space().dolfin_element().evaluate_basis_all(phi_basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())
                e_p += 0.5*particle.properties['q']*np.dot(phi_coefficients, phi_basis_matrix)
        return e_p

    def kinetic_energy(self):
        e_k = 0.0
        for cwp in self.particle_map.itervalues():
            for particle in cwp.particles:
                e_k += 0.5*particle.properties['m']*np.sum(np.asarray(particle.velocity)**2)
        return e_k

    def charge_density(self, f, object_vertices = None):
        'Particle charge weigthed at nodes'
        v2d = df.vertex_to_dof_map(f.function_space())

        for cwp in self.particle_map.itervalues():
            f_coefficients = np.zeros(f.function_space().dolfin_element().space_dimension())
            f_basis_matrix = np.zeros(f.function_space().dolfin_element().space_dimension())
            f.restrict(f_coefficients,
                       f.function_space().dolfin_element(),
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            c = cwp.entities(0)
            dof = v2d[c]
            for particle in cwp.particles:
                x = particle.position
                # Compute velocity at position x
                f.function_space().dolfin_element().evaluate_basis_all(f_basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())
                f_coefficients += particle.properties['q']*f_basis_matrix/cwp.volume()
            f.vector()[dof] = f_coefficients

        q_rho = 0.0
        if not self.object_type == None:
            for cwp in self.particle_map.itervalues():
                c = cwp.entities(0)
                for i in range(len(c)):
                    for j in range(len(object_vertices)):
                        if c[i] == object_vertices[j]:
                            dof = v2d[c[i]]
                            q_rho += f.vector()[dof][0]*cwp.volume()
        return f, q_rho

    def current_density(self, J_e, J_i):
        'Particle charge density times velocity weigthed at nodes'

        for cwp in self.particle_map.itervalues():
            J_e_coefficients = np.zeros(J_e.function_space().dolfin_element().space_dimension())
            J_i_coefficients = np.zeros(J_i.function_space().dolfin_element().space_dimension())
            J_basis_matrix = np.zeros((J_e.function_space().dolfin_element().space_dimension(), self.num_tensor_entries))

            J_e.restrict(J_e_coefficients,
                       J_e.function_space().dolfin_element(),
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            J_i.restrict(J_i_coefficients,
                       J_i.function_space().dolfin_element(),
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            c = cwp.entities(0)
            cell_dof = J_i.function_space().dofmap().cell_dofs(cwp.index())

            for particle in cwp.particles:
                x = particle.position
                u = particle.velocity
                # Compute velocity at position x
                J_e.function_space().dolfin_element().evaluate_basis_all(J_basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())
                if np.sign(particle.properties['q']) == -1.:
                    J_e_coefficients[:3] += (particle.properties['q']/cwp.volume())*u[0]*J_basis_matrix[:3, 0]
                    J_e_coefficients[3:] += (particle.properties['q']/cwp.volume())*u[1]*J_basis_matrix[3:, 1]
                    # J_e_coefficients[:] += (particle.properties['q']/cwp.volume())*J_e_coefficients*np.dot(u, J_basis_matrix.T)[:]
                if np.sign(particle.properties['q']) == 1.:
                    J_i_coefficients[:3] += (particle.properties['q']/cwp.volume())*u[0]*J_basis_matrix[:3, 0]
                    J_i_coefficients[3:] += (particle.properties['q']/cwp.volume())*u[1]*J_basis_matrix[3:, 1]
                    # J_i_coefficients[:] += (particle.properties['q']/cwp.volume())*J_i_coefficients*np.dot(u, J_basis_matrix.T)[:]
            J_e.vector()[cell_dof] = J_e_coefficients/2.0
            J_i.vector()[cell_dof] = J_i_coefficients/2.0
        return J_e, J_i

    def step(self, E, J_e, J_i, t_step, object_charge, dt, B = None):
        'Move particles by leap frog or/and Boris algorithm'
        start = df.Timer('shift')
        e_k = 0.0

        for cwp in self.particle_map.itervalues():
            J_e_coefficients = np.zeros(J_e.function_space().dolfin_element().space_dimension())
            J_i_coefficients = np.zeros(J_i.function_space().dolfin_element().space_dimension())
            J_basis_matrix = np.zeros((J_e.function_space().dolfin_element().space_dimension(), self.num_tensor_entries))
            J_e.restrict(J_e_coefficients,
                       J_e.function_space().dolfin_element(),
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            J_i.restrict(J_i_coefficients,
                       J_i.function_space().dolfin_element(),
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            # Restrict once per cell
            E.restrict(self.coefficients,
                       self.element,
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)
            c = cwp.entities(0)
            cell_dof = J_i.function_space().dofmap().cell_dofs(cwp.index())

            for particle in cwp.particles: #i,particle in enumerate(cwp.particles)
                x = particle.position
                u = particle.velocity

                J_e.function_space().dolfin_element().evaluate_basis_all(J_basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())

                u_old = u[:]

                if t_step == 0:
                    dt /= 2.

                if not self.B_field:
                    # leap frog step
                    u[:] = u[:] + dt*(particle.properties['q']/particle.properties['m'])*np.dot(self.coefficients, self.basis_matrix)[:]
                else:
                    # Boris step
                    assert self.dim == 3
                    t = np.tan(particle.properties['q']*dt/(2.*particle.properties['m'])*B)
                    s = 2*t/(1+t[0]**2+t[1]**2+t[2]**2)
                    v_minus = u + 0.5*dt*(particle.properties['q']/particle.properties['m'])*np.dot(self.coefficients, self.basis_matrix)
                    v_minus_cross_t = np.cross(v_minus, t)
                    v_prime = v_minus + v_minus_cross_t
                    v_prime_cross_s = np.cross(v_prime, s)
                    v_plus = v_minus + v_prime_cross_s
                    u[:] = v_plus[:] + 0.5*dt*(particle.properties['q']/particle.properties['m'])*np.dot(self.coefficients, self.basis_matrix)[:]

                # Kinetic energy
                e_k += (1./2.)*particle.properties['m']*np.dot(u_old[:], u[:])

                if np.sign(particle.properties['q']) == -1.:
                    J_e_coefficients[:3] += (particle.properties['q']/cwp.volume())*u[0]*J_basis_matrix[:3, 0]
                    J_e_coefficients[3:] += (particle.properties['q']/cwp.volume())*u[1]*J_basis_matrix[3:, 1]
                    # J_e_coefficients[:] += (particle.properties['q']/cwp.volume())*J_e_coefficients*np.dot(u, J_basis_matrix.T)[:]
                if np.sign(particle.properties['q']) == 1.:
                    J_i_coefficients[:3] += (particle.properties['q']/cwp.volume())*u[0]*J_basis_matrix[:3, 0]
                    J_i_coefficients[3:] += (particle.properties['q']/cwp.volume())*u[1]*J_basis_matrix[3:, 1]
                    # J_i_coefficients[:] += (particle.properties['q']/cwp.volume())*J_i_coefficients*np.dot(u, J_basis_matrix.T)[:]
                x[:] = x[:] + dt*u[:]
            J_e.vector()[cell_dof] = J_e_coefficients/2.0
            J_i.vector()[cell_dof] = J_i_coefficients/2.0
        # Recompute the map
        stop_shift = start.stop()
        start = df.Timer('relocate')
        info = self.relocate(object_charge)
        stop_reloc = start.stop()
        # We return computation time per process
        return (stop_shift, stop_reloc, e_k, info, J_e, J_i)

    def relocate(self, object_charge):
        # Relocate particles on cells and processors
        p_map = self.particle_map
        # Map such that map[old_cell] = [(new_cell, particle_id), ...]
        # Ie new destination of particles formerly in old_cell
        new_cell_map = defaultdict(list)
        particels_inside_object = []

        for cwp in p_map.itervalues():
            for i, particle in enumerate(cwp.particles):
                point = df.Point(*particle.position)
                # Search only if particle moved outside original cell
                if not cwp.contains(point):
                    found = False
                    # Check neighbor cells
                    # for neighbor in df.cells(cwp):
                    for neighbor in cwp.neighbors:
                        if neighbor.contains(point):
                            new_cell_id = neighbor.index()
                            found = True
                            break
                    # Do a completely new search if not found by now
                    if not found:
                        new_cell_id = self.locate(particle, particle.velocity)
                    # Record to map
                    new_cell_map[cwp.index()].append((new_cell_id, i))

        # Rebuild locally the particles that end up on the process. Some
        # have cell_id == -1, i.e. they are on other process
        list_of_escaped_particles = []
        list_of_escaped_particles_velocity = []
        for old_cell_id, new_data in new_cell_map.iteritems():
            # We iterate in reverse becasue normal order would remove some
            # particle this shifts the whole list!
            for (new_cell_id, i) in sorted(new_data,
                                           key=lambda t: t[1],
                                           reverse=True):
                particle = p_map.pop(old_cell_id, i)

                if new_cell_id == -1 or new_cell_id == __UINT32_MAX__ :
                    list_of_escaped_particles.append(particle)
                    list_of_escaped_particles_velocity.append(particle.velocity)
                else:
                    p_map += self.mesh, new_cell_id, particle, particle.velocity

        # Create a list of how many particles escapes from each processor
        self.my_escaped_particles[0] = len(list_of_escaped_particles)
        # Make all processes aware of the number of escapees
        comm.Allgather(self.my_escaped_particles, self.tot_escaped_particles)

        # Send particles to root
        if self.myrank != 0:
            for particle in list_of_escaped_particles:
                particle.send(0)

        # Receive the particles escaping from other processors
        if self.myrank == 0:
            for proc in self.other_processes:
                for i in range(self.tot_escaped_particles[proc]):
                    self.particle0.recv(proc)
                    list_of_escaped_particles.append(copy.deepcopy(self.particle0))
                    list_of_escaped_particles_velocity.append(copy.deepcopy(self.particle0.velocity))

        # What to do with escaped particles
        particles_inside_object = []
        particles_outside_domain = []
        for i in range(len(list_of_escaped_particles)):
            p = list_of_escaped_particles[i]
            x = p.position
            q = p.properties['q']
            d = len(x)

            for dim in range(len(x)):
                l_min = self.mesh.coordinates()[:,dim].min()
                l_max = self.mesh.coordinates()[:,dim].max()
                l = l_max - l_min

                if not self.B_field:# periodic boundary conditions
                    if x[dim] < l_min or x[dim] > l_max:
                        x[dim] = (x[dim]+abs(l_min))%l + l_min
                else:# Dirichlet boundary conditions
                    if x[dim] < l_min or x[dim] > l_max:
                        particles_outside_domain.append(i)
                    # add particles

            # If the particle hits the object remove it
            if self.object_type == 'spherical_object':
                if d == 2:
                    s0 = [self.object_info[0], self.object_info[1]]
                    r0 = self.object_info[2]
                if d == 3:
                    s0 = [self.object_info[0], self.object_info[1], self.object_info[2]]
                    r0 = self.object_info[3]
                if np.dot(x-s0, x-s0) < r0**2:
                    particles_inside_object.append(i)
            if self.object_type == 'cylindrical_object':
                s0 = [self.object_info[0], self.object_info[1]]
                r0 = self.object_info[2]
                h0 = self.object_info[3]
                h2 = self.mesh.coordinates()[:,2].max()
                z0 = (h2-h0)/2.      # Bottom point of cylinder
                z1 = (h2+h0)/2.      # Top point of cylinder
                if (x[2] > z0 and x[2] < z1 and np.dot(x[:2]-s0, x[:2]-s0) < r0**2):
                    particles_inside_object.append(i)

        # particles_inside_object = set(particles_inside_object)
        # particles_inside_object = list(particles_inside_object)
        particles_outside_domain = set(particles_outside_domain)
        particles_outside_domain  = list(particles_outside_domain)

        if not self.object_type == None:
            # Remove particles inside the object and accumulate the charge
            for i in reversed(particles_inside_object):
                p = list_of_escaped_particles[i]
                object_charge += p.properties['q']
                list_of_escaped_particles.remove(p)
        if self.B_field:
            # Inject new particles at the boundaries
            L = [0]*2*self.dim
            for i in range(self.dim):
                L[i] = self.mesh.coordinates()[:,i].min()
                L[i+self.dim] = self.mesh.coordinates()[:,i].max()

            pos, vel, p_properties, n_electrons, n_ions = inject_particles(L,
                                                            self.count_e,
                                                            self.count_i,
                                                            self.mu_e,
                                                            self.mu_i,
                                                            self.sigma_e,
                                                            self.sigma_i,
                                                            self.dt)

            self.add_particles(pos, vel, p_properties)
            print("Injected ", n_electrons, " electrons.")
            print("Injected ", n_ions, " ions.")
            # Remove particles exiting domain
            for i in reversed(particles_outside_domain):
                p = list_of_escaped_particles[i]
                list_of_escaped_particles.remove(p)
        # Put all travelling particles on all processes, then perform new search
        travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
        travelling_particles_velocity = comm.bcast(list_of_escaped_particles_velocity, root=0)
        self.add_particles(travelling_particles, travelling_particles_velocity)
        return object_charge

    def total_number_of_particles(self):
        'Return number of particles in total and on process.'
        num_p = self.particle_map.total_number_of_particles()
        tot_p = comm.allreduce(num_p)
        return (tot_p, num_p)

    def locate(self, particle, velocity):
        'Find mesh cell that contains particle.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
            # Convert particle to point
            point = df.Point(*particle.position)
            return self.tree.compute_first_entity_collision(point)
        else:
            return self.locate(Particle(particle, velocity), velocity)

    def scatter(self, fig, skip=1):
        'Scatter plot of all particles on process 0'
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        ax = fig.gca()

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves should send to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)
        else:
            # Receive on master
            received = defaultdict(list)
            received[0] = [copy.copy(p.position)
                           for cwp in p_map.itervalues()
                           for p in cwp.particles]
            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    received[proc].append(copy.copy(self.particle0.position))

            cmap = cmx.get_cmap('jet')
            cnorm = colors.Normalize(vmin=0, vmax=self.num_processes)
            scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

            for proc in received:
                # Plot only if there is something to plot
                particles = received[proc]
                if len(particles) > 0:
                    xy = np.array(particles)

                    ax.scatter(xy[::skip, 0], xy[::skip, 1],
                               label='%d' % proc,
                               c=scalarMap.to_rgba(proc),
                               edgecolor='none')
            ax.legend(loc='best')
            ax.axis([0, 1, 0, 1])

    def scatter_new(self, fig, skip=1):
        'Scatter plot of all particles on process 0'
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        ax = fig.gca()

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves should send to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)
        else:
            # Receive on master
            received_ions = defaultdict(list)
            received_electrons = defaultdict(list)
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    if np.sign(p.properties['q']) == 1.:
                        received_ions[0].append(copy.copy(p.position))
                    elif np.sign(p.properties['q']) == -1.:
                        received_electrons[0].append(copy.copy(p.position))
            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    if np.sing(self.particle0.properties['q']) == 1.:
                        received_ions[proc].append(copy.copy(self.particle0.position))
                    elif np.sign(self.particle0.properties['q']) == -1.:
                        received_electrons[proc].append(copy.copy(self.particle0.position))

            cmap = cmx.get_cmap('viridis')
            cnorm = colors.Normalize(vmin=0, vmax=self.num_processes)
            scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
            l_min = self.mesh.coordinates()[:,0].min()
            l_max = self.mesh.coordinates()[:,1].max()
            # theta goes from 0 to 2pi
            theta = np.linspace(0, 2*np.pi, 100)

            # the radius of the circle
            r = np.sqrt(0.25)

            # compute x1 and x2
            x1 = np.pi + r*np.cos(theta)
            x2 = np.pi + r*np.sin(theta)

            ax.plot(x1, x2, c='k', linewidth=3)
            ax.set_aspect(1)

            for proc in received_ions:
                # Plot only if there is something to plot
                ions = received_ions[proc]
                electrons = received_electrons[proc]
                if (len(ions) > 0 or len(electrons) > 0):
                    xy_ions = np.array(ions)
                    xy_electrons = np.array(electrons)
                    ax.scatter(xy_ions[::skip, 0], xy_ions[::skip, 1],
                               label='ions',
                               marker='o',
                               c='r',
                               edgecolor='none')
                    ax.scatter(xy_electrons[::skip, 0], xy_electrons[::skip, 1],
                               label='electrons',
                               marker = 'o',
                               c='b',
                               edgecolor='none')
            ax.legend(loc='best')
            ax.axis([l_min, l_max, l_min, l_max])

    def particle_distribution(self):
        # Psarticle distribution
        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves should send to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)
        else:
            # Receive on master
            received_ions = defaultdict(list)
            received_electrons = defaultdict(list)
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    if np.sign(p.properties['q']) == 1.:
                        received_ions[0].append(copy.copy(p.velocity))
                    elif np.sign(p.properties['q']) == -1.:
                        received_electrons[0].append(copy.copy(p.velocity))
            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    if np.sign(self.particle0.properties['q']) == 1.:
                        received_ions[proc].append(copy.copy(self.particle0.velocity))
                    elif np.sign(self.particle0.properties['q']) == -1.:
                        received_electrons[proc].append(copy.copy(self.particle0.velocity))

            for proc in received_ions:
                # Plot only if there is something to plot
                ions = received_ions[proc]
                electrons = received_electrons[proc]
                if (len(ions) > 0 and len(electrons) > 0):
                    geo_dim = self.mesh.geometry().dim()
                    v_ions = np.array(ions)
                    v_electrons = np.array(electrons)
                    alpha_i = np.std(v_ions, ddof=1)
                    alpha_e = np.std(v_electrons, ddof=1)
                    speed_distribution(v_ions, geo_dim, alpha_i, 1)
                    speed_distribution(v_electrons, geo_dim, alpha_e, 0)

    def bar(self, fig):
        'Bar plot of particle distribution.'
        ax = fig.gca()

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        if self.myrank == 0 and self.num_processes > 1:
            ax.bar(np.array(self.all_processes)-0.25, all_particles, 0.5)
            ax.set_xlabel('proc')
            ax.set_ylabel('number of particles')
            ax.set_xlim(-0.25, max(self.all_processes)+0.25)
            return np.sum(all_particles)
        else:
            return None

    def write_to_file(self, to_file):

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves should send to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)
        else:
            # Receive on master
            received_ions = defaultdict(list)
            received_electrons = defaultdict(list)
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    if np.sign(p.properties['q']) == 1.:
                        received_ions[0].append(copy.copy(p.position))
                    elif np.sign(p.properties['q']) == -1.:
                        received_electrons[0].append(copy.copy(p.position))
            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    if np.sing(self.particle0.properties['q']) == 1.:
                        received_ions[proc].append(copy.copy(self.particle0.position))
                    elif np.sign(self.particle0.properties['q']) == -1.:
                        received_electrons[proc].append(copy.copy(self.particle0.position))

            for proc in received_ions:
                # Plot only if there is something to plot
                ions = received_ions[proc]
                electrons = received_electrons[proc]
                if (len(ions) > 0 and len(electrons) > 0):
                    xy_ions = np.array(ions)
                    xy_electrons = np.array(electrons)
                    d = xy_ions.shape[1]
                    if d == 2:
                        for p1, p2 in map(None,xy_ions, xy_electrons):
                            if not p1 is None:
                                to_file.write("%s %f %f %f\n" %('C', p1[0], p1[1], 0.0))
                            if not p2 is None:
                                to_file.write("%s %f %f %f\n" %('O', p2[0], p2[1], 0.0))
                    elif d == 3:
                        for p1, p2 in map(None,xy_ions, xy_electrons):
                            if not p1 is None:
                                to_file.write("%s %f %f %f\n" %('C', p1[0], p1[1], p1[2]))
                            if not p2 is None:
                                to_file.write("%s %f %f %f\n" %('O', p2[0], p2[1], p2[2]))


# Simple initializers for particle positions

from math import pi, sqrt
from itertools import product

comm = pyMPI.COMM_WORLD


class RandomGenerator(object):
    '''
    Fill object by random points.
    '''
    def __init__(self, domain, rule):
        '''
        Domain specifies bounding box for the shape and is used to generate
        points. The rule filter points of inside the bounding box that are
        axctually inside the shape.
        '''
        assert isinstance(domain, list)
        self.domain = domain
        self.rule = rule
        self.dim = len(domain)
        self.rank = comm.Get_rank()

    def generate(self, N, method='full'):
        'Genererate points.'
        assert len(N) == self.dim
        assert method in ['full', 'tensor']

        if self.rank == 0:
            # Generate random points for all coordinates
            if method == 'full':
                n_points = np.product(N)
                points = np.random.rand(n_points, self.dim)
                for i, (a, b) in enumerate(self.domain):
                    points[:, i] = a + points[:, i]*(b-a)
            # Create points by tensor product of intervals
            else:
                # Values from [0, 1) used to create points between
                # a, b - boundary
                # points in each of the directiosn
                shifts_i = np.array([np.random.rand(n) for n in N])
                # Create candidates for each directions
                points_i = (a+shifts_i[i]*(b-a)
                            for i, (a, b) in enumerate(self.domain))
                # Cartesian product of directions yield n-d points
                points = (np.array(point) for point in product(*points_i))


            # Use rule to see which points are inside
            points_inside = np.array(filter(self.rule, points))
        else:
            points_inside = None

        points_inside = comm.bcast(points_inside, root=0)

        return points_inside

class RandomRectangle(RandomGenerator):
    def __init__(self, ll, ur):
        ax, ay = ll[0], ll[1]
        bx, by = ur[0], ur[1]
        assert ax < bx and ay < by
        RandomGenerator.__init__(self, [[ax, bx], [ay, by]], lambda x: True)


class RandomCircle(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2) < radius
                                 )

class RandomBox(RandomGenerator):
    def __init__(self, ll, ur):
        ax, ay, az = ll[0], ll[1], ll[2]
        bx, by, bz = ur[0], ur[1], ur[2]
        assert ax < bx and ay < by and az < bz
        RandomGenerator.__init__(self, [[ax, bx], [ay, by], [az, bz]], lambda x: True)


class RandomSphere(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius],
                  [center[2]-radius, center[2]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2+
                                                (x[2]-center[2])**2) < radius
                                 )
