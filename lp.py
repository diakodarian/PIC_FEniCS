from collections import defaultdict, namedtuple
from itertools import count, ifilter
import dolfin as df
import numpy as np

def add_timer(f):
    def f_timed(*args, **kwargs):
        t = df.Timer()
        ans = f(*args, **kwargs)
        dt = t.stop()
        info('\t%s took %g' % (f.func_name, dt))
        return ans
    return f_timed

lg_count = namedtuple('Local_Global_count', ['lc', 'gc'])
def plen(comm, stuff):
    '''Local and global count of stuff.'''
    lc = len(stuff)
    gc = comm.allreduce(lc)
    return lg_count(lc, gc)

# IDEAs:
# 1) Is search accelerated by bounding boxes? E.g. don't allow particles that
# left to enter the 'ring' search
# 2) The neighb. patch could go acrross process so when c == -1 we would ship to
# other rank also this cell (that particle left) and it could lookup part of
# patch where to loop for [need small dt to be efficient so that search most
# often terminates in the patch
# 3) Figure out the CPU layout and only have CPU talk to its neighbors
class CellWithParticles(df.Cell):
    '''
    Dolfin cell holding a set of particles which are keys in the
    lp_collection dictionary.
    '''
    def __init__(self, lp_collection, cell_id):
        mesh = lp_collection.mesh
        df.Cell.__init__(self, mesh, cell_id)
        # NOTE: the choice for set here is to allow for removing left particles
        # by difference
        self.particles = set([])
        # Make cell aware of its neighbors
        tdim = lp_collection.dim
        neighbors = sum((vertex.entities(tdim).tolist() for vertex in df.vertices(self)), [])
        neighbors = set(neighbors) - set([cell_id])   # Remove self
        self.neighbors = map(lambda neighbor_index: df.Cell(mesh, neighbor_index), neighbors)

    def __add__(self, particle):
        '''Add a particle.'''
        self.particles.add(particle)

    def __len__(self):
        '''Number of particles'''
        return len(self.particles)


class LPCollection(object):
    '''
    Collection of Lagrangian particles. Particle data is stored in a dictionary
    and cells which contain them hold reference to the data.
    '''
    def __init__(self, V, property_layout=None, debug=True):
        # The cell neighbors requires cell-cell connectivity which is here
        # defined as throught cell - vertex - cell connectivity
        mesh = V.mesh()
        self.dim = mesh.geometry().dim()
        assert self.dim == mesh.topology().dim()
        mesh.init(0, self.dim)
        self.mesh = mesh

        # Locating particles in cells done by bbox collisions
        self.tree = mesh.bounding_box_tree()
        self.lim = mesh.topology().size_global(self.dim)

        # Velocity evaluation is done by restriction the function on a cell. For
        # this we prealocate data
        element = V.dolfin_element()

        num_tensor_entries = 1
        for i in range(element.value_rank()): num_tensor_entries *= element.value_dimension(i)

        self.coefficients = np.zeros(element.space_dimension())
        self.basis_matrix = np.zeros((element.space_dimension(), num_tensor_entries))
        self.element = element
        self.num_tensor_entries = num_tensor_entries

        # Particles and cells are stored in dicts and we refer to each by ints.
        # For cell this is the cell index, particles get a ticket from their
        # counter. # NOTE: When particle leaves CPU it is removed from the
        # dictionary in turn particles.keys may develop 'holes'.
        self.particles = {}
        self.cells = {}
        self.ticket = count()

        # Property layout here is the map which maps property name to
        # length of a vector that represents property value. By default
        # particles only store the position
        if property_layout is None: property_layout = []
        property_layout = [('x', self.dim)] + property_layout
        props, sizes = map(list, zip(*property_layout))
        assert all(v > 0 for v in sizes)
        offsets = [0] + sizes
        self.psize = sum(offsets)    # How much to store per particle
        self.offsets = np.cumsum(offsets)
        self.keys = props

        # Finally the particles are send in circle  (prev) -> (this) --> (next)
        comm = mesh.mpi_comm().tompi4py()
        assert comm.size == 1 or comm.size % 2 == 0
        self.next_rank = (comm.rank + 1) % comm.size
        self.prev_rank = (comm.rank - 1) % comm.size
        self.comm = comm

        if debug:
            self.__add_particles_local = add_timer(self.__add_particles_local)
            self.__add_particles_global = add_timer(self.__add_particles_global)
            self.__update = add_timer(self.__update)

    # Most common in API ---

    def step(self, u, dt, verbose=False):
        'Move particles by forward Euler x += u*dt'
        # Update positions of particles
        for c in self.cells.itervalues():
            vertex_coords, orientation = c.get_vertex_coordinates(), c.orientation()
            # Restrict once per cell
            u.restrict(self.coefficients, self.element, c, vertex_coords, c)
            for p in c.particles:
                x = self.get_x(p)
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix, x, vertex_coords, orientation)
                x[:] = x[:] + dt*np.dot(self.coefficients, self.basis_matrix)[:]
        # Update cells/particles
        self.__update(verbose)

    def add_particles(self, particles, verbose=False):
        '''Add new particles to collection.'''
        # How many particles do we start and end up with
        if verbose: start = plen(self.comm, particles).gc

        # Figure out which particles cannot be added locally on CPU
        not_found = self.__add_particles_local(particles)
        # Send unfoung particles to other CPUs to see if can be added there
        count = len(not_found)
        count_global = self.comm.allgather(count)
        not_found = self.__add_particles_global(count_global, not_found)

        if verbose:
            missing = plen(self.comm, not_found).gc
            info('Wanted to add %d particles. Found %d.' % (start, start-missing))

    # Convenience ---

    def get_x(self, particle):
        '''Return position of particle.'''
        return self.particles[particle][:self.dim]

    def get_property(self, particle, prop):
        '''Get property of particle.'''
        i = self.keys.index(prop)
        f, l = self.offsets[i], self.offsets[i+1]
        select = lambda p: self.particles[p][f:l]

        if isinstance(particle, int): return select(particle)

        return map(select, particle)

    def cell_count(self):
        '''Local and global cell count of cells in collection.'''
        return plen(self.comm, self.cells)

    def particle_count(self):
        '''Local and global particle count of particles in collection.'''
        return plen(self.comm, self.particles)

    def find_cell(self, x):
        '''Find cell which contains x on this CPU, -1 if not found.'''
        point = df.Point(*x)
        c = self.tree.compute_first_entity_collision(point)
        return c if c < self.lim else -1

    # Core ---

    def __add_particles_local(self, particles):
        '''Search CPU for cells that have particles.'''
        not_found = []
        for p in particles:
            x = p[:self.dim]

            c = self.find_cell(x)
            if c > -1:
                # A found particle gets a new unique tag
                tag = next(self.ticket)
                self.particles[tag] = p
                if c not in self.cells: self.cells[c] = CellWithParticles(self, c)
                # and is added to the cell
                self.cells[c] + tag
            else:
                not_found.append(p)
        return not_found

    def __add_particles_global(self, count_global, not_found):
        '''Search other CPUs for cells that have particles.'''
        loop = 1
        # NOTE: if all the particles were in the computational domain then the
        # loop should terminate (at most) once the particles not found on some
        # process travel the full circle. Whathever is not found once the loop
        # is over is outside domain.
        while max(count_global) > 0 and loop < self.comm.size:
            loop += 1
            received = np.zeros(count_global[self.prev_rank]*self.psize, dtype=float)
            sent = np.array(not_found).flatten()
            # Send to next and recv from previous. NOTE: module prevents deadlock
            if self.comm.rank % 2:
                self.comm.Send(sent, self.next_rank, self.comm.rank)
                self.comm.Recv(received, self.prev_rank, self.prev_rank)
            else:
                self.comm.Recv(received, self.prev_rank, self.prev_rank)
                self.comm.Send(sent, self.next_rank, self.comm.rank)
            # Find cells on this CPU for received particles
            received = received.reshape((-1, self.psize))
            not_found = self.__add_particles_local(received)
            count = len(not_found)
            count_global = self.comm.allgather(count)
        self.comm.barrier()
        return not_found

    def __update(self, verbose=False):
        '''
        Update particle and cell dictionaries based on new position of
        particles.
        '''
        if verbose: start = self.particle_count().gc
        cell_map = defaultdict(list)    # Collect new cells with particles
        empty_cells = []                # Cells to be removed from self.cells
        for c in self.cells.itervalues():
            left = []
            for p in c.particles:
                x = self.get_x(p)

                point = df.Point(*x)
                found = c.contains(point)
                # Search only if particle moved outside original cell
                if not found:
                    left.append(p)
                    # Check first neighbor cells
                    for neighbor in c.neighbors:
                        found = neighbor.contains(point)
                        if found:
                            new_cell = neighbor.index()
                            break
                    # Do a completely new search if not found by now, can be -1
                    if not found: new_cell = self.find_cell(x)
                    # Record to map
                    cell_map[new_cell].append(p)
            # Remove from cell the particles that left it
            c.particles.difference_update(set(left))
            if len(c.particles) == 0: empty_cells.append(c.index())
        # Remove cells with no particles
        for c in empty_cells: self.cells.pop(c)

        # Add locally found particles
        local_cells = ifilter(lambda x: x != -1, cell_map.iterkeys())
        for c in local_cells:
            if c not in self.cells: self.cells[c] = CellWithParticles(self, c)
            for p in cell_map[c]: self.cells[c] + p

        # Ship particles not found on this CPU to others
        non_local_particles = cell_map.get(-1, [])
        count_global = self.comm.allgather(len(non_local_particles))
        not_found = self.__add_particles_global(count_global,
                                                [self.particles[i] for i in non_local_particles])
        # Finally remove these particles
        for p in non_local_particles: self.particles.pop(p)

        if verbose:
            stop = self.particle_count().gc
            info('Before update %d, after update %d' % (start, stop))

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, VectorFunctionSpace, info, Timer, interpolate
    from dolfin import Expression
    from mpi4py import MPI as pyMPI
    import sys

    mesh = UnitSquareMesh(20, 20)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression(('-(x[1]-0.5)', 'x[0]-0.5'), degree=1), V)

    nparticles = int(sys.argv[1])
    dt = 0.001

    # TEST1: Check correctness
    if False:
        property_layout = [('x1', 2)]
        lpc = LPCollection(V, property_layout)

        particles_x0 = 0.8*np.random.rand(nparticles, 2)

        x, y = particles_x0[:, 0], particles_x0[:, 1]
        particles_x1 = particles_x0 + dt*np.c_[-(y-0.5), (x-0.5)]

        particles = np.c_[particles_x0, particles_x1]

        lpc.add_particles(particles)
        lpc.step(v, dt)

        loc_error = max(np.linalg.norm(lpc.get_property(p, 'x') - lpc.get_property(p, 'x1'))
                        for p in lpc.particles)
        glob_error = lpc.comm.allreduce(loc_error, op=pyMPI.MAX)
        info('%g %g' % (loc_error, glob_error))

    # Timing
    lpc = LPCollection(V)
    size = lpc.comm.size

    particles = 0.8*np.random.rand(nparticles/size, 2)
    lpc.add_particles(particles, verbose=True)

    t = Timer('LP')
    for i in range(10): lpc.step(v, dt, verbose=True)
    dt = t.stop()

    count = lpc.particle_count()
    info('Stepped %d particles in %g s' % (count.gc, dt))
