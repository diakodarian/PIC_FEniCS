from __future__ import print_function
from LagrangianParticles_test import RandomBox, RandomSphere
from LagrangianParticles_test import RandomCircle, RandomRectangle
from particleDistribution import speed_distribution
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
from pynverse import inversefunc
import sys

comm = pyMPI.COMM_WORLD

def random_1d_positions(L, N_e, N_i):
    a = L[0]
    b = L[1]

    n_e_points = np.product(N_e)
    points = np.random.rand(n_e_points)
    initial_electron_positions = a + points*(b-a)

    n_i_points = np.product(N_i)
    points = np.random.rand(n_i_points)
    initial_ion_positions = a + points*(b-a)

    return initial_electron_positions, initial_ion_positions

def random_2d_positions(L, N_e, N_i, random_domain):
    l1 = L[0]
    w1 = L[1]
    l2 = L[2]
    w2 = L[3]
    if random_domain == 'shpere':
        initial_electron_positions = \
        RandomCircle([0.5,0.5], 0.5).generate([N_e, 1])
        initial_ion_positions = \
        RandomCircle([0.5,0.5], 0.5).generate([N_i, 1])
    elif random_domain == 'box':
        initial_electron_positions = \
        RandomRectangle([l1,w1], [l2,w2]).generate([N_e, 1])
        initial_ion_positions = \
        RandomRectangle([l1,w1], [l2,w2]).generate([N_i, 1])

    return initial_electron_positions, initial_ion_positions

def random_3d_positions(L, N_e, N_i, random_domain):
    l1 = L[0]
    w1 = L[1]
    h1 = L[2]
    l2 = L[3]
    w2 = L[4]
    h2 = L[5]
    if random_domain == 'shpere':
        initial_electron_positions = \
                RandomSphere([0.5,0.5,0.5], 0.5).generate([N_e, 1,1])
        initial_ion_positions = \
                RandomSphere([0.5,0.5,0.5], 0.5).generate([N_i, 1,1])
    elif random_domain == 'box':
        initial_electron_positions = \
        RandomBox([l1,w1,h1],[l2,w2,h2]).generate([N_e,1,1])
        initial_ion_positions = \
        RandomBox([l1,w1,h1],[l2,w2,h2]).generate([N_i,1,1])

    return initial_electron_positions, initial_ion_positions

def cylindrical_object(L, initial_ion_positions, initial_electron_positions,
                        object_info):

    # By default the cylinder is aligned with the z-axis,
    # and placed at center of the domain
    l1 = L[0]
    w1 = L[1]
    h1 = L[2]
    l2 = L[3]
    w2 = L[4]
    h2 = L[5]
    s0 = [object_info[0], object_info[1]]
    r0 = object_info[2]
    h0 = object_info[3]

    z0 = (h2-h0)/2.      # Bottom point of cylinder
    z1 = (h2+h0)/2.      # Top point of cylinder
    print(z0, "  ", z1)
    index_e = []
    index_i = []
    for i in range(len(initial_ion_positions)):
        x = initial_ion_positions[i]
        if (x[2] > z0 and x[2] < z1 and np.dot(x[:2]-s0, x[:2]-s0) < r0**2):
            index_i.append(i)
    for i in range(len(initial_electron_positions)):
        x = initial_electron_positions[i]
        if (x[2] > z0 and x[2] < z1 and np.dot(x[:2]-s0, x[:2]-s0) < r0**2):
            index_e.append(i)

    initial_electron_positions = np.delete(initial_electron_positions, index_e,\
                                            axis=0)
    initial_ion_positions = np.delete(initial_ion_positions, index_i, axis=0)

    len_e = len(index_e)
    len_i = len(index_i)
    while len_e > 0:
        index_e = []
        initial_electron_positions_sec = \
        RandomBox([l1,w1, h1], [l2,w2, h2]).generate([len_e, 1, 1])
        for i in range(len(initial_electron_positions_sec)):
            x = initial_electron_positions_sec[i]
            if (x[2] > z0 and x[2] < z1 and np.dot(x[:2]-s0, x[:2]-s0) < r0**2):
                index_e.append(i)
        initial_electron_positions_sec = \
        np.delete(initial_electron_positions_sec, index_e, axis=0)
        initial_electron_positions = np.append(initial_electron_positions,\
                                     initial_electron_positions_sec, axis=0)
        len_e = len(index_e)
    while len_i > 0:
        index_i = []
        initial_ion_positions_sec = \
        RandomBox([l1,w1,h1], [l2,w2,h2]).generate([len_i, 1, 1])
        for i in range(len(initial_ion_positions_sec)):
            x = initial_ion_positions_sec[i]
            if (x[2] > z0 and x[2] < z1 and np.dot(x[:2]-s0, x[:2]-s0) < r0**2):
                index_i.append(i)
        initial_ion_positions_sec = \
        np.delete(initial_ion_positions_sec, index_i, axis=0)
        initial_ion_positions = np.append(initial_ion_positions,\
                                initial_ion_positions_sec, axis=0)
        len_i = len(index_i)

    return initial_electron_positions, initial_ion_positions

def spherical_object(L, initial_ion_positions, initial_electron_positions,
                    object_info):
    d = int(len(L)/2)
    if d == 2:
        l1 = L[0]
        w1 = L[1]
        l2 = L[2]
        w2 = L[3]
        s0 = [object_info[0], object_info[1]]
        r0 = object_info[2]
    if d == 3:
        l1 = L[0]
        w1 = L[1]
        h1 = L[2]
        l2 = L[3]
        w2 = L[4]
        h2 = L[5]
        s0 = [object_info[0], object_info[1], object_info[2]]
        r0 = object_info[3]
    index_e = []
    index_i = []
    for i in range(len(initial_ion_positions)):
        x = initial_ion_positions[i]
        if np.dot(x-s0, x-s0) < r0**2:
            index_i.append(i)
    for i in range(len(initial_electron_positions)):
        x = initial_electron_positions[i]
        if np.dot(x-s0, x-s0) < r0**2:
            index_e.append(i)

    initial_electron_positions = np.delete(initial_electron_positions, index_e,\
                                           axis=0)
    initial_ion_positions = np.delete(initial_ion_positions, index_i, axis=0)

    len_e = len(index_e)
    len_i = len(index_i)
    while len_e > 0:
        index_e = []
        if d == 2:
            initial_electron_positions_sec = \
            RandomRectangle([l1,w1], [l2,w2]).generate([len_e, 1])
        if d == 3:
            initial_electron_positions_sec = \
            RandomBox([l1,w1, h1], [l2,w2, h2]).generate([len_e, 1, 1])
        for i in range(len(initial_electron_positions_sec)):
            x = initial_electron_positions_sec[i]
            if np.dot(x-s0, x-s0) < r0**2:
                index_e.append(i)
        initial_electron_positions_sec = \
        np.delete(initial_electron_positions_sec, index_e, axis=0)
        initial_electron_positions = np.append(initial_electron_positions, \
                                     initial_electron_positions_sec, axis=0)
        len_e = len(index_e)
    while len_i > 0:
        index_i = []
        if d == 2:
            initial_ion_positions_sec = \
            RandomRectangle([l1,w1], [l2,w2]).generate([len_i, 1])
        if d == 3:
            initial_ion_positions_sec = \
            RandomBox([l1,w1,h1], [l2,w2,h2]).generate([len_i, 1, 1])
        for i in range(len(initial_ion_positions_sec)):
            x = initial_ion_positions_sec[i]
            if np.dot(x-s0, x-s0) < r0**2:
                index_i.append(i)
        initial_ion_positions_sec = np.delete(initial_ion_positions_sec,\
                                    index_i, axis=0)
        initial_ion_positions = np.append(initial_ion_positions, \
                                initial_ion_positions_sec, axis=0)
        len_i = len(index_i)

    return initial_electron_positions, initial_ion_positions

def Langmuir_waves_perturbations(n_electrons, l1,l2):

    f = lambda x: x-np.cos(4*np.pi*x)/(4.*np.pi)
    inv_f = inversefunc(f, domain=[l1,l2], open_domain=True)
    random_positions = np.random.uniform(-.079,0.92, n_electrons)
    positions = inv_f(random_positions)
    return np.array(positions)

def Langmuir_waves_positions(initial_electron_positions, n_electrons,
                             l1, l2):
    oscillation_type = "2" # type 1 uses pynverse
    if oscillation_type == "1":
        x_comp = Langmuir_waves_perturbations(n_electrons, l1,l2)
        initial_electron_positions[:,0] = x_comp
    if oscillation_type == "2":
        A = 0.1
        x_comp = initial_electron_positions[:,0]
        delta = A*np.sin(x_comp)
        initial_electron_positions[:,0] += delta

    x_comp = initial_electron_positions[:,0]
    l = l2 - l1
    for i in range(len(x_comp)):
        if x_comp[i] < l1 or x_comp[i] > l2:
            x_comp[i] = (x_comp[i]+abs(l1))%l + l1
    initial_electron_positions[:,0] = x_comp
    return initial_electron_positions

def initialize_particle_positions(N_e, N_i, L, random_domain, initial_type,
                                  object_info=None):
    d = int(len(L)/2)
    # Initial particle positions
    if d == 3:
        initial_electron_positions, initial_ion_positions = \
        random_3d_positions(L, N_e, N_i, random_domain)
    if d == 2:
        initial_electron_positions, initial_ion_positions = \
        random_2d_positions(L, N_e, N_i, random_domain)

    n_ions = len(initial_ion_positions)
    n_electrons = len(initial_electron_positions)

    initial_positions = []

    if initial_type == 'spherical_object':
        initial_electron_positions, initial_ion_positions = \
        spherical_object(L, initial_ion_positions, initial_electron_positions,\
                        object_info)
        initial_positions.extend(initial_electron_positions)

    if initial_type == 'cylindrical_object':
        initial_electron_positions, initial_ion_positions = \
        cylindrical_object(L, initial_ion_positions, \
                           initial_electron_positions, object_info)
        initial_positions.extend(initial_electron_positions)

    if initial_type == 'random':
        initial_positions.extend(initial_electron_positions)

    if initial_type == 'Langmuir_waves':
        initial_electron_positions = \
        Langmuir_waves_positions(initial_electron_positions, n_electrons,\
                                L[0], L[2])
        initial_positions.extend(initial_electron_positions)

    initial_positions.extend(initial_ion_positions)
    initial_positions = np.array(initial_positions)
    n_total_particles = len(initial_positions)

    if comm.Get_rank() == 0:
        print("Total number of particles: ", n_total_particles)
        print("Total number of electrons: ", n_electrons)
        print("Total number of ions: ", n_ions)

    return initial_positions, n_total_particles, n_electrons, n_ions

def random_velocities(n_electrons, n_ions, d, alpha_e, alpha_i):
    # Initial Gaussian distribution of velocity components
    mu, sigma = 0., 1. # mean and standard deviation
    initial_electron_velocities = \
    np.reshape(alpha_e * np.random.normal(mu, sigma, d*n_electrons),
                                    (n_electrons, d))
    initial_ion_velocities = \
    np.reshape(alpha_i * np.random.normal(mu, sigma, d*n_ions),
                                    (n_ions, d))
    return initial_electron_velocities, initial_ion_velocities

def object_velocities(n_electrons, n_ions, d, alpha_e, alpha_i):
    # Initial Gaussian distribution of velocity components
    mu, sigma = 0., 1. # mean and standard deviation
    initial_electron_velocities = \
    np.reshape(alpha_e * np.random.normal(mu, sigma, d*n_electrons),
                                    (n_electrons, d))
    initial_ion_velocities = \
    np.reshape(alpha_i * np.random.normal(mu, sigma, d*n_ions),
                                    (n_ions, d))
    return initial_electron_velocities, initial_ion_velocities

def Langmuir_waves_velocities(d, n_electrons, n_ions):
    initial_electron_velocities = np.zeros((n_electrons, d))
    initial_ion_velocities = np.zeros((n_ions, d))
    return initial_electron_velocities, initial_ion_velocities

def intialize_particle_velocities(n_electrons, n_ions, L,
                                  alpha_e, alpha_i, initial_type):
    d = int(len(L)/2)
    if initial_type == 'random':
        initial_electron_velocities, initial_ion_velocities = \
        random_velocities(n_electrons, n_ions, d, alpha_e, alpha_i)
    if initial_type == 'Langmuir_waves':
        initial_electron_velocities, initial_ion_velocities = \
        Langmuir_waves_velocities(d, n_electrons, n_ions)
    if initial_type == 'spherical_object':
        initial_electron_velocities, initial_ion_velocities = \
        object_velocities(n_electrons, n_ions, d, alpha_e, alpha_i)
    if initial_type == 'cylindrical_object':
        initial_electron_velocities, initial_ion_velocities = \
        object_velocities(n_electrons, n_ions, d, alpha_e, alpha_i)

    initial_velocities = []
    initial_velocities.extend(initial_electron_velocities)
    initial_velocities.extend(initial_ion_velocities)
    initial_velocities = np.array(initial_velocities)
    return initial_velocities

def intialize_particle_properties(n_electrons, n_ions, w, q_e, q_i, m_e, m_i):
    # Add charge of particles to properties
    properties = {}
    key = 'q'
    properties.setdefault(key, [])
    properties[key].append(w*q_e)

    for i in range(n_electrons-1):
        properties.setdefault(key, [])
        properties[key].append(w*q_e)
    for i in range(n_ions):
        properties.setdefault(key, [])
        properties[key].append(w*q_i)

    # Add mass of particles to properties
    key = 'm'
    properties.setdefault(key, [])
    properties[key].append(w*m_e)

    for i in range(n_electrons-1):
        properties.setdefault(key, [])
        properties[key].append(w*m_e)
    for i in range(n_ions):
        properties.setdefault(key, [])
        properties[key].append(w*m_i)

    return properties

def initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i,
                       alpha_e, alpha_i, object_info, random_domain,
                       initial_type):

    initial_positions, n_total_particles, n_electrons, n_ions = \
    initialize_particle_positions(N_e, N_i, L, random_domain, initial_type, object_info)
    initial_velocities = intialize_particle_velocities(n_electrons, n_ions, L,
                                  alpha_e, alpha_i, initial_type)
    properties = \
    intialize_particle_properties(n_electrons, n_ions, w, q_e, q_i, m_e, m_i)

    return initial_positions, initial_velocities, properties, n_electrons

if __name__ == '__main__':

    d = 3
    N_e = 20000          # Number of electrons
    N_i = 20000        # Number of ions
    # Physical parameters
    rho_p = 8.*N_e        # Plasma density
    T_e = 1.        # Temperature - electrons
    T_i = 1.        # Temperature - ions
    kB = 1.         # Boltzmann's constant
    e = 1.        # Elementary charge
    Z = 1          # Atomic number
    m_e = 1.        # particle mass - electron
    m_i = 100.        # particle mass - ion

    alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
    alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

    q_e = -e     # Electric charge - electron
    q_i = Z*e    # Electric charge - ions
    w = rho_p/N_e

    l1 = 0.
    l2 = 2.*np.pi
    w1 = 0.
    w2 = 2.*np.pi
    h1 = 0.
    h2 = 2.*np.pi

    # The object:
    object_type = 'cylindrical_object' # Options sphere or cylinder or None
    if object_type == 'spherical_object':
        x0 = np.pi
        y0 = np.pi
        z0 = np.pi
        r0 = 0.5
        if d == 2:
            object_info = [x0, y0, r0]
            l = [l1, w1, l2, w2]
        elif d==3:
            object_info = [x0, y0, z0, r0]
            l = [l1, w1, h1, l2, w2, h2]
    if object_type == 'cylindrical_object':
        x0 = np.pi
        y0 = np.pi
        r0 = 0.5
        h0 = 4.0

        object_info = [x0, y0, r0, h0]
        l = [l1, w1, h1, l2, w2, h2]

    initial_positions, initial_velocities, properties, n_electrons = \
    initial_conditions(N_e, N_i, l, w, q_e, q_i, m_e, m_i,
                           alpha_e, alpha_i, object_info, random_domain = 'box',
                           initial_type = object_type)
    # print("positions: ", initial_positions)
    # print("positions: ", initial_velocities)
    # print("positions: ", properties)
    fig = plt.figure()
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if object_type == 'cylindrical_object':
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(
            patches.Rectangle(
                (l2/2.-r0, (h2-h0)/2.),
                2*r0,
                h0,
                fill=False      # remove background
            )
        )
    if object_type == 'spherical_object':
        # theta goes from 0 to 2pi
        theta = np.linspace(0, 2*np.pi, 100)
        # the radius of the circle
        r = np.sqrt(0.25)
        # compute x1 and x2
        x1 = np.pi + r*np.cos(theta)
        x2 = np.pi + r*np.sin(theta)
        ax = fig.gca()
        ax.plot(x1, x2, c='k', linewidth=3)
        ax.set_aspect(1)

    skip = 1
    xy_electrons = initial_positions[:n_electrons]
    xy_ions = initial_positions[n_electrons:]

    electrons_tmp = []
    ions_tmp = []

    if (d == 3 and object_type=='spherical_object'):
        for i in range(xy_electrons.shape[0]):
            if np.abs(xy_electrons[i,2] - np.pi) < 1e-1:
                electrons_tmp.append(xy_electrons[i,:2])
        for i in range(xy_ions.shape[0]):
            if np.abs(xy_ions[i,2] - np.pi) < 1e-1:
                ions_tmp.append(xy_ions[i,:2])
        xy_electrons = np.array(electrons_tmp)
        xy_ions = np.array(ions_tmp)

    if (d == 3 and object_type=='cylindrical_object'):
        for i in range(xy_electrons.shape[0]):
            if np.abs(xy_electrons[i,1] - np.pi) < 1e-1:
                electrons_tmp.append(xy_electrons[i,::2])
        for i in range(xy_ions.shape[0]):
            if np.abs(xy_ions[i,1] - np.pi) < 1e-1:
                ions_tmp.append(xy_ions[i,::2])
        xy_electrons = np.array(electrons_tmp)
        xy_ions = np.array(ions_tmp)
    print(xy_electrons.shape, "   ", xy_ions.shape)
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
    ax.axis([0, l2, 0, w2])

    if object_type == None:
        fig = plt.figure()
        count, bins, ignored = plt.hist(initial_positions[:n_electrons,0], 600, normed=True)

        f = 0.1*np.sin(4.*np.pi*bins)
        #print(sum(f))
        plt.plot(bins, f,linewidth=2, color='r')

        # print('initial_positions: ', initial_positions)
        # print('initial_velocities: ', initial_velocities)
        # print('properties:' , properties)
    plt.show()
