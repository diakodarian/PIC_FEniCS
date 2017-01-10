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

def Langmuir_waves_positions(n_electrons, l):

    f = lambda x: x-np.cos(4*np.pi*x)/(4.*np.pi)
    inv_f = inversefunc(f, domain=[l[0], l[1]], open_domain=True)
    random_positions = np.random.uniform(-.079,0.92, n_electrons)
    positions = inv_f(random_positions)
    return np.array(positions)

def initialize_particle_positions(N_e, N_i, l, d, random_domain, initial_type):
    # Initial particle positions
    if d == 3:
        if random_domain == 'shpere':
            initial_electron_positions = \
                    RandomSphere([0.5,0.5,0.5], 0.5).generate([N_e, N_e, N_e])
            initial_ion_positions = \
                    RandomSphere([0.5,0.5,0.5], 0.5).generate([N_i, N_i, N_i])
        elif random_domain == 'box':
            initial_electron_positions = \
            RandomBox([l[0],l[2],l[4]],[l[1],l[3],l[5]]).generate([N_e,N_e,N_e])
            initial_ion_positions = \
            RandomBox([l[0],l[2],l[4]],[l[1],l[3],l[5]]).generate([N_i,N_i,N_i])
    if d == 2:
        if random_domain == 'shpere':
            initial_electron_positions = \
            RandomCircle([0.5,0.5], 0.5).generate([N_e, N_e])
            initial_ion_positions = \
            RandomCircle([0.5,0.5], 0.5).generate([N_i, N_i])
        elif random_domain == 'box':
            initial_electron_positions = \
            RandomRectangle([l[0],l[2]], [l[1],l[3]]).generate([N_e, N_e])
            initial_ion_positions = \
            RandomRectangle([l[0],l[2]], [l[1],l[3]]).generate([N_i, N_i])

    n_ions = len(initial_ion_positions)
    n_electrons = len(initial_electron_positions)

    initial_positions = []
    if initial_type == 'random':
        initial_positions.extend(initial_electron_positions)
    if initial_type == 'Langmuir_waves':
        x_comp = Langmuir_waves_positions(n_electrons, l)
        initial_electron_positions[:,0] = x_comp
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

def Langmuir_waves_velocities(x, n_electrons, n_ions, d, l, alpha_e):
    # Initial Gaussian distribution of the x-component of electrons velocity
    #mu, sigma = 0., 1. # mean and standard deviation
    #A = 0.1 # Amplitude of oscillations
    #p = 5 # Number of oscillation periods
    initial_electron_velocities = np.zeros((n_electrons, d))
    #initial_electron_velocities[:,0] = A*np.sin(2*p*np.pi*(x-l[0])/(l[1]-l[0]))
    initial_ion_velocities = np.zeros((n_ions, d))

    return initial_electron_velocities, initial_ion_velocities

def intialize_particle_velocities(x, n_electrons, n_ions, d, l,
                                  alpha_e, alpha_i, initial_type):
    # if initial_type == 'random':
    #     initial_electron_velocities, initial_ion_velocities = \
    #     random_velocities(n_electrons, n_ions, d, alpha_e, alpha_i)
    # if initial_type == 'Langmuir_waves':
    #     initial_electron_velocities, initial_ion_velocities = \
    #     Langmuir_waves_velocities(x, n_electrons, n_ions, d, l, alpha_e)
    mu, sigma = 0., 1. # mean and standard deviation
    initial_electron_velocities = \
    np.reshape(alpha_e * np.random.normal(mu, sigma, d*n_electrons),
                                    (n_electrons, d))
    initial_ion_velocities = \
    np.reshape(alpha_i * np.random.normal(mu, sigma, d*n_ions),
                                    (n_ions, d))

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
    properties[key].append(m_e)

    for i in range(n_electrons-1):
        properties.setdefault(key, [])
        properties[key].append(m_e)
    for i in range(n_ions):
        properties.setdefault(key, [])
        properties[key].append(m_i)

    return properties

def initial_conditions(N_e, N_i, l, d, w, q_e, q_i, m_e, m_i,
                       alpha_e, alpha_i, random_domain,
                       initial_type):

    initial_positions, n_total_particles, n_electrons, n_ions = \
    initialize_particle_positions(N_e, N_i, l, d, random_domain, initial_type)
    # The x-component of electron positions
    x = initial_positions[:n_electrons, 0]
    initial_velocities = \
    intialize_particle_velocities(x, n_electrons, n_ions, d, l,
                                  alpha_e, alpha_i, initial_type)
    properties = \
    intialize_particle_properties(n_electrons, n_ions, w, q_e, q_i, m_e, m_i)

    return initial_positions, initial_velocities, properties, n_electrons

if __name__ == '__main__':
    d = 2              # Space dimension
    N_e = 30          # Number of electrons
    N_i = 30        # Number of ions
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
    q_i = Z*e  # Electric charge - ions
    w = rho_p/N_e

    l1 = 0.
    l2 = 1.
    w1 = 0.
    w2 = 1.
    h1 = 0.
    h2 = 1.
    l = [l1, l2, w1, w2, h1, h2]
    initial_positions, initial_velocities, properties, n_electrons = \
    initial_conditions(N_e, N_i, l, d, w, q_e, q_i, m_e, m_i,
                           alpha_e, alpha_i, random_domain = 'box',
                           initial_type = 'Langmuir_waves')
    fig = plt.figure()
    count, bins, ignored = plt.hist(initial_positions[:n_electrons,0], 600, normed=True)

    f = 1+np.sin(2.*np.pi*bins)
    print(sum(f))
    plt.plot(bins, f,linewidth=2, color='r')
    plt.show()
    # print('initial_positions: ', initial_positions)
    # print('initial_velocities: ', initial_velocities)
    # print('properties:' , properties)
