import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

def gaussian_distribution(v, mu, sigma):
     return 1/(sigma * np.sqrt(2 * np.pi))* np.exp( - (v-mu)**2 / (2*sigma**2) )

def boltzmann_distribution(v, mu, sigma, d):
    if d == 2:
        if mu == 0:
            return (v/sigma**2)*np.exp( - v**2 / (2.*sigma**2) )
        else:
            return (v/mu)**0.5*(1./(np.sqrt(2*np.pi)*sigma**2))*(np.exp( - (v - mu)**2 / (2.*sigma**2) ) -np.exp( - (v + mu)**2 / (2.*sigma**2) ))
    if d == 3:
        if mu == 0:
            return 4.*np.pi*(v)**2/(sigma*np.sqrt(2 * np.pi))**3*np.exp( - (v - mu)**2 / (2.*sigma**2) )
        else:
            return (v/mu)*(1./(np.sqrt(2*np.pi)*sigma))*(np.exp( - (v - mu)**2 / (2.*sigma**2)) - np.exp( - (v + mu)**2 / (2.*sigma**2) ))

def hist_plot(v, mu, sigma, d, distribution, n_bins, file_name):
    fig = plt.figure()
    count, bins, ignored = plt.hist(v, n_bins, normed=True)
    if distribution == 'Gaussian':
        f = gaussian_distribution(bins, mu, sigma)
        title = 'Gaussian distribution'
        x_label = 'velocity, v_i'
        y_label = 'f(v_i)'
    if distribution == 'Boltzmann':
        f = boltzmann_distribution(bins, mu, sigma, d)
        title = 'Maxwell-Boltzmann distribution'
        x_label = 'speed, v'
        y_label = 'f(v)'
    plt.plot(bins,f,linewidth=2, color='r')
    fig.suptitle(title, fontsize=20)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    fig.savefig(file_name,bbox_inches='tight',dpi=100)

def speed_distribution(v, mu, sigma, ions=0):
    n_bins = 200
    d = len(v[0,:])
    v_2 = np.zeros(len(v))
    if ions == 1:
        file_names = ["i_velocity_xcomp.png", "i_velocity_ycomp.png",
                      "i_velocity_zcomp.png", "ion_speeds.png"]
    elif ions == 0:
        file_names = ["e_velocity_xcomp.png", "e_velocity_ycomp.png",
                      "e_velocity_zcomp.png", "electron_speeds.png"]
    for i in range(d):
        v_i = v[:,i]  # velocity components
        s = np.std(v_i, ddof=0)
        print "std:: ", s, ",   alpha:  ", sigma[i]
        print "mean: ", np.mean(v_i), "   mu:", mu[i]
        #hist_plot(v_i, sigma[i], mu[i], d, 'Gaussian', n_bins, file_names[i])
        v_2 += v_i**2 # Speed squared
    v_sqrt = np.sqrt(v_2) # Speed
    mu_speed = np.sqrt(np.dot(mu,mu))
    hist_plot(v_sqrt, mu_speed, sigma[0], d, 'Boltzmann', n_bins, file_names[-1])
    plt.show()

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    d = 2
    n_particles = 10000000
    T = 300        # Temperature - electrons
    m = 9.10938356e-31 # particle mass - electron
    kB = 1.38064852e-23 # Boltzmann's constant

    alpha_e = np.sqrt(kB*T/m) # Boltzmann factor

    mu = [1.,2.]#,1.]
    sigma = [30.,30.]#,2.]#[alpha_e, alpha_e, alpha_e]

    velocities = np.empty((n_particles,d))

    for i in range(d):
        velocities[:,i] = np.random.normal(mu[i], sigma[i], n_particles)

    speed_distribution(velocities, mu, sigma)
