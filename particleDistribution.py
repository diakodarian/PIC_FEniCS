import numpy as np
import matplotlib.pyplot as plt

def gaussian_distribution(v, sigma, mu=0.0):
     return 1/(sigma * np.sqrt(2 * np.pi))* np.exp( - (v-mu)**2 / (2*sigma**2) )

def boltzmann_distribution(v, sigma, mu=0.0):
    return 4.*np.pi*v**2/(sigma*np.sqrt(2 * np.pi))**3*np.exp( - (v - mu)**2 / (2.*sigma**2) )

def hist_plot(distribution, v, sigma, mu = 0.0, n_bins = 100):
    fig = plt.figure()
    count, bins, ignored = plt.hist(v, n_bins, normed=True)
    if distribution == 'Gaussian':
        f = gaussian_distribution(bins, sigma, 0.0)
        title = 'Gaussian distribution'
        x_label = 'velocity, v_i'
        y_label = 'f(v_i)'
    if distribution == 'Boltzmann':
        f = boltzmann_distribution(bins, sigma, 0.0)
        title = 'Maxwell-Boltzmann distribution'
        x_label = 'speed, v'
        y_label = 'f(v)'
    plt.plot(bins,f,linewidth=2, color='r')
    fig.suptitle(title, fontsize=20)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    
def speed_distribution(v, d, alpha):
    n_bins = 100
    v_2 = np.zeros(len(v))
    for i in range(d):
        v_i = v[:,i]  # velocity components
        s = np.std(v_i, ddof=1)
        print "std:: ", s, ",   alpha:  ", alpha
        hist_plot('Gaussian', v_i, alpha, 0.0, n_bins)
        v_2 += v_i**2 # Speed squared
    v_sqrt = np.sqrt(v_2) # Speed
    hist_plot('Boltzmann', v_sqrt, alpha, 0.0, n_bins)
    plt.show()

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    d = 3
    n_particles = 10000
    T = 300        # Temperature - electrons
    m = 9.10938356e-31 # particle mass - electron
    kB = 1.38064852e-23 # Boltzmann's constant

    alpha_e = np.sqrt(kB*T/m) # Boltzmann factor
    print alpha_e
    mu, sigma = 0., 1. # mean and standard deviation
    initial_velocities = np.reshape(alpha_e * np.random.normal(mu, sigma, d*n_particles),
                                    (n_particles, d))
    speed_distribution(initial_velocities, d, alpha_e)
