import numpy as np
import matplotlib.pyplot as plt

def speed_distribution(v, d, alpha):
    v_2 = np.zeros(len(v))
    for i in range(d):
        v_i = v[:,i]

        s = np.std(v_i, ddof=1)
        print "std:: ", s , ",   alpha:  ", alpha
        plt.figure()
        count, bins, ignored = plt.hist(v_i, 100, normed=True)
        plt.plot(bins, 1/(alpha * np.sqrt(2 * np.pi))* np.exp( - bins**2 / (2*alpha**2) ),linewidth=2, color='r')

        v_2 += v_i**2
    v_sqrt = np.sqrt(v_2)
    plt.figure()
    count, bins, ignored = plt.hist(v_sqrt, 100, normed=True)
    plt.plot(bins, 4.*np.pi*bins**2/(alpha*np.sqrt(2 * np.pi))**3*np.exp( - (bins)**2 / (2.*alpha**2) ),linewidth=2, color='r')
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
