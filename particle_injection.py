import numpy as np
from initial_conditions import random_1d_positions


def inject_particles_2d(Lx, Ly, count_e, count_i):
    dim = len(count_e)/2
    count = []
    count.append(count_e)
    count.append(count_i)
    p_positions = []

    for i in range(len(count)):
        num = count[i]
        e_pos_x1, e_pos_x2 = random_1d_positions(Lx, num[2], num[3])
        e_pos_y1, e_pos_y2 = random_1d_positions(Ly, num[0], num[1])

        e_pos_y = np.concatenate([e_pos_y1, e_pos_y2])
        e_pos_x = np.concatenate([e_pos_x1, e_pos_x2])

        x_e = np.empty((len(e_pos_x), dim))
        y_e = np.empty((len(e_pos_y), dim))

        for i in range(len(e_pos_y)):
            if i < num[0]:
                y_e[i,0] = Lx[0]
            else:
                y_e[i,0] = Lx[1]
            y_e[i,1] = e_pos_y[i]

        for i in range(len(e_pos_x)):
            if i < num[2]:
                x_e[i,1] = Ly[0]
            else:
                x_e[i,1] = Ly[1]
            x_e[i,0] = e_pos_x[i]

        e_pos = np.concatenate([x_e, y_e])
        p_positions.extend(e_pos)

    p_positions = np.array(p_positions)
    return p_positions

if __name__ == '__main__':
    l1 = 0.
    l2 = 2*np.pi
    w1 = 0.
    w2 = 2*np.pi
    Lx = [l1, l2]
    Ly = [w1, w2]

    count_e = [1, 2, 3, 4]
    count_i = [2, 3, 4, 5]
    p = inject_particles_2d(Lx, Ly, count_e, count_i)
    print p
