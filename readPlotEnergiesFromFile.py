from numpy import *
import matplotlib.pyplot as plt

import os,sys
from scipy import stats

t = []
ek = []
ep = []

f = open('energies.txt', 'r').readlines()
N = len(f)-1
for i in range(0,N):
    w = f[i].split()
    t.append(w[0])
    ek.append(w[1])
    ep.append(w[2])

# Plot the figure
g_ratio = 1.61803398875
width = 12
length = width*g_ratio

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 18*1.1
plt.rcParams['axes.labelsize'] = 18*1.1
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 16*1.5
plt.rcParams['ytick.labelsize'] = 16*1.5
plt.rcParams['legend.fontsize'] = 14*1.1
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.color_cycle'] = '#348ABD', '#A60628', '#15efe4','#b009bf', '#1e0eb2', '#33c452', '#ef7b15', '#841654', '#2a296d', '#bcbf09'

l = array(t, dtype=float)*1400000

fig = plt.figure()
plt.plot(t, ek, label='Kinetic energy')
plt.plot(t, ep, label='Potential energy')
#plt.plot(t, l, label='linear')

plt.xlabel('Time $[s]$',fontsize=18)
plt.ylabel('Energies',fontsize=18)
plt.title('Langmuir waves',fontsize=20)
# plt.xlim()
# plt.ylim()
plt.legend(loc='best')
plt.grid(True)
fig.set_tight_layout(True)
fig.savefig('energies.eps',bbox_inches='tight',dpi=100)
fig.savefig('energies.png',bbox_inches='tight',dpi=100)
plt.show()
