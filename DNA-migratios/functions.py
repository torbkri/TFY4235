#%%
import numpy as np
import matplotlib.pyplot as plt

#Constants:

ALPHA = 0.2
L = 20E-6
T = 1
TIMESTEP = 0.000001
KT = 26E-3
DU = 80
RI = 12E-9
NU=1E-3
GAMMA = 6 * np.pi * RI * NU
W = DU / GAMMA / L**2

#%%

#Functions 

def reduce(x:float,t:float) -> tuple[float,float]:
    return (x/L, t * W )

def inv_reduce(x:float, t:float)->tuple[float,float]:
    return (x * L, t / W)

def potential(x:float,t:float, flashing:bool = True) -> float:
    #all units are reduced

    if(x < ALPHA): u = x / ALPHA
    else: u = (1 - x) / (1 - ALPHA)
    f = 1
    if flashing:
        if(t < W * 3 * T / 4):
            f = 0
    return u * f

def force(x:float,t:float, flashing:bool = True) -> float:
    #all units are reduced
    du = 0
    #check if the potential is active
    time = 1
    if(flashing):
        if(t<W*3*T/4):time = 0
    
    #force is derivative of potential
    if(x < ALPHA): return - 1 / ALPHA * time
    else: return 1 / (ALPHA-1) * time


def euler_scheme(x:float, t:float, flashing:bool = True) -> float:
    #all units are reduced
    Brownian = np.random.normal(0,1)
    return x + force(x,t,flashing=flashing) * TIMESTEP + np.sqrt(2 * KT * TIMESTEP / DU) * Brownian
    
# %%
pos = np.zeros((100000))
time = np.zeros((100000))

#Running for a single particle for 10 steps:
# for i in range(100):
x,t = 0,0
for j in range(100000):
    x,t = euler_scheme(x,t,False),t+TIMESTEP
    pos[j] = x
    time[j] = t
# inv_reduce(x,t)
# print(f"position: {round(x,8)} \t time: {t}")
# %%
fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(time,pos)
# ax[1].plot(time)

# print(pos)
# %%
