import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft



from numpy import ma
from matplotlib import ticker, cm

# Параметры пучка
wavelength = 780e-9
E0 = 1
c = 3e+8
dt = 7.5e-15
dx = 10 * wavelength


TAU = 6.5e-15 / 5
XTAU = wavelength * 40

omega0 = 2 * np.pi * c / wavelength
dots = 2 ** 10
f = omega0 / (2 * np.pi)

t_ = np.linspace(-20e-15, 20e-15, dots)
x_ = np.linspace(-4 * dx, 4 * dx, dots)

DX = x_[1] - x_[2]
T, X = np.meshgrid(t_, x_, sparse=True)
E_x = E0 * np.exp(-2 * (X / XTAU) ** 2) * np.exp(-2 * (T / TAU) ** 2) * np.sin(omega0 * T)


#######################################
#### Спектры времени/ пространства ####
#######################################

E_x_spec_x = np.fft.fftshift(np.fft.fft(E_x, axis=0), axes=0)
E_x_spec_t = np.fft.fft(E_x, axis=1)
E_x_spec_xt = np.fft.fft2(E_x)


##### Фокус
F = 3.75e-3



def k():
    # kx = 2 * np.pi * np.fft.fftfreq(dots, DX)/c
    kx = (np.fft.fftshift(np.fft.fftfreq(dots))/(x_[1]-x_[2])/c)[::-1] #* 2 * np.pi
    plt.plot(kx)
    plt.show()

    # kx, kx = np.meshgrid(kx, kx)
    k_ = (np.fft.fftshift(np.fft.fftfreq(dots))/(t_[1]-t_[2])/c * 2 * np.pi)[::-1]
    
    plt.plot(k_)
    plt.show()
    k_n, k_n = np.meshgrid(k_, k_)
    # print(kx.max())# * n_w(w_)
    
    kz = np.sqrt(k_n ** 2 - kx.T ** 2)[dots//2+1:]
    plt.pcolormesh(kz)
    plt.show()
    return kx, kz, k_


#### Линза 
def linza(x, k_):
    return np.exp(-(1j * k_ / (2 * F)) * x ** 2)



###### Надо для линзы (решетка по X:k и Y:x)
kx, kz, k_ = k()
kv, xv = np.meshgrid(k_, x_)

########## Расчет прохождения пучка через линзу ##########
# E_x_lens = np.fft.fftshift(E_x_spec_t, axes = 1) * linza(xv, kv)
E_x_lens = np.fft.fftshift(linza(xv, kv), axes = 1) * E_x_spec_t
E_x_lens = np.fft.ifft(E_x_lens, axis = 1)


#######################################
############### ГРАФИКИ ###############
#######################################

fig, axes = plt.subplots(3, 2, figsize=(8,8))

## Первая колонка ###
ax = axes[0,0]
ax.pcolormesh(abs(E_x_spec_t), cmap='inferno')

ax = axes[1,0]
ax.pcolormesh(np.fft.fftshift(linza(xv, kv), axes = 1).real, cmap='inferno')


ax = axes[2,0]
ax.pcolormesh(E_x_lens.real, cmap='inferno')

### Вторая Колонка ###
ax = axes[0,1]
ax.pcolormesh(abs(E_x_spec_x), cmap='inferno')

ax = axes[1,1]
ax.pcolormesh(abs(np.fft.fftshift(E_x_spec_xt)), cmap='inferno')


ax = axes[2,1]
ax.pcolormesh(E_x.real, cmap='inferno')

plt.show()


#######################################
####### РАСЧЕТ В ПРОСТРАНСТВЕ #########
#######################################

def C_():
    return np.fft.fft2(E_x_lens)[:,dots//2+1:]

# fig, axes = plt.subplots(1, 3, figsize=(8,8))
# ax = axes[0]
# ax.pcolormesh(abs(C_()))
# ax = axes[1]
# ax.plot(np.exp(-1j * kz * F/10))
# ax = axes[2]
# ax.plot(np.exp(-1j * kz * F*10))
# plt.show()

def g_x(z):
    kx, kz, k_ = k()
    g = C_() * np.exp(1j * kz/(2*z) *(x_[dots//2+1:]**2))# * np.exp(-1j * kz * z)
    return np.fft.ifft2(g).real

def g_z(z):
    kx, kz, k_ = k()
    g = (kx[dots//2+1:] * C_() * np.exp(-1j * kz[:,dots//2+1:] * z)/kz[:,dots//2+1:])
    return np.fft.ifft2(g).real

# def angular(z):
#     fft_c = np.fft.fft2(np.fft.fft2(E_x_lens) * np.exp(1j * k_/(2*z) *(X**2)))
#     c = np.fft.ifft2(fft_c)
#     return c

# fig, axes = plt.subplots(2, 2, figsize=(8,8))
# ax = axes[0,0]
# ax.pcolormesh(abs(E_x_spec_x), cmap='inferno')
# ax = axes[1,0]
# ax.pcolormesh(abs(E_x_spec_t), cmap='inferno')
# ax = axes[0,1]
# ax.set_ylim([500, 520])
# ax.pcolormesh(abs(E_x), cmap='inferno')
# ax = axes[1,1]
# ax.pcolormesh(abs(E_x_lens), cmap='inferno')
# plt.show()

fig, axes = plt.subplots(3, 2, figsize=(8,8))

ax = axes[0,0]
ax.pcolormesh(g_z(0), cmap='inferno')
ax = axes[1,0]
ax.pcolormesh(g_z(F/2), cmap='inferno')
ax = axes[2,0]
ax.pcolormesh(g_z(F), cmap='inferno')

ax = axes[0,1]
ax.pcolormesh(g_x(0), cmap='inferno')
ax = axes[1,1]
ax.pcolormesh(g_x(F/2), cmap='inferno')
ax = axes[2,1]
ax.pcolormesh(g_x(F), cmap='inferno')
plt.show()
