'''
    ncrc: neuronal culture reservoir computing
        This work is accompanied by a scientific publication.
        Please cite: A.M. Houben, A.-C. Haeb, J. Garcia-Ojalvo & J. Soriano, Reservoir computing in simulated neuronal cultures: Effect of network structure, (in press)
    Copyright (C) 2025 Akke Mats Houben (akke@akkehouben.net)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
import matplotlib.pyplot as plt

import nc_sim
import ncrc

Dt = 1
IS_MODULAR = True

P_rewire = 0.2

# readout
readout_Dt  = 1e-3
readout_win = 1
N_cross     = 1

# culture parameters:
width   = 1.00  # width / height
rho     = 300   # neuron density
alpha   = 0.5   # connection probability

# network pattern:
block_size  = 0.001
chan_width  = 0.2

# neuron parameters:
gA      = 6     # excitatory PSP strength
gG      = 2*gA  # inhibitory PSP strength
sigma   = 2     # noise drive

# stimulation parameters:
input_Dt                    = 1e-3
stim_amp                    = gA    # input pulse strength   
digits                      = [1, 4]# input digits
t0                          = 30    # start after t0 seconds to stimulate
N_rep                       = 100   # number of input reps
input_len                   = 50e-3 # length of each input (ms)
inter_input_interval_min    = 5     # min time between inputs (s)
inter_input_interval_max    = 10    # max time between inputs (s)
max_freq                    = 500   # max stim freq. per channel (Hz)

# generate network:
Hm = int(width/block_size)
H = np.zeros((Hm, Hm))-1
if IS_MODULAR:
    for i in range(Hm):
        y = i*width/Hm + block_size
        for j in range(Hm):
            x = j*width/Hm + block_size
            if y <= 3*width/8 and ( x <= 3*width/8 or x >= 5*width/8 ):
                H[i,j] = 0
            elif y >= 5*width/8 and ( x <= 3*width/8 or x >= 5*width/8 ):
                H[i,j] = 0

            if y > 3*width/8 and y < 5*width/8 and x >= (3*width/16-chan_width/2)  and x <= (3*width/16+chan_width/2):
                H[i,j] = -2
            if y > 3*width/8 and y < 5*width/8 and x >= (13*width/16-chan_width/2)  and x <= (13*width/16+chan_width/2):
                H[i,j] = -2

            if x > 3*width/8 and x < 5*width/8 and y >= (3*width/16-chan_width/2)  and y <= (3*width/16+chan_width/2):
                H[i,j] = -2
            if x > 3*width/8 and x < 5*width/8 and y >= (13*width/16-chan_width/2)  and y <= (13*width/16+chan_width/2):
                H[i,j] = -2
else:   # not modular
    for i in range(Hm):
        y = i*width/Hm + block_size
        for j in range(Hm):
            x = j*width/Hm + block_size
            if y >= width/8 and y < 7*width/8 and x >= width/8 and x < 7*width/8:
                H[i,j] = 0

X, Y = nc_sim.axons.place_neurons(width, width, H, rho=rho)
N = len(X)
H[H==-2] = 0
H[H==-1] = 100

W0 = nc_sim.axons.grow_W(width, width, X, Y, H=H)
W = np.multiply(W0, np.random.rand(N,N) > (1-alpha))

W_perm = np.copy(W)
# create rewired connectivity:
for i in range(N):
    for j in range(N):
        if W_perm[i,j] != 0 and np.random.rand() > (1-P_rewire):
            W_perm[i,j] = 0
        if W0[i,j] !=0 and W[i,j] == 0 and np.random.rand() > (1-P_rewire):
            W_perm[i,j] = W0[i,j]

# generate inputs:
input_neuron_ixs = ncrc.rc.generate_inputs.select_input_neurons(X, Y, 0, width/2., fraction=0.3)
input_y, input_t, It, Ic = ncrc.rc.generate_inputs.gen_inputs(digits, input_neuron_ixs, N_rep, t0=t0, Dt=input_Dt, input_len=input_len, inter_input_interval_min=inter_input_interval_min, inter_input_interval_max=inter_input_interval_max, max_freq=max_freq)
Ic = np.array(Ic)
It = np.array(It)*1e3   # s to ms
Ic = Ic[np.argsort(It)]
It = np.sort(It)

Tn = np.max(It) + inter_input_interval_max*1e3

# init network:
nw = nc_sim.neurons.izhikevich.init_network(len(X), W, Tn=Tn, sigma=sigma, Dt=Dt)
nw_perm = nc_sim.neurons.izhikevich.init_network(len(X), W_perm, Tn=Tn, sigma=sigma, Dt=Dt)

# run dynamics:
input_ix = 0

Nt = nw['params']['Nt']
Iext = np.zeros(N)
for n in np.arange(1, Nt):
    #if n  % (Nt//100) == 1: print(n-1, Nt)
    t = n*Dt    # current time (ms)
    Iext[:] = 0
    while input_ix < len(It) and It[input_ix] <= t:
        Iext[Ic[input_ix]] = stim_amp
        input_ix += 1

    nc_sim.neurons.izhikevich.simulation_step(n, nw, Iext=Iext)
    nc_sim.neurons.izhikevich.simulation_step(n, nw_perm, Iext=Iext)

spikes_T = np.array(nw['spike_T'])*1e-3
spikes_Y = np.array(nw['spike_Y'], dtype=int)
spikes_T_perm = np.array(nw_perm['spike_T'])*1e-3
spikes_Y_perm = np.array(nw_perm['spike_Y'], dtype=int)

# analyse responses:
input_t = np.array(input_t)
input_y = np.array(input_y, dtype=int)

classes = np.unique(input_y)
taus = np.arange(int(-100e-3/readout_Dt), int(300e-3/readout_Dt), 1)*readout_Dt

T = np.array(spikes_T)
Y = np.array(spikes_Y, dtype=int)
for y in input_neuron_ixs:
    T = np.delete(T, Y==y)
    Y = np.delete(Y, Y==y)
Tn = np.max(input_t)+taus[-1]+readout_win
X = ncrc.rc.analyse.spikes_to_traces(T, Y, Tn, win_sigma=5e-3, Dt=readout_Dt)

T_perm = np.array(spikes_T_perm)
Y_perm = np.array(spikes_Y_perm, dtype=int)
for y in input_neuron_ixs:
    T_perm = np.delete(T_perm, Y_perm==y)
    Y_perm = np.delete(Y_perm, Y_perm==y)
Tn = np.max(input_t)+taus[-1]+readout_win
X_perm = ncrc.rc.analyse.spikes_to_traces(T_perm, Y_perm, Tn, win_sigma=5e-3, Dt=readout_Dt)

scores = np.zeros((len(taus), N_cross))
for tau_ix, tau in enumerate(taus):
    for cross_ix in range(N_cross):
        input_t_train, input_y_train, _, _ = ncrc.rc.analyse.split_train_test(input_t, input_y, 1)
        _, _, input_t_test, input_y_test = ncrc.rc.analyse.split_train_test(input_t, input_y, 0)
        
        train_reservoir_states, train_labels = ncrc.rc.analyse.collect_states(input_t_train, input_y_train, tau, X, winT=readout_win, Dt=readout_Dt)
        test_reservoir_states, test_labels = ncrc.rc.analyse.collect_states(input_t_test, input_y_test, tau, X_perm, winT=readout_win, Dt=readout_Dt)

        lm = ncrc.rc.analyse.fit_lm(train_reservoir_states, train_labels)

        score = lm.score(test_reservoir_states, test_labels)

        scores[tau_ix, cross_ix] = score

        

taus, scores = ncrc.rc.analyse.score(input_t, input_y, spikes_T, spikes_Y, input_ixs=input_neuron_ixs)

plt.plot(taus, scores)
plt.show()


# plot:
#plt.pcolormesh(np.arange(Hm)*width/Hm, np.arange(Hm)*width/Hm, H)
#plt.plot(X, Y, '.')
#plt.plot(X[input_neuron_ixs], Y[input_neuron_ixs], 'o')
#plt.show()

print('done!')
