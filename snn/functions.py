import numpy as np
import math
from snn.components.neuron_class import Neuron
from snn.components.synapse_class import Synapse
from snn.components.network_class import Network

def get_psp_waveform(dt=1e-3, len_psp = 100, tau1 = 7e-3, tau2 = 5e-3):
    """
    Function: Get a double exponential waveform to serve as Post Synaptic Potential 
    with desired time step, time constants and duration
        PSP = ( exp(-t/tau1)-exp(-t/tau2) )/max_val
        where division by max_val is to ensure peak value is 1
    
    Parameters: 
        dt: float ()
            The timestep in seconds. default = 0.001
        len_psp: int 
            The duration of time PSP waveform. default = 100
        tau1: float
            The longer time constant of the waveform. Decides the decay timescale. Default 0.007
        tau2: float
            The short time constant of the waveform. Along with tau1, it decides the rise timescale and peak time. default 0.005
    
    Returns: 
        h: numpy array
            The peak-normalized PSP waveform. 
        
    """
    t = dt*np.arange(0, len_psp)
    h = np.exp(-t/tau1) - np.exp(-t/tau2) #double exponential waveform
    h = h/np.amax(h) #peak normalized
    return h

def freq(spk_train, dt, get_avg = False, get_f_loc=False, get_f_interpolated=False):
    '''
    returns frequency of oscillation vs time
    '''
    t_spike = dt*np.argwhere(spk_train).reshape((-1,))
    if len(t_spike)<2:
        return 0
    else:
        isi =  np.array([t_spike[i+1]-t_spike[i] for i in range(len(t_spike)-1)])
        f_loc = np.array([(t_spike[i+1]+t_spike[i])/2 for i in range(len(t_spike)-1)])
        f = (1/isi)
        if get_f_interpolated:
            f_intp = np.zeros(spk_train.shape)
            for i in range(len(t_spike)-1):
                f_intp[int(t_spike[i]/dt):int(t_spike[i+1]/dt)] = f[i]
            f_intp[int(t_spike[-1]/dt):] = f[len(t_spike)-2] #keep last few timesteps same as previous oscillation period
            return f, f_intp, f_loc
        elif get_avg:
            return np.mean(f)
        elif get_f_loc:
            return f, f_loc
        else:
            return f
        
        
def create_network(num_neurons, w_self, w_cross, adjacency_mat, psp_waveform, v_init, T, t_ref=5e-3, tau_mem=20e-3, weighted=False):
    '''
    creates and returns a network with desired connectivity
    '''
    # create neurons
    net = Network('net')
    for i in range(num_neurons):
        net.add_neuron(Neuron(str(i), t_ref=t_ref, tau_mem=tau_mem))
    
    # create synapses
    syn = np.argwhere(adjacency_mat)
    
    if weighted:
        
        for j in range(syn.shape[0]):
            net.add_synapse(Synapse(str(syn[j,0])+str(syn[j,1]),net.neurons[syn[j,0]],net.neurons[syn[j,1]], 
                                    psp_waveform, w=adjacency_mat[syn[j,0],syn[j,1]]))
        
    else: #adjacency matrix binary-> symmetric connectivity
        
        for j in range(syn.shape[0]):

            # self connections
            if syn[j,0]==syn[j,1]:
                net.add_synapse(Synapse(str(syn[j,0])+str(syn[j,1]),net.neurons[syn[j,0]],net.neurons[syn[j,1]], psp_waveform, w=w_self))

            #cross connections
            else:
                net.add_synapse(Synapse(str(syn[j,0])+str(syn[j,1]),net.neurons[syn[j,0]],net.neurons[syn[j,1]], psp_waveform, w=w_cross))
        
    probe = net.initialize(probe=True, T_total=T)
    net.v[:] = v_init.copy() 
    return net, probe
    
def settling_time(f_t1, f_t2, dt=1e-3, f_th=1e-3):
    '''
    returns settling time
    '''
    delf = np.abs(f_t1-f_t2)
    if np.sum(delf[-5:])<f_th:
        t_settling = round(dt*np.argwhere(delf>f_th)[-1,0], 3)
    else:
        t_settling = -1
    return t_settling
    
    
def phase(spk_train, dt, f):
    '''
    spk_train is a binary spike train of size (T, 1)
    '''
    T = len(spk_train)
    phi = np.zeros((T,1))   
    assert (f.shape==spk_train.shape)
    t = np.linspace(0, (T-1)*dt, T)
    t_spike = dt*np.argwhere(spk_train).reshape((-1,))
    t_spk_intp = np.zeros(spk_train.shape)
    for i in range(len(t_spike)-1):
        t_spk_intp[int(t_spike[i]/dt):int(t_spike[i+1]/dt)] = t_spike[i]
    t_spk_intp[int(t_spike[-1]/dt):] = t_spike[-1]
    phi = (360*np.multiply(f, t-t_spk_intp))%360 
    return phi

def osc_state(spk_train_all, dt):
    '''
    returns the frequency and phase of all oscillators as functions of time
    '''
    N, T = spk_train_all.shape
    f = np.zeros(spk_train_all.shape)
    phi = np.zeros(spk_train_all.shape)
    for n in range(N):
        _,f_n,_ = freq(spk_train_all[n,:].reshape((-1,)), dt, get_f_interpolated=True)
        phi_n = phase(spk_train_all[n, :].reshape((-1,)), dt, f_n)
        f[n,:] = f_n
        phi[n,:] = phi_n
    return f, phi

def settling_phase(spk_train_all, dt):
    f, phi = osc_state(spk_train_all, dt)
    phi_diff = (phi-phi[0,:])%360 #0th neuron is reference
    phi_ss = np.mean(phi_diff[:,-5:], axis=1) #average of last 5
    return phi_ss, phi_diff
