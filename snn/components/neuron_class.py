import numpy as np
import time
import math
import matplotlib.pyplot as plt


class Neuron:
    """
    Creates a neuron object. Stores relevant constants, state of the neuron and incoming and outgoing synapses
    """
    def __init__(self, title, v_th=1, is_noisy=False, t_ref=5e-3, tau_mem=20e-3):
        """
        initialize attributes like membrane potential, spike, and other constants
        """
        self.name = title
        self.id = -1 #index in the network
        
        self.v = 0 #membrane potential
        self.v_th = v_th
        self.v_in = 0 #total input postsynaptic voltage
        self.tau_mem = tau_mem
        self.t_ref = t_ref #refractory period: in s
        self.ref_state = 0 #indicates neuron's refractory state
        
        self.spike = 0 #spike: 1 for spike issued, 0 for no spike issued
#         self.is_noisy = is_noisy
        self.in_syn = [] #incoming synapses 
        self.out_syn = [] #outgoing synapses 

        
        
