import numpy as np
import time
import math

        
class Synapse:
    """
    Creates a synapse object, which stores all relevant quantities like synaptic weight, post synaptic potential,
    whether the synapse is noisy, etc
    """
    def __init__(self, title, pre_neuron, post_neuron, psp_waveform, w=0.5, is_noisy=False):
        """
        initialize attributes
        """
        self.name = title
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.w = w #can change due to stdp
        
        self.psp_waveform = psp_waveform
        self.psp_state = [0] # list, allows multiple pre spikes
        self.psp = 0 #this is redundant, can be obtained from psp_waveform and psp_state, but included for convenience
        
        self.id = -1 #index in network.synapses
        self.is_noisy = is_noisy