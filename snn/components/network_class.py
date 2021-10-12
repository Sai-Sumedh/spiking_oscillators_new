import numpy as np
import time
import math
from snn.components.neuron_class import Neuron


class Network:
    """
    Creates a Network object, with methods to update the state of the entire network, 
    and monitor the time course of activity using a probe
    Completely vectorized: both neuronal state and synaptic state are updated 
    in a vectorized fashion (all quantities at once), 
    as opposed to using 'for' loops over neurons and synapses
    """
    
    def __init__(self, title):
        """
        initialize attributes
        """
        self.name = title #name of the network
        self.neurons = [] #list of neurons in the network
        self.synapses = [] #list of synapses in the network
        
        #neuron params- constants (parameters of all neurons)
        self.v_th=[]
        self.tau_mem = []
        self.t_ref = []
        
        #neuron params- state (state of all neurons)
        self.v = []
        self.ref_state = []
        self.spike = []
        self.v_in = []
        
        #synapse params- constants (parameters of all synapses)
        self.pre_neuron_id = []
        self.post_neuron_id = []
        
        #synapse params- state (state of all synapses)
        self.w = []
        self.psp_state= []
        self.psp = []
        self.psp_waveform = []
        
        self.v_noise = []
        
    def add_neuron(self, neuron_x):
        """
        Function: adds a neuron to the network
        
        Parameters:
            neuron_x: Neuron object
                The neuron to be added to the network
        Returns:
            Nothing. updates Neuron.id, Network.neurons
        
        """
        self.neurons.append(neuron_x)
        neuron_x.id = len(self.neurons)-1 #since last element
    
    def add_synapse(self, syn):
        """
        Function: adds a synapse to the network
        Parameters: 
            syn: Synapse object
            
        Returns:
            Nothing. Updates Synapse.id, Network.synapses, Network.pre_neuron, 
            Network.post_neuron 
        """
        self.synapses.append(syn)
        syn.id = len(self.synapses)-1 #last element added
        syn.pre_neuron.out_syn.append(syn)
        syn.post_neuron.in_syn.append(syn)
    
    # to be used after adding all neurons
    def load_constants(self, max_temp_summ=10):
        """
        Function: the constant parameters of neurons and synapses are loaded into the network 
        (where they are stored in vectorized fashion)
        Parameters:
            max_temp_summ: int
                Maximum number of Post Synaptic Potentials which can be summed temporally
                Due to the nature of implementation, only finitely many consecutive spikes of a 
                single pre-neuron can affect the post synaptic potential in a synapse. Default=10
        Returns:
            Nothing. Updates Network parameters
        """
        #neuron params
        num_neurons = len(self.neurons)
        self.v_th = np.array([self.neurons[j].v_th for j in range(num_neurons)]).reshape((-1,1))
        self.tau_mem = np.array([self.neurons[j].tau_mem for j in range(num_neurons)]).reshape((-1,1))
        self.t_ref = np.array([self.neurons[j].t_ref for j in range(num_neurons)]).reshape((-1,1))
        
        #synapse params
        num_synapses = len(self.synapses)
        
        self.pre_neuron_id = np.array([self.synapses[j].pre_neuron.id for j in range(num_synapses)])
        self.post_neuron_id = np.array([self.synapses[j].post_neuron.id for j in range(num_synapses)])
        if num_synapses>0:
            max_psp_len = max([len(syn.psp_waveform) for syn in self.synapses])
            self.psp_waveform = np.zeros((num_synapses, max_psp_len))
            for j in range(num_synapses):
                self.psp_waveform[j, :len(self.synapses[j].psp_waveform)] = self.synapses[j].psp_waveform
        
    
    def create_probe(self, T_total):
        """
        Function: Creates a dictionary with time evolution of all Network quantities of interest
        Parameters: 
            T_total: int
                Total duration (number of timesteps) of the simulation 
        Returns:
            out: dict
                Stores 'v', 'spike', 'v_in', 'psp', 'w', 'v_noise'
                i.e. time evolution of synaptic and neuronal quantities of interest
        """
        N_neurons = len(self.neurons)
        M_synapses = len(self.synapses)
        v_mem = np.zeros((N_neurons, T_total))
        spike_out = np.zeros((N_neurons, T_total))
        v_in = np.zeros((N_neurons, T_total))
        psp_syn = np.zeros((M_synapses, T_total))
        w_syn = np.zeros((M_synapses, T_total))
        v_noise = np.zeros((M_synapses, T_total))
        out = {'v':v_mem, 'spike':spike_out, 'v_in':v_in, 'psp':psp_syn, \
              'w':w_syn, 'v_noise':v_noise}
        return out
    
    
    def initialize(self, probe=False, T_total=0, max_temp_summ=10):
        """
        Function: initialize Network state with state of all neurons and synapses and load constants. 
        Also, create probe if needed
        Parameters: 
            probe: bool
                True if probe is to be created
            T_total: int
                Total number of timesteps of the simulation
            max_temp_summ: int
                Maximum number of Post Synaptic Potentials which can be summed temporally
                Due to the nature of implementation, only finitely many consecutive spikes of a 
                single pre-neuron can affect the post synaptic potential in a synapse. Default=10
        Returns:
            out: dict containing all relevant quantities (if probe is True)
            Updates Network state, constants, etc.
            
        """
        # neuron parameters
        num_neurons = len(self.neurons)
        self.v = np.array([self.neurons[j].v for j in range(num_neurons)]).reshape((-1,1)).astype(float)
        self.ref_state = np.array([self.neurons[j].ref_state for j in range(num_neurons)]).reshape((-1,1))
        self.spike = np.array([self.neurons[j].spike for j in range(num_neurons)]).reshape((-1,1))
        self.v_in = np.array([self.neurons[j].v_in for j in range(num_neurons)]).reshape((-1,1)).astype(float)
        self.v_noise = np.array([0 for j in range(len(self.synapses))]).reshape((-1,1)).astype(float)
        
        #synapse parameters
        num_synapses = len(self.synapses)
        self.w = np.array([self.synapses[j].w for j in range(num_synapses)]).reshape((-1,1))
        self.psp_state = np.zeros((num_synapses, max_temp_summ))
        self.psp = np.array([self.synapses[j].psp for j in range(num_synapses)]).reshape((-1,1))
        self.load_constants(max_temp_summ)
        
        if probe:
            out = self.create_probe(T_total)
            return out
    
    def save_state(self):
        """
        Function: Saves the Network state back into individual neurons and synapses
        Parameters:
            None
        Returns:
            None. Updates the state of neurons and synapses of the network
        """
        for j in range(len(self.neurons)):
            self.neurons[j].v = self.v[j,0]
            self.neurons[j].ref_state = self.ref_state[j,0]
            self.neurons[j].spike = self.spike[j,0]
            self.neurons[j].v_in = self.v_in[j,0]
        
        for j in range(len(self.synapses)):
            self.synapses[j].w = self.w[j, 0]
            self.synapses[j].psp_state = self.psp_state[j, :]
            self.synapses[j].psp = self.psp[j, 0]
            
        
    def update_neuron_params(self, dt=1e-3, tau_spk_tr=15e-3, a_spk_tr=3e-3):
        """
        Function: Updates the Network parameters related to neurons 
        (v, spike, refractory state)
        Parameters:
            dt: float
                Timestep in seconds
            tau_spk_tr: float
                time constant of decay of spike trace, default=0.015
            a_spk_tr: float
                amplitude of increase in spike trace when a spike occurs (in the corresponding neuron)
        Returns:
            None
            Updates Network state (neuron parameters)
        """
        #first: find indices of neurons needing each kind of update
        active = (self.ref_state == 0) #not in refractory period
        no_spike = (self.spike == 0) 
        use_eqn = (active*no_spike) #indices of neurons which need to use LIF equation
        reset = active*(~no_spike) #have spiked at t-1, need reset at t
        
        #refractory state v=0, spike=0, ref_state>0
        self.ref_state[~active] -= 1 #count down
        
        #LIF equation: 0=<v<v_th, spike=0, ref_state=0
        int_fac = np.exp(-dt/self.tau_mem[use_eqn]) #integrating factor
        self.v[use_eqn] =  (1-int_fac)*self.v_in[use_eqn] + int_fac*self.v[use_eqn]
        spiked_now = (self.v>self.v_th)
        self.spike[spiked_now*use_eqn] = 1
        
        #reset: v>v_th, spike=1, ref_state=0
        self.v[reset] = 0
        self.spike[reset] = 0
        self.ref_state[reset] = (self.t_ref[reset]/dt).astype(int)
    
    # update synaptic parameters
    def update_synaptic_params(self):
        """
        Function: Updates all synaptic parameters in Network (Network.w, psp, etc)
        Parameters:
        Returns:
            None
            Updates all synaptic parameters in Network (uses fully vectorized computation)
        """
        
        #decrement psp state
        if len(self.synapses)>0:
            self.psp_state[self.psp_state != 0] -= 1

            #find indices of synapses which had a pre-synaptic neuron spike at t-1
            syn_with_prespike = (self.spike[self.pre_neuron_id] == 1).reshape((-1,))
            assert syn_with_prespike.shape == (len(self.synapses),)

            #shift psp state left by 1 step for syn with pre spike
            self.psp_state[syn_with_prespike, :-1] = self.psp_state[syn_with_prespike, 1:]
            #add K-1 to last column of syn with pre spike
            K = self.psp_waveform.shape[1] # max of psp waveform lengths over all psp waveforms
            self.psp_state[syn_with_prespike, -1] = K-1

            #update psp
            #convert psp state to psp waveform indices
            psp_indices = (K-1-self.psp_state).astype(int)

            syn_indicator = (np.arange(0, len(self.synapses)).reshape((-1,1)) + \
                             np.zeros(psp_indices.shape)).astype(int) #synapse indices
            assert syn_indicator.shape == psp_indices.shape
            psp_contri = self.w*(self.psp_waveform[syn_indicator,psp_indices])
            assert psp_contri.shape == psp_indices.shape
            psp_contri[self.psp_state==0] = 0 #no contribution from psp_state=0
            total_psp = np.sum(psp_contri, axis=1).reshape((-1,1))
            assert total_psp.shape == self.psp.shape
            self.psp = total_psp
        
    def update_v_in(self):
        """
        Function: Update the input voltage of neurons using Post Synaptic Potentials of 
        all incoming synapses to each neuron
        Parameters: 
            None
        Returns:
            None
            Updates Network.v_in
        """
        #clear previous v_in
        self.v_in[:] =0
        if len(self.synapses)>0:
            np.add.at(self.v_in, self.post_neuron_id, self.psp) #in place addition over repeated indices
    
    def add_v_neuron_external(self, v_neuron_ext=None):
        """
        Function: Adds external voltage to neuron membrane potential
        Parameters:
            v_neuron_ext: numpy array with size (num_neurons, 1)  
        Returns:
            None
            Updates Network.v_in by adding v_ext to it
        """
        if v_neuron_ext is not None:
            self.v_in = self.v_in + v_neuron_ext.reshape((-1,1))
    
    def add_v_synapse_external(self, v_synapse_ext=None):
        """
        
        Function: Adds synaptic noise to Post Synaptic Potential for noisy synapses
        Parameters: 
            v_synapse_ext: numpy array, shape=(num_synapses, 1)
        Returns:
            None
            Updates Network.psp
        """
        if v_synapse_ext is not None:
            assert v_synapse_ext.shape == self.psp.shape
            self.psp = self.psp + v_synapse_ext
    
    
    def update_state(self, dt=1e-3, v_neuron_ext=None, out_probe=None, \
                     timestep=None, v_synapse_ext=None):
        """
        Function: Updates the Network state using previously defined functions
            Saves the state to probe
        Parameters:
            dt: float
                Timestep in seconds
            v_neuron_ext: numpy array with size (num_neurons, 1) 
            out_probe: dict with keys 'v', 'spike', 'v_in', 'psp', 'w'
            timestep: int
                Current timestep in the simulation
            v_synapse_ext: numpy array, shape=(M, 1) where M=number of synapses
                Synaptic noise at this timestep 
        Returns:
            None
            Updates Network state and out_probe
        """
        
        self.update_synaptic_params() #uses spike[t-1]
        self.add_v_synapse_external(v_synapse_ext) #adds v_synapse_ext to self.psp[t] 
                                                       #to get updated self.psp[t]
        self.update_neuron_params(dt) #uses v_in[t-1] and other neuron params
        self.update_v_in() #updated last to ensure correctness: it resets and computes v_in
        self.add_v_neuron_external(v_neuron_ext)
        self.save_state()
        
        if out_probe is not None:
            out_probe['v'][:,timestep] = self.v[:,0].reshape((-1,))
            out_probe['spike'][:,timestep]=self.spike[:,0].reshape((-1,))
            out_probe['v_in'][:,timestep] = self.v_in[:,0].reshape((-1,))
            #out_probe['v_noise'][:, timestep] = self.v_noise.reshape((-1,))
            
            out_probe['psp'][:,timestep] = self.psp.reshape((-1,))
            out_probe['w'][:, timestep] = self.w.reshape((-1,))
            