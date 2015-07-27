import numpy as np
import RL_utils as ru
from numba import jit
import utility as ut
import random
import math


# -------------------------------------------------------------------------------------
# Note to self:  Remember to check that all non-chosen action values are being caried 
# forward from trial to trial.
# -------------------------------------------------------------------------------------


def kernel_Qs(session, bias, CK, SSK):
    'Evaluate modifier for Q_low due to kernels.'
    kernel_Qs      = - np.ones(session.n_trials + 1) * bias         # Positive bias leads to choosing high.
    kernel_Qs[1:] += 2. * (0.5 - session.CTSO['choices']) * CK       # Positive choice kernel promotes repeating choice.
    kernel_Qs[1:] += 2. * (0.5 - session.CTSO['second_steps']) * SSK # Promotes rotational biases.
    return kernel_Qs

class _RL_agent:

    def __init__(self):

        self.n_params = len(self.param_names)
        self.calculates_gradient = False

    def add_kernel_params(self):
        self.param_names  += ['bias','CK' , 'SSK']
        self.param_ranges += ['unc' ,'unc', 'unc']

# -------------------------------------------------------------------------------------
# New agent
# -------------------------------------------------------------------------------------

class New_agent(_RL_agent):
    'Mixture agent with decays.'

    def __init__(self, use_kernels = True):

        self.name = 'Mix_dec_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha', 'iTemp', 'lambd', 'W' ,  'tlr' , 'D '  , 'tdec']

        self.param_ranges = ['unit', 'pos'   , 'unit' ,'unit', 'unit', 'unit', 'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, transitions, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CTSO')
        session_start_trials = session.blocks['session_start_trials']
        # Unpack parameters.
        alpha, iTemp, lambd, W, tlr, D, tdec = params_T[:7]   # Transition decay rate.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values (low, high).
        Q_td_s = np.zeros([n_trials + 1 , 2])       # Model free second step action values (right, left).
        trans_probs = np.zeros([n_trials + 1 , 2])  # Transition probabilities for low and high pokes.
        trans_probs[0,:] = 0.5  # Initialize first trial transition probabilities.

        for i, (f, c, t, s, o) in enumerate(zip(session_start_trials,
                                                choices, transitions, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.
            ns = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D)   # First step forgetting.
            Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D)   # Second step forgetting.

            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] + \
                            alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o           # Second step TD update.

            # Update transition probabilities.

            trans_probs[i+1,nc] = trans_probs[i,nc] - tdec * (trans_probs[i,nc] - 0.5)  # Transition prob. forgetting.

            trans_probs[i+1,c] = (1. - tlr) * trans_probs[i,c] + tlr * (s == 0)         # Transition prob. update.

        # Evaluate choice probabilities and likelihood. 

        Q_mb = trans_probs * np.tile(Q_td_s[:,0],[2,1]).T + \
                (1. - trans_probs) * np.tile(Q_td_s[:,1],[2,1]).T # Model based action values. 

        Q_net = W * Q_mb + (1. - W) * Q_td_f # Mixture of model based and model free values.

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'Q_mb'        : Q_mb[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_mb'        : iTemp * W *       (Q_mb [:-1,1]  - Q_mb [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood


# -------------------------------------------------------------------------------------
# Mixture_decay_agent
# -------------------------------------------------------------------------------------

class Mixture_decay_agent(_RL_agent):
    'Mixture agent with decays.'

    def __init__(self, use_kernels = True):

        self.name = 'Mix_dec_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha', 'iTemp', 'lambd', 'W' ,  'tlr' , 'D '  , 'tdec']

        self.param_ranges = ['unit', 'pos'   , 'unit' ,'unit', 'unit', 'unit', 'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        alpha, iTemp, lambd, W, tlr, D, tdec = params_T[:7]   # Transition decay rate.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values (low, high).
        Q_td_s = np.zeros([n_trials + 1 , 2])       # Model free second step action values (right, left).
        trans_probs = np.zeros([n_trials + 1 , 2])  # Transition probabilities for low and high pokes.
        trans_probs[0,:] = 0.5  # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.
            ns = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D)   # First step forgetting.
            Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D)   # Second step forgetting.

            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] + \
                            alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o           # Second step TD update.

            # Update transition probabilities.

            trans_probs[i+1,nc] = trans_probs[i,nc] - tdec * (trans_probs[i,nc] - 0.5)  # Transition prob. forgetting.

            trans_probs[i+1,c] = (1. - tlr) * trans_probs[i,c] + tlr * (s == 0)         # Transition prob. update.

        # Evaluate choice probabilities and likelihood. 

        Q_mb = trans_probs * np.tile(Q_td_s[:,0],[2,1]).T + \
                (1. - trans_probs) * np.tile(Q_td_s[:,1],[2,1]).T # Model based action values. 

        Q_net = W * Q_mb + (1. - W) * Q_td_f # Mixture of model based and model free values.

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'Q_mb'        : Q_mb[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_mb'        : iTemp * W *       (Q_mb [:-1,1]  - Q_mb [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Mix_dec_3LR_agent
# -------------------------------------------------------------------------------------

class Mix_dec_3LR_agent(_RL_agent):
    '''Mixture agent with decays using seperate learning and decay rates for
     first and second step of model free and model based state values.'''

    def __init__(self, use_kernels = True):

        self.name = 'Mix_dec_3LR_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha_1', 'D_1 ', 'alpha_2', 'D_2 ', 'alpha_3', 'D_3 ' ,  'tlr' , 'tdec', 'iTemp', 'lambd', 'W'  ]

        self.param_ranges = ['unit'   , 'unit', 'unit'   , 'unit', 'unit'   , 'unit' , 'unit' , 'unit', 'pos'  , 'unit' ,'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        alpha_1, D_1, alpha_2, D_2, alpha_3, D_3, tlr, tdec, iTemp, lambd, W = params_T[:11]
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values  (low, high).
        Q_td_s = np.zeros([n_trials + 1 , 2])       # Model free second step action values (right, left).
        Q_mb_s = np.zeros([n_trials + 1 , 2])       # Second step action values used by model-based system (right, left).
        trans_probs = np.zeros([n_trials + 1 , 2])  # Transition probabilities for low and high pokes.
        trans_probs[0,:] = 0.5  # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.
            ns = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D_1)   # First step forgetting.
            Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D_2)   # Second step forgetting.
            Q_mb_s[i+1,ns] = Q_mb_s[i, ns] * (1. - D_3)   # Second step forgetting.

            Q_td_f[i+1,c] = (1. - alpha_1) * Q_td_f[i,c] + \
                            alpha_1 * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha_2) * Q_td_s[i,s] +  alpha_2 * o           # Second step TD update.
            Q_mb_s[i+1,s] = (1. - alpha_3) * Q_mb_s[i,s] +  alpha_3 * o           # Second step TD update.

            # Update transition probabilities.

            trans_probs[i+1,nc] = trans_probs[i,nc] - tdec * (trans_probs[i,nc] - 0.5)  # Transition prob. forgetting.
            trans_probs[i+1,c] = (1. - tlr) * trans_probs[i,c] + tlr * (s == 0)         # Transition prob. update.

        # Evaluate choice probabilities and likelihood. 

        Q_mb = trans_probs * np.tile(Q_mb_s[:,0],[2,1]).T + \
                (1. - trans_probs) * np.tile(Q_mb_s[:,1],[2,1]).T # Model based action values. 

        Q_net = W * Q_mb + (1. - W) * Q_td_f # Mixture of model based and model free values.

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'Q_mb'        : Q_mb[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_mb'        : iTemp * W *       (Q_mb [:-1,1]  - Q_mb [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Mixture_winstay_agent
# -------------------------------------------------------------------------------------

class Mixture_winstay_agent(_RL_agent):
    'Mixture agent with decays.'

    def __init__(self, use_kernels = True):

        self.name = 'Mix_dec_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha', 'iTemp', 'lambd', 'W' ,  'tlr' , 'D '  , 'tdec', 'winstay']

        self.param_ranges = ['unit', 'pos'   , 'unit' ,'unit', 'unit', 'unit', 'unit', 'unc'    ]

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        alpha, iTemp, lambd, W, tlr, D, tdec, winstay = params_T[:8]   # Transition decay rate.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values (low, high).
        Q_td_s = np.zeros([n_trials + 1 , 2])       # Model free second step action values (right, left).
        trans_probs = np.zeros([n_trials + 1 , 2])  # Transition probabilities for low and high pokes.
        trans_probs[0,:] = 0.5  # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.
            ns = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D)   # First step forgetting.
            Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D)   # Second step forgetting.

            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] + \
                            alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o           # Second step TD update.

            # Update transition probabilities.

            trans_probs[i+1,nc] = trans_probs[i,nc] - tdec * (trans_probs[i,nc] - 0.5)  # Transition prob. forgetting.

            trans_probs[i+1,c] = (1. - tlr) * trans_probs[i,c] + tlr * (s == 0)         # Transition prob. update.

        # Evaluate choice probabilities and likelihood. 

        Q_mb = trans_probs * np.tile(Q_td_s[:,0],[2,1]).T + \
                (1. - trans_probs) * np.tile(Q_td_s[:,1],[2,1]).T # Model based action values. 

        Q_net = W * Q_mb + (1. - W) * Q_td_f # Mixture of model based and model free values.

        Q_net[np.arange(1, n_trials), choices[:-1]] += (outcomes[:-1] - 0.5) * winstay # Win-stay lose-shift effect.        

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'Q_mb'        : Q_mb[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_mb'        : iTemp * W *       (Q_mb [:-1,1]  - Q_mb [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Winstay_agent
# -------------------------------------------------------------------------------------

class Winstay_agent(_RL_agent):

    def __init__(self, use_kernels = True):

        self.name = 'Winstay_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['winstay']

        self.param_ranges = ['unc'    ]

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        winstay = params_T[0]   # Transition decay rate.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_net = np.zeros([n_trials + 1 , 2]) 

        Q_net[np.arange(1, n_trials), choices[:-1]] += (outcomes[:-1] - 0.5) * winstay # Win-stay lose-shift effect.        

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, 1.)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Model_free_agent
# -------------------------------------------------------------------------------------

class Model_free_agent(_RL_agent):
    'Mixture agent with decays.'

    def __init__(self, use_kernels = True):

        self.name = 'Model_free_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha', 'iTemp', 'lambd', 'D '  ]

        self.param_ranges = ['unit', 'pos'   , 'unit' , 'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        alpha, iTemp, lambd, D = params_T[:4]   # Q value decay parameter.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values (low, high).
        Q_td_s = np.zeros([n_trials + 1 , 2])       # Model free second step action values (right, left).

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.
            ns = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D)   # First step forgetting.
            Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D)   # Second step forgetting.

            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] + \
                            alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o       # Second step TD update.

        # Evaluate choice probabilities and likelihood. 

        Q_net = Q_td_f

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Extended state agent.
# -------------------------------------------------------------------------------------

class Extended_state_agent(_RL_agent):

    def __init__(self, use_kernels = True):

        self.name = 'Ext_state_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha1', 'iTemp', 'lambd', 'D' ]

        self.param_ranges = ['unit'  , 'pos'  , 'unit'  ,'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        n_trials = len(choices)
        prev_second_steps = np.zeros(n_trials + 1, int)
        prev_second_steps[1:] = second_steps

        # Unpack parameters.
        alpha, iTemp, lambd, D = params_T[:4]   # Q value decay parameter.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]
 
        #Variables.
        Q_td_f = np.zeros([n_trials + 1 , 2, 2])         # Model free action values at first step. indicies: trial, first step, previous second step.
        Q_td_s = np.zeros([n_trials + 1 , 2])            # Model free action values at second step.

        for i, (c, s, v, o) in enumerate(zip(choices, second_steps, prev_second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step. (0 or 1)
            ns = 1 - s  # State not reached at second step. (0 or 1)
            nv = 1 - v  # State not reached at second step on previous trial (0 or 1)

            # Update model free action values. 

            if True: #use_Q_decay: 

                Q_td_f[i+1,nc,  v] = Q_td_f[i+1,nc,  v] * (1. - D)   # First step forgetting.
                Q_td_f[i+1, 0, nv] = Q_td_f[i+1, 0, nv] * (1. - D)   # First step forgetting.
                Q_td_f[i+1, 1, nv] = Q_td_f[i+1, 1, nv] * (1. - D)   # First step forgetting.

                Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D) # Second step forgetting.


            Q_td_f[i+1, c, v] = (1. - alpha) * Q_td_f[i+1, c, v] + \
                          alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o           # Second step TD update.

        # Evaluate choice probabilities and likelihood. 

        Q_net = Q_td_f[np.arange(n_trials + 1), :, prev_second_steps]

        if True:# use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            pass
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Arbitration_agent
# -------------------------------------------------------------------------------------

class Arbitration_agent(_RL_agent):
    '''Mixture agent which uses basic reliability based arbitration 
    between model-based and model-free.'''

    def __init__(self, use_kernels = True):

        self.name = 'Arb_agent'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha', 'iTemp', 'lambd', 'W' ,  'tlr' , 'D '  , 'tdec', 'A'  , 'alr' ]

        self.param_ranges = ['unit', 'pos'   , 'unit' ,'unc', 'unit', 'unit' , 'unit', 'unc', 'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):#, return_trial_data = False):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        alpha, iTemp, lambd, W, tlr, D, tdec, A, alr = params_T[:9]   # Learning rate for arbitration.
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values (low, high).
        Q_td_s = np.zeros([n_trials + 1 , 2])       # Model free second step action values (right, left).
        arb    = np.zeros(n_trials + 1)             # Arbitration parameter, positive means more model based.
        trans_probs = np.zeros([n_trials + 1 , 2])  # Transition probabilities for low and high pokes.
        trans_probs[0,:] = 0.5  # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.
            ns = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D)   # First step forgetting.
            Q_td_s[i+1,ns] = Q_td_s[i, ns] * (1. - D)   # Second step forgetting.

            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] + \
                            alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s])) # First step TD update.
      
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o           # Second step TD update.

            # Update transition probabilities.

            trans_probs[i+1,nc] = trans_probs[i,nc] - tdec * (trans_probs[i,nc] - 0.5)  # Transition prob. forgetting.
            state_prediction_error = (s == 0) - trans_probs[i,c]
            trans_probs[i+1,c] = trans_probs[i,c] + tlr * state_prediction_error         # Transition prob. update.

            # Update Arbitration.

            arb[i + 1] = arb[i] + alr * (abs(state_prediction_error) - arb[i])

        # Evaluate choice probabilities and likelihood. 

        Q_mb = trans_probs * np.tile(Q_td_s[:,0],[2,1]).T + \
                (1. - trans_probs) * np.tile(Q_td_s[:,1],[2,1]).T # Model based action values. 

        W_arb = np.tile(ru.sigmoid(W - A * arb),[2,1]).T  # Trial by trial model basedness.

        Q_net = W_arb * Q_mb + (1. - W_arb) * Q_td_f # Mixture of model based and model free values.

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'Q_mb'        : Q_mb[:-1,:],
                    'W_arb'       : W_arb[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_mb'        : iTemp * W *       (Q_mb [:-1,1]  - Q_mb [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood


# -------------------------------------------------------------------------------------
# Model_free_agent
# -------------------------------------------------------------------------------------

class TD1(_RL_agent):
    'TD1 (direct reinforcement) agent.'

    def __init__(self, use_kernels = True):

        self.name = 'TD1 agent.'

        self.use_kernels = use_kernels  # Use bias, choice kernel and second step kernels.

        self.param_names  = ['alpha', 'iTemp', 'D '  ]

        self.param_ranges = ['unit', 'pos'   , 'unit']

        if use_kernels: self.add_kernel_params()

        _RL_agent.__init__(self)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, 'CSO')

        # Unpack parameters.
        alpha, iTemp, D = params_T[:3]  
        if self.use_kernels: bias, CK, SSK  = params_T[-3:]

        #Variables.
        n_trials = len(choices)
        Q_td_f = np.zeros([n_trials + 1 , 2])       # Model free first step action values (low, high).

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c  # Action not chosen at first step.

            # Update model free action values. 

            Q_td_f[i+1,nc] = Q_td_f[i, nc] * (1. - D)                 # First step forgetting.
            Q_td_f[i+1, c] = (1. - alpha) * Q_td_f[i,c] +  alpha * o  # First step TD update.

        # Evaluate choice probabilities and likelihood. 

        Q_net = Q_td_f

        if self.use_kernels:
            Q_net[:,0] += kernel_Qs(session, bias, CK, SSK)

        choice_probs = ru.array_softmax(Q_net, iTemp)
        trial_log_likelihood = ru.protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if False:#return_trial_data:
            return {'Q_net'       : Q_net[:-1,:],  # Action values
                    'Q_td'        : Q_td_f[:-1,:],
                    'P_net'       : iTemp *           (Q_net[:-1,1] - (Q_net[:-1,0] + bias)), # Preferences.
                    'P_td'        : iTemp * (1 - W) * (Q_td_f [:-1,1]  - Q_td_f [:-1,0]),
                    'P_k'         : - kernel_Qs(session, 0., CK, SSK)[:-1],
                    'choice_probs': choice_probs}
        else:
            return session_log_likelihood


