import numpy as np
import RL_utils as ru
import time
import RL_plotting as rp
import pylab as p
import utility as ut
from RL_utils import softmax_bin as softmax

# -------------------------------------------------------------------------------------
# session_based_log_reg
# -------------------------------------------------------------------------------------

class _session_based_log_reg():
    '''
    Superclass for logistic regression agents which evaluate log likelihood for entier sessions 
    with a single function call rather than trial by trial.
    '''

    def __init__(self):#, n_back):

        #self.n_back = n_back

        self.n_params = 1 + len(self.predictors) #* n_back

        self.param_ranges = ('all_unc', self.n_params)
        self.param_names  = ['bias'] + self.predictors

        self.use_only_first_n_mins = False #Set to number of minutes to only consider trials at start of session.

        self.pop_params = {'means' : np.zeros(self.n_params), 'SDs'   : 0.3}

        if not hasattr(self, 'trial_select'):
            self.trial_select = False

        self.calculates_gradient = True
  

    def _select_trials(self, session):
        return session.select_trials(self.trial_select, self.n_exclude, self.use_only_first_n_mins)

    def session_likelihood(self, session, params_T, eval_grad = False):

        bias = params_T[0]
        weights = params_T[1:]

        choices = session.CTSO['choices']

        if not hasattr(session,'predictors'):
            predictors = self._get_session_predictors(session) # Get array of predictors
        else:
            predictors = session.predictors

        assert predictors.shape[0] == session.n_trials,  'predictor array does not match number of trials.'
        assert predictors.shape[1] == len(weights), 'predictor array does not match number of weights.'

        if self.trial_select: # Only use subset of trials.
            trials_to_use = self._select_trials(session)
            choices = choices[trials_to_use]
            predictors = predictors[trials_to_use,:]

        # Evaluate session log likelihood.

        Q = np.dot(predictors,weights) + bias
        P = ru.logistic(Q)  # Probability of making choice 1
        Pc = 1 - P - choices + 2. * choices * P  

        session_log_likelihood = sum(ru.protected_log(Pc)) 

        # Evaluate session log likelihood gradient.

        if eval_grad:
            dLdQ  = - 1 + 2 * choices + Pc - 2 * choices * Pc
            dLdB = sum(dLdQ) # Likelihood gradient w.r.t. bias paramter.
            dLdW = sum(np.tile(dLdQ,(len(weights),1)).T * predictors, 0) # Likelihood gradient w.r.t weights.
            session_log_likelihood_gradient = np.append(dLdB,dLdW)
            return (session_log_likelihood, session_log_likelihood_gradient)
        else:
            return session_log_likelihood



# -------------------------------------------------------------------------------------
# Kernels only.
# -------------------------------------------------------------------------------------


class kernels_only(_session_based_log_reg):

    '''
    Equivilent to RL agent using only bias, choice kernel (stay), and second step kernel (side)
    '''

    def __init__(self):

        self.name = 'kernels_only'

        self.predictors = ['choice', 'side']

        _session_based_log_reg.__init__(self)


    def _get_session_predictors(self, session):
        '''Calculate and return values of predictor variables for all trials in session.
        '''
        
        choices, second_steps = ut.CTSO_unpack(session.CTSO, 'CS', float)

        predictors = np.array((choices, second_steps)).T - 0.5
        predictors = np.vstack((np.zeros(2),predictors[:-1,:]))  # First trial events predict second trial etc.

        return predictors

# -------------------------------------------------------------------------------------
# Configurable logistic regression Model.
# -------------------------------------------------------------------------------------

class config_log_reg(_session_based_log_reg):

    '''
    Configurable logistic regression agent. Arguments:

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags         - By default each predictor is only used at a lag of -1 (i.e. one trial predicting the next).
                 The lags argument is used to specify the use of additional lags for specific predictors:
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on the previous 3 trials
                 should be used as predictors, while the choices on the previous 2 trials should be used

    norm         - Set to True to normalise predictors such that each has the same mean absolute value.

    orth        - The orth argument is used to specify an orthogonalization scheme.  
                 orth = [('trans_CR', 'choice'), ('trCR_x_out', 'correct')] will orthogonalize trans_CR relative
                 to 'choice' and 'trCR_x_out' relative to 'correct'.

    mov_ave_CR   - Specifies whether transitions are classified common or rare based on block structue (False)
                 or based on a moving average of recent choices (True).

    trial_select - If mov_ave_CR set to false, trial select controls how many trials following reversals in the 
                 Transition matrix are excluded from the analsis. 
    '''


    def __init__(self, predictors = ['side', 'side_x_out', 'correct','choice','outcome','trans_CR', 'trCR_x_out'],
                lags = {}, norm = False, orth = False, n_exclude = 20, mov_ave_CR = False):

        self.name = 'config_lr'
        self.base_predictors = predictors # predictor names ignoring lags.
        self.orth = orth 
        self.norm = norm

        self.predictors = [] # predictor names including lags.
        for predictor in self.base_predictors:
            if predictor in lags.keys():
                for i in range(lags[predictor]):
                    self.predictors.append(predictor + '-' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_predictors = len(self.predictors)

        self.mov_ave_CR = mov_ave_CR 

        if self.mov_ave_CR: # Use moving average of recent transitions to evaluate 
            self.tau = 10.  # common vs rare transitions.
            self.trial_select = False
        else:               # Use block structure to evaluate common vs rare transitions. 
            self.trial_select = 'mor'
            self.n_exclude = n_exclude

        _session_based_log_reg.__init__(self)

    def _get_session_predictors(self, session):
        '''Calculate and return values of predictor variables for all trials in session.
        '''

        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions_AB, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, dtype = bool)
        trans_state = session.blocks['trial_trans_state']    # Trial by trial state of the tranistion matrix (A vs B)

        if self.mov_ave_CR:
            trans_mov_ave = np.zeros(len(choices))
            trans_mov_ave[1:] = (5./3.) * ut.exp_mov_ave(transitions_AB - 0.5, self.tau, 0.)[:-1] # Average of 0.5 for constant 0.8 transition prob.
            transitions_CR = 2 * (transitions_AB - 0.5) * trans_mov_ave
            transition_CR_x_outcome = 2. * transitions_CR * (outcomes - 0.5) 
            choices_0_mean = 2 * (choices - 0.5)
        else:  
            transitions_CR = transitions_AB == trans_state
            transition_CR_x_outcome = transitions_CR == outcomes 

        bp_values = {} 

        for p in self.base_predictors:

            if p == 'correct':  # 0.5, 0, -1 for high poke being correct, neutral, incorrect option.
                bp_values[p] = 0.5 * (session.blocks['trial_rew_state'] - 1) * \
                              (2 * session.blocks['trial_trans_state'] - 1)  
      
            elif p == 'side': # 0.5, -0.5 for left, right side reached at second step. 
                bp_values[p] = second_steps - 0.5

            elif p == 'side_x_out': # 0.5, -0.5.  Side predictor invered by trial outcome.
                bp_values[p] = (second_steps == outcomes) - 0.5

            # The following predictors all predict stay probability rather than high vs low.
            # e.g the outcome predictor represents the effect of outcome on stay probabilty.
            # This is implemented by inverting the predictor dependent on the choice made on the trial.

            elif p ==  'choice': # 0.5, - 0.5 for choices high, low.
                bp_values[p] = choices - 0.5

            elif p == 'good_side': # 0.5, 0, -0.5 for reaching good, neutral, bad second link state.
                bp_values[p] = 0.5 * (session.blocks['trial_rew_state'] - 1) * (2 * (second_steps == choices) - 1)
                    
            elif p ==  'outcome': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = (outcomes == choices) - 0.5

            elif p ==  'block':     # 0.5, -0.5 for A , B blocks.
                bp_values[p] = (trans_state == choices) - 0.5

            elif p == 'block_x_out': # 0.5, -0.5 for A , B blocks inverted by trial outcome.
                bp_values[p] = ((outcomes == trans_state) == choices) - 0.5

            elif p ==  'trans_CR': # 0.5, -0.5 for common, rare transitions.     
                if self.mov_ave_CR:            
                    bp_values[p] = transitions_CR * choices_0_mean 
                else: 
                    bp_values[p] = ((transitions_CR) == choices)  - 0.5

            elif p == 'trCR_x_out': # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
                if self.mov_ave_CR: 
                    bp_values[p] = transition_CR_x_outcome * choices_0_mean 
                else:
                    bp_values[p] = (transition_CR_x_outcome  == choices) - 0.5

            elif p ==  'trans_CR_rew': # 0.5, -0.5, for common, rare transitions on rewarded trials, otherwise 0.
                    if self.mov_ave_CR: 
                        bp_values[p] = transitions_CR * choices_0_mean * outcomes
                    else: 
                        bp_values[p] = (((transitions_CR) == choices)  - 0.5) * outcomes

            elif p ==  'trans_CR_non_rew': # 0.5, -0.5, for common, rare transitions on non-rewarded trials, otherwise 0.
                    if self.mov_ave_CR: 
                        bp_values[p] = transitions_CR * choices_0_mean * ~outcomes
                    else: 
                        bp_values[p] = (((transitions_CR) == choices)  - 0.5) * ~outcomes

        # predictor orthogonalization.

        if self.orth: 
            for A, B in self.orth: # Remove component of predictor A that is parrallel to predictor B. 
                bp_values[A] = bp_values[A] - ut.projection(bp_values[B], bp_values[A])

        # predictor normalization.
        if self.norm:
            for p in self.base_predictors:
                bp_values[p] = bp_values[p] * 0.5 / np.mean(np.abs(bp_values[p]))

        # Generate lagged predictors from base predictors.

        predictors = np.zeros([session.n_trials, self.n_predictors])

        for i,p in enumerate(self.predictors):  
            if '-' in p: # Get lag from predictor name.
                lag = int(p.split('-')[1]) 
                bp_name = p.split('-')[0]
            else:        # Use default lag.
                lag = 1
                bp_name = p
            predictors[lag:, i] = bp_values[bp_name][:-lag]

        return predictors
















