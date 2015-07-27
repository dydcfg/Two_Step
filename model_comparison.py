import numpy as np
import RL_utils as ru
import RL_plotting as rp
import plotting as pl
import session as ss
import time
from copy import deepcopy
import utility as ut
import model_fitting as mf
import pylab as p
from functools import partial
from multiprocessing import Pool

#mp_pool = Pool(4)

p.ion()
p.rcParams['pdf.fonttype'] = 42

# -------------------------------------------------------------------------------------
# Model comparison.
# -------------------------------------------------------------------------------------

def BIC_model_comparison(population_fits):
    ''' Compare goodness of different fits using integrated BIC'''    
    sorted_fits = sorted(population_fits, key = lambda fit: fit['BIC_score'])
    print 'BIC_scores:'
    for fit in sorted_fits:
        print '{} : '.format(round(fit['BIC_score'])) + fit['agent_name']
    print 'The best fitting model is: ' + sorted_fits[0]['agent_name']
 
def eval_calibration(sessions, agent, population_fit, use_MAP = True, n_bins = 10, fixed_widths = False, to_plot = False):
    '''Caluculate real choice probabilities as function of model choice probabilities.'''

    session_fits = population_fit['MAP_fits']

    assert len(session_fits[0]['params_T']) == agent.n_params, 'agent n_params does not match population_fit.'
    assert len(sessions) == len(session_fits), 'Number of fits does not match number of sessions.'
    assert population_fit['agent_name'] == agent.name, 'Agent name different from that used for fits.'

    # Create arrays containing model choice probabilites and true choices for each trial.
    session_choices, session_choice_probs = ([],[])
    for fit, session in zip(session_fits, sessions):
        if use_MAP:
            params_T = fit['params_T']
        else:
            params_T = ru.sample_params_T_from_pop_params(population_fit['pop_params'], agent) 
        session_choices.append(session.CTSO['choices'].tolist())
        session_choice_probs.append(agent.session_likelihood(session, params_T,
                                                     return_trial_data = True)['choice_probs'])
    choices = np.hstack(session_choices)
    choice_probs = np.vstack(session_choice_probs)[:, 1]

    # Calculate true vs model choice probs.
    true_probs  = np.zeros(n_bins)
    model_probs = np.zeros(n_bins)
    if fixed_widths: # Bins of equal width in model choice probability.
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
    else: # Bins of equal trial number.
        choices = choices[np.argsort(choice_probs)]
        choice_probs.sort()
        bin_edges = choice_probs[np.round(np.linspace(0,len(choice_probs) - 1, n_bins + 1)).astype(int)]
        bin_edges[0] = 0.
    for b in range(n_bins):
        true_probs[b] = np.mean(choices[np.logical_and(
                            bin_edges[b] < choice_probs,
                            choice_probs <= bin_edges[b + 1])])
        model_probs[b] = np.mean(choice_probs[np.logical_and(
                            bin_edges[b] < choice_probs,
                            choice_probs <= bin_edges[b + 1])])
        calibration = {'true_probs': true_probs, 'model_probs': model_probs}
    if to_plot: rp.calibration_plot(calibration)
    print('Fraction correct: {}'.format(sum((choice_probs > 0.5) == choices.astype(bool)) / float(len(choices))))
    chosen_probs = np.hstack([choice_probs[choices == 1], 1. - choice_probs[choices == 0]])
    print('Geometric mean choice prob: {}'.format(np.exp(np.mean(np.log(chosen_probs)))))
    return calibration


def model_comparison_robustness(sessions, agents, task, n_eval = 100, n_sim = 100):
    ''' Model comparison includeing an estimation of how robust is the conlusion about
    which model is best.
    The approach taken is as follows:
    1. Evaluate the quality off fit of the models to the data provided using the
    specified metric (e.g. BIC score)
    2. Using the best fitting model generate a population of simulated datasets, each of
    which is the same size as the real dataset.
    3. Fit all model to each simulated dataset and evaluate the BIC scores for the fit.
    4. Plot the distibutions of BIC scores for each model, and the distribution of BIC 
    score difference between the best fitting model and each other model.
    '''
    print('Fitting real data.')
    model_fits = [mf.fit_population(sessions, agent, eval_BIC = n_eval) for agent in agents]
    best_agent_n = np.argmin([fit['BIC_score'] for fit in model_fits])
    best_agent = agents[best_agent_n]
    best_agent_fit =  model_fits[best_agent_n]
    simulated_datasets = []
    for i in range(n_sim):
        simulated_datasets.append(sim_sessions_from_pop_fit(task, best_agent,
                                                               best_agent_fit, use_MAP = False))

    # simulated_data_fits, i, n_fits = ([], 1, len(agents) * n_sim )
    # for agent in agents:
    #     agent_simdata_fits = []
    #     init_params = None 
    #     for sim_data in simulated_datasets:
    #         print('Simulated dataset fit {} of {}'.format(i, n_fits))
    #         agent_simdata_fits.append(mf.fit_population(sim_data, agent, 
    #                                   eval_BIC = n_eval, pop_init_params = init_params))
    #         init_params = agent_simdata_fits[-1]['pop_params'] 
    #         i += 1
    #     simulated_data_fits.append(agent_simdata_fits)

    fit_func = partial(fit_agent_to_sim_data, simulated_datasets = simulated_datasets, n_eval = n_eval)
    simulated_data_fits = mp_pool.map(fit_func, agents)

    mod_comp = {'agents'              : agents,
                'sessions'            : sessions,
                'task'                : task,
                'best_agent_n'        : best_agent_n,
                'model_fits'          : model_fits,
                'simulated_datasets'  : simulated_datasets,
                'simulated_data_fits' : simulated_data_fits}

    plot_BIC_dists(mod_comp)

    return mod_comp

def fit_agent_to_sim_data(agent, simulated_datasets, n_eval):

    agent_simdata_fits = []
    init_params = None 
    for sim_data in simulated_datasets:
        agent_simdata_fits.append(mf.fit_population(sim_data, agent, 
                                  eval_BIC = n_eval, pop_init_params = init_params))
        init_params = agent_simdata_fits[-1]['pop_params'] 
    return agent_simdata_fits


def plot_BIC_dists(mod_comp, n_bins = 100):
    'Plot results of model comparison.'
    agents = mod_comp['agents']

    sim_data_BIC_scores = np.array([[fit['BIC_score'] for fit in agent_simdata_fits] for 
                                     agent_simdata_fits in mod_comp['simulated_data_fits']])


    BIC_diffs = sim_data_BIC_scores - np.tile(sim_data_BIC_scores[mod_comp['best_agent_n'],:],(len(agents),1))

    BIC_score_range = (sim_data_BIC_scores.min() - 1, sim_data_BIC_scores.max() + 1)
    BIC_diffs_range = (BIC_diffs.min() - 1, BIC_diffs.max() + 1)
    cols = p.cm.rainbow(np.linspace(0,1,len(agents)))
    p.figure(1)
    p.clf()
    p.subplot(2,1,1)
    for i, agent in enumerate(agents):
        p.hist(sim_data_BIC_scores[i,:], n_bins, BIC_score_range, color = cols[i],
               histtype='stepfilled', alpha = 0.5, label= agent.name)
    y_lim = p.ylim()
    for i, agent in enumerate(agents):
        p.plot([mod_comp['model_fits'][i]['BIC_score']],[y_lim[1]/2.],'o', color = cols[i])
    p.ylim(np.array(y_lim)*1.1)
    p.legend()
    p.xlabel('BIC score')
    p.subplot(2,1,2)
    for i, agent in enumerate(agents):
        if not BIC_diffs[i,0] == 0:
            p.hist(BIC_diffs[i,:], n_bins, BIC_diffs_range, color = cols[i],
                   histtype='stepfilled', alpha = 0.5, label= agent.name)
    p.ylim(np.array(p.ylim())*1.1)
    p.xlabel('BIC score difference')


def plot_fit_consistency(population_fits, plot_true = True, fig_no = 1):

    fit_means = np.array([pf['pop_params']['means'] for pf in population_fits])
    fit_SDs = np.array([pf['pop_params']['SDs'] for pf in population_fits])
    true_means = population_fits[0]['pop_params_true']['means']
    true_SDs = population_fits[0]['pop_params_true']['SDs']
    n_params = fit_means.shape[1]
    n_fits = fit_means.shape[0]
    x = np.arange(n_fits)/float(n_fits)
    ymin = np.min(fit_means - fit_SDs) - 0.2
    ymax = np.max(fit_means + fit_SDs) + 0.2

    if not len(true_means) == n_params:
        plot_true = False

    p.figure(fig_no)
    p.clf()

    for i in range(n_params):
        p.subplot(1, n_params, i + 1)
        if plot_true:
            p.plot([0.45,0.45], [true_means[i] - true_SDs[i], true_means[i] + true_SDs[i]], 'r', linewidth = 2)         
        for f in range(n_fits):
            p.plot([x[f],x[f]], [fit_means[f,i] - fit_SDs[f,i], fit_means[f,i] + fit_SDs[f,i]],'b')
            p.locator_params(axis = 'y', nbins = 4)
            p.xticks([])
            p.xlabel(population_fits[0]['param_names'][i])
            p.ylim(ymin,ymax)


# -------------------------------------------------------------------------------------
# Simulated data generation.
# -------------------------------------------------------------------------------------

class simulated_session():
    '''Stores agent parameters and simulated data, supports plotting as for experimental
    session class.
    '''
    def __init__(self, task, agent, n_trials = 1000):
        '''Simulate session with current agent and task parameters.'''
        agent.reset()
        task.reset(n_trials)
        self.param_names = agent.param_names
        self.pop_params_true = agent.pop_params
        self.true_params_T = agent.get_params_T()
        try: # Not possible for e.g. unit range params_T with value 0 or 1.
            self.true_params_U = agent.get_params_U()
        except Exception: 
            pass
        self.reward_probs = task.reward_probs
        self.n_trials = n_trials
        self.all_trial_events = []

        while not task.end_session:
            first_link = agent.choose()
            second_step, outcome = task.trial(first_link)
            agent.update_action_values(first_link, second_step, outcome)
            self.all_trial_events.append((first_link, second_step, outcome))

        if hasattr(task,'blocks'):
            self.blocks = deepcopy(task.blocks)
        
        # Convert trial events to CTSO representation.
        f, s, o = zip(*all_trial_events)
        self.CTSO = {'choices'      : np.array(f, int),
                     'transitions'  : ((np.array(s)-np.array(f)) == 2).astype(int),
                     'second_steps' : np.array(s, int) - 2,
                     'outcomes'     : np.array(o, int)}

    def plot(self):pl.plot_session(self)

    def select_trials(self, selection_type = 'inc', last_n = 20, first_n_mins = False,
                      block_type = 'all'):
        return ss.select_trials(self, selection_type, last_n, first_n_mins, block_type)
            
def simulate_sessions(agent, task,  n_sessions = 10, n_trials = 1000,
                      pop_params = None, randomize_params = True):
    '''Simulate a population of sessions.  If list of integers is passed as the n_trials argument,
    a set of sessions with number of trials given by the list elements is simulated, 
    overriding the n_sessions argument.
    By default agent parameters are randomised for each session using agent.randomize_params().
    If randomise_params is False and pop_params are provided, the pop_params means are used as the 
    agent parameters.'''

    sessions = []
    if pop_params:
        agent.pop_params = pop_params
    if not type(n_trials) == list:
        n_trials = [n_trials] * n_sessions
    for n_t in n_trials:
        if randomize_params:
            agent.randomize_params()
        elif pop_params:
            agent.set_params_U(pop_params['means'])
        sessions.append(simulated_session(task, agent, n_t))
    return sessions

def test_population_fitting(task, agent, n_sessions = 8, n_trials = 1000, pop_params = None):
    '''Simulate a set of sessions using parameters drawn from normal distributions
    specified by pop_params.  Then fit the agent model to the simulated data and plot
    correspondence between true and fitted paramter values.
    '''
    sessions = simulate_sessions(task, agent, n_sessions, n_trials, pop_params)
    ML_fits, MAP_fits, pop_params = mf.fit_population(sessions, agent, max_iter = 15)
    rp.plot_true_fitted_params(sessions, ML_fits, MAP_fits)
    return (sessions, ML_fits, MAP_fits, pop_params)

def sim_sessions_from_pop_fit(task,agent,population_fit, use_MAP = False, enlarge = 1):
    '''Simulate sessions using parameter values from population fit.
    If use_MAP is true, simulated sessions use the MAP paramter values,
    otherwise parameter values are drawn randomly from the population 
    level distributions.  The number of trials in the simulated sessions
    matches those in the orginal dataset.
    The enlarge parameter can be used to produce larger datasets by simulating multiple 
    sessions for each session in the real data set.
    '''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    agent.pop_params = {'SDs'  : population_fit['pop_params']['SDs'],
                        'means': population_fit['pop_params']['means']}
    sessions = []
    for i in range(enlarge):
        for n_trials, MAP_fit in zip(population_fit['n_trials'],
                                     population_fit['MAP_fits']):
            if use_MAP:
                agent.set_params_U(MAP_fit['params_U'])
            else:
                agent.randomize_params()
            sessions.append(simulated_session(task, agent, n_trials))
    return sessions
