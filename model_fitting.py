import numpy as np
import RL_utils as ru
import RL_plotting as rp
import plotting as pl
import model_comparison as mc
import scipy.optimize as op
import time
from functools import partial
from copy import deepcopy
import utility as ut
from multiprocessing import Pool, cpu_count

n_cpu = cpu_count()

# -------------------------------------------------------------------------------------
# Fitting
# -------------------------------------------------------------------------------------

def per_subject_fit(sessions, agent, max_change = 0.01, parallel = True):
    ''' Fit agent model seperately to sessions from each subject. Returns list
    of subject fits, each of which is a population_fit.
    '''
    subject_fits = []
    sIDs = sorted(set([s.subject_ID for s in sessions]))
    for sID in sIDs:
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        subject_fit = fit_population(subject_sessions, agent, max_change = 0.01, parallel = True)
        subject_fit['sID'] = sID
        subject_fits.append(subject_fit)
    return subject_fits


def longitudinal_fit(experiment, agent, epoch_len = 4, max_iter = 15):
    '''Perform population fitting longditudinally through an experiment doing 
    population_fit on non-overlapping epochs each containing epoch_len days.'''
    epoch_start_days = range(1,experiment.n_days, epoch_len)
    epoch_fits = []
    for start_day in epoch_start_days:
        epoch_sessions = experiment.get_sessions('all',range(start_day,start_day + epoch_len))
        population_fit = fit_population(epoch_sessions, agent, max_iter)
        population_fit['start_day'] = start_day
        epoch_fits.append(population_fit)
    return epoch_fits


def fit_population(sessions, agent, max_iter = 200, min_iter = 2, repeats = 5, pop_init_params = None,
                   eval_BIC = False, eval_calib = False, verbose = False, max_change = 0.005,
                   parallel = True):
    ''' Fits population level parameters using the Expectation Maximisation 
    method from Huys et al. 
    '''   

    start_time = time.time()

    for i, session in enumerate(sessions):session.number = i #assign session ID number (used for error messages).

    # If agent selects subset of trials for fitting, exclude sessions for which agent 
    # trial selection criterion does not select any trials.   

    if hasattr(agent,'_select_trials') and bool(agent.trial_select):
        n_ses = len(sessions)
        sessions = [s for s in sessions if sum(agent._select_trials(s)) > 0]
        n_excluded = n_ses - len(sessions)
    else:
        n_excluded = 0
        
    #param_evol variable stores values of the population parameters as they evolve during the EM fitting.
    param_evol = {'means'   : [], 'vars'        : [], 'liks'        : [],
                  'means_T' : [], 'upper_conf_T': [], 'lower_conf_T': [],
                  'means_U' : [], 'upper_conf_U': [], 'lower_conf_U': []} 

    # Initialise prior for first round of MAP.
    if pop_init_params: 
        print('Initial paramter estimate provided.')
        pop_means = pop_init_params['means']
        pop_vars  = pop_init_params['SDs']**2
    else:
        pop_means =   np.zeros(agent.n_params)
        pop_vars  =   np.ones(agent.n_params) * 6.25


    if parallel: # Set up parallel processing pool of workers.        
        if parallel == True:
            p = Pool(n_cpu)
        else:
            p = Pool(parallel)

    if hasattr(agent,'_get_session_predictors'):
        # Precalculate predictors for logistic regression agents, note:
        # not currently using multiprocessing as can't pickle instancemethod.
        for session in sessions:
            session.predictors = agent._get_session_predictors(session)

    k  =  1
    while k <= max_iter:

        # E - Step: Evaluate the new posterior distribution of the subject 
        # parameters used to calculate expectation of log likelihood.

        print('\nMaximum a posteriori fitting... Round: ' + str(k)),

        MAP_fit_func = partial(fit_session, agent = agent , pop_means = pop_means,
                                 pop_vars = pop_vars, repeats = repeats, verbose = verbose)

        if parallel:
            MAP_fits = p.map(MAP_fit_func, sessions)
        else:
            MAP_fits = map(MAP_fit_func, sessions)

        for session, fit in zip(sessions, MAP_fits):
            session.init_params_U = fit['params_U'] # Store session fits as inital parameters for next round of fitting.


        pop_likelihood = np.sum(np.array([fit['likelihood']  for fit in MAP_fits]))
        sub_params_U  = np.array([fit['params_U']  for fit in MAP_fits])  
        sub_diag_hess = np.array([fit['diag_hess'] for fit in MAP_fits])

        sub_diag_hess[sub_diag_hess > -1e-15] = -1e-15 # Set any hessians that are 0 to small negative 
                                                       # value to prevent divide by zero error.

        # M - step: Adjust population distribution mean and variance to 
        # maximise the expectation of the log likelihood.

        pop_means = np.mean(sub_params_U, 0)
        pop_vars =  np.mean(sub_params_U ** 2 +  1. / -sub_diag_hess, 0) - pop_means ** 2.
        
        if verbose: # Print info about hessians as sanity check.
            max_index = np.unravel_index(np.argmax(sub_diag_hess),np.shape(sub_diag_hess))
            print('Maximum sub_diag_hess: {:.4} , at session: {}, param: {}, param value U: {:.4}'
             .format(np.max(sub_diag_hess), max_index[0], max_index[1], sub_params_U[max_index]))

        # Store information about population parameter evolution.

        param_evol['means'].append(pop_means)
        param_evol['vars'].append(pop_vars)
        param_evol['liks'].append(pop_likelihood)

        err_bar_2_SD = 2. * np.sqrt(pop_vars)    
        param_evol['means_T'].append(ru.trans_UT(pop_means, agent.param_ranges))        
        param_evol['upper_conf_T'].append(ru.trans_UT(pop_means + err_bar_2_SD, agent.param_ranges))
        param_evol['lower_conf_T'].append(ru.trans_UT(pop_means - err_bar_2_SD, agent.param_ranges))

        #Test for convergence: Evaluate maximum change in mean and 95% confidence intervals in true space. 
        if k >= min_iter:             
            max_param_change = np.max(np.abs([param_evol['means_T'][-1 ]     - param_evol['means_T'][-2],
                                              param_evol['upper_conf_T'][-1] - param_evol['upper_conf_T'][-2],
                                              param_evol['lower_conf_T'][-1] - param_evol['lower_conf_T'][-2]]))
            print('Max change T: {:.4}'.format(max_param_change)),
            if max_param_change < max_change:
                print ('\nEM fitting Converged.')
                break

        k += 1
        repeats = 1 # Only use multiple initial values for gradient descent on first round, then initialise with previous values.

    print('Elapsed time: ' + str(time.time() - start_time))

    if parallel:p.close()

    for session, MAP_fit in zip(sessions, MAP_fits):
        del session.init_params_U          # Remove fitting initial params from sessions.
        if hasattr(session, 'subject_ID'):
            MAP_fit['sID'] = session.subject_ID # Record animal ID on MAP fits.

    pop_params = {'means' : pop_means, 'SDs' : np.sqrt(pop_vars), 'ranges': agent.param_ranges}
    if hasattr(agent,'param_names'): pop_params['names'] = agent.param_names

    population_fit = {'MAP_fits'    : MAP_fits,
                      'pop_params'  : pop_params,
                      'param_evol'  : param_evol,
                      'agent_name'  : agent.name,
                      'param_names' : agent.param_names,
                      'n_trials'    :[s.n_trials for s in sessions]}

    if hasattr(sessions[0], 'true_params_T'):
        # If simulated data, store true values of parameters in population_fit
        population_fit['true_values'] = {'params_U' : [session.true_params_U for session in sessions],
                                         'params_T' : [session.true_params_T for session in sessions]} 
        population_fit['pop_params_true'] = sessions[0].pop_params_true
        population_fit['param_names_true'] = sessions[0].param_names
    
    if eval_calib & hasattr(agent, 'trial_choice_prob'):
        print('Evaluating calibration')
        population_fit['calibration'] = mc.eval_calibration(sessions, agent, population_fit)
 
    if eval_BIC:  
         if eval_BIC == True: eval_BIC = 100 # Default to 100 draws if True passed in as eval_BIC argument.
         print('Evaluating integrated BIC score, n_draws: {}'.format(eval_BIC))
         BIC_score, integrated_likelihood, choice_prob = evaluate_BIC(sessions, agent, pop_params, eval_BIC, parallel)
         population_fit['BIC_score'] = BIC_score
         population_fit['choice_prob'] = choice_prob
         population_fit['integrated_likelihood'] = integrated_likelihood
         population_fit['MAP_likelihood'] = evaluate_MAP_fit_likelihood(population_fit,sessions,agent)

    if hasattr(sessions[0],'predictors'):    
        for session in sessions:
            del session.predictors

    if n_excluded > 0:
            print('{} of {} sessions excluded by agent trial selection criterion.'.format(n_excluded, n_ses))

    return population_fit

def fit_session(session, agent, pop_means = None, pop_vars = None, repeats = 3, verbose = False):
    '''Find maximum a posteriori parameter values for agent for given session and means and variances of 
    population level prior distributions. '''

    if pop_means is None: # No prior passed in, use (almost completely) uninformative prior.
        pop_means = np.zeros(agent.n_params)
        pop_vars  = np.ones(agent.n_params) * 100.

    use_init_params = hasattr(session, 'init_params_U')

    fit_func = lambda params_U: session_log_posterior(params_U, session, agent, pop_means, pop_vars, sign = - 1.)

    good_fit_found = False
    while not good_fit_found: # Check based on positive values of hessian.

        fits = []
        for i in xrange(repeats): # Perform fitting. 
            if use_init_params:
                init_params_U = session.init_params_U
            else:
                init_params_U = ru.random_params(agent.param_ranges)
            
            fits.append(op.minimize(fit_func, init_params_U, jac = agent.calculates_gradient,
                                    options = {'disp': verbose, 'gtol': 1e-7}))   
            use_init_params = False # Only use provided initial parameters for first repeat.

        fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.
        hess_func = lambda params_U: session_log_posterior(params_U, session, agent, pop_means, pop_vars, sign = 1., eval_grad = False)

        hessdiag = ru.Hess_diag(hess_func, fit['x'], 1e-4)

        if np.max(hessdiag) > 0.:
            print('Bad fit. Repeating.')
        else:
            good_fit_found = True

    session_fit = {'params_U'   : fit['x'],
                   'params_T'   : ru.trans_UT(fit['x'], agent.param_ranges),
                   'likelihood' : - fit['fun'], 
                   'diag_hess'  : hessdiag} 

    return session_fit

def session_log_posterior(params_U, session, agent, pop_means, pop_vars, eval_grad = True, sign = 1.):
    '''Evaluates the log posterior probability of behaviour in a single session 
    for a given set of parameter values and population level mean and variances.
    '''

    params_T = ru.trans_UT(params_U, agent.param_ranges)

    log_prior_prob = - (len(params_U) / 2.) * np.log(2 * np.pi) - np.sum(np.log(pop_vars)) / 2. \
                     - sum((params_U - pop_means) ** 2. / (2 * pop_vars))

    if agent.calculates_gradient and eval_grad:
        log_likelihood, log_likelihood_gradient_T = agent.session_likelihood(session, params_T, eval_grad = True)

        log_likelihood_gradient_U = ru.trans_grad_TU(params_T, log_likelihood_gradient_T, agent.param_ranges)

        log_prior_prob_gradient = ((pop_means - params_U) / pop_vars)

        log_posterior_prob = log_likelihood + log_prior_prob
        log_posterior_grad = log_likelihood_gradient_U + log_prior_prob_gradient

        return (sign * log_posterior_prob, sign * log_posterior_grad)

    else:

        log_likelihood = agent.session_likelihood(session, params_T)

        log_posterior_prob = log_likelihood + log_prior_prob

        return (sign * log_posterior_prob)

def evaluate_MAP_fit_likelihood(population_fit, sessions, agent):
    data_log_likelihood = 0 
    for MAP_fit, session in zip(population_fit['MAP_fits'],sessions):
        data_log_likelihood += agent.session_likelihood(session, MAP_fit['params_T'])
    return data_log_likelihood


def evaluate_BIC(sessions, agent, pop_params, n_draws = 100, parallel = True):
    '''Return the integrated BIC score for given agent model, sessions
     & agent population parameter distribution.'''

    if 'pop_params' in pop_params.keys(): # allow population fit to be passed in instead of pop_params
        pop_params = pop_params['pop_params']

    int_func = partial(_session_integrated_log_likelihood, agent = agent, pop_params = pop_params, n_draws =  n_draws)

    if parallel:
        p = Pool(n_cpu)
        integrated_likelihood = np.sum(p.map(int_func, sessions))
        p.close()
    else:
        integrated_likelihood = np.sum(map(int_func, sessions))
    
    n_params = 2 * agent.n_params # Factor of 2 as each agent param has mean and variance as population params.
    if hasattr(agent,'_select_trials') and bool(agent.trial_select):
        n_trials = sum([sum(agent._select_trials(s)) for s in sessions])
    else:
        n_trials = sum([s.n_trials for s in sessions])

    BIC_score = - 2. * integrated_likelihood + n_params * np.log(n_trials)

    choice_prob = np.exp(integrated_likelihood/n_trials)
    return (BIC_score, integrated_likelihood, choice_prob)

def _session_integrated_log_likelihood(session, agent, pop_params, n_draws):
    '''Estimate integrated log likelihood for one session by importance sampling.'''
    log_likelihood_samples = np.zeros(n_draws)
    for i in range(n_draws):
        sample_params_T = ru.sample_params_T_from_pop_params(pop_params, agent)
        log_likelihood_samples[i] = agent.session_likelihood(session, sample_params_T)
    # Subtract maximum log likelihood sample from each sample before taking exponent to
    # protect against exponent returning zero for large negative log likelihoods.
    max_log_lik_sample = max(log_likelihood_samples)   
    session_int_log_lik = max_log_lik_sample + \
                          np.log(np.sum(np.exp(log_likelihood_samples - \
                                               max_log_lik_sample))/n_draws)
    return session_int_log_lik


def grad_check(session, agent, params_T = [], verbose = True):
    'Check analytical likelihood gradient returned by agent.'
    if len(params_T) == 0:
        params_T = ru.random_params(agent.param_ranges, return_T = True)
    fit_func  = lambda params_T: agent.session_likelihood(session, params_T, eval_grad = True)
    lik_func  = lambda params_T: fit_func(params_T)[0]
    grad_func = lambda params_T: fit_func(params_T)[1]
    l2error = op.check_grad(lik_func, grad_func, params_T)
    if verbose:
        print('Error between finite difference and analytic derivatives = ' + str(l2error))
        if l2error > 1e-3:
            print('Params_T: {}'.format(agent.get_params_T()))
    else:
        return(l2error, params_U)


