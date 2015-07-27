from __future__ import division, print_function
import numpy as np
import model_fitting as mf
from random import shuffle, choice
import RL_plotting as rp
import plotting as pl
import pylab as p
from scipy.special import binom
from scipy.stats import ttest_ind, ttest_rel
from RL_agents import _RL_agent
import time
import sys

p.ion()
p.rcParams['pdf.fonttype'] = 42

def group_info(sessions):
    return {'n_subjects'  : len(set([s.subject_ID for s in sessions])),
            'n_sessions' : len(sessions),
            'n_blocks'   : sum([len(s.blocks['start_trials']) - 1 for s in sessions]),
            'n_trials'   : sum([s.n_trials for s in sessions])}


def run_tests(sessions_A, sessions_B, RL_agent, LR_agent, perm_type, title,
                   max_change_LR = 0.01, max_change_RL = 0.05, n_resample = 1000,
                   test_time = 20, parallel = False, save_exp = None, test_data = None):
    ''' Run a suite of different comparisons on two groups of sessions.'''

    if not test_data:
        test_data = {'title'       : title,
                     'group_A_info': group_info(sessions_A),
                     'group_B_info': group_info(sessions_B),
                     'perm_type'   : perm_type}

        test_data['trial_rate'] =  trial_rate_test(sessions_A, sessions_B, perm_type, test_time, n_resample)
        # Reversal analysis.
        test_data['reversal_analysis'] = reversal_test(sessions_A, sessions_B, perm_type, n_resample, by_type = False)
        if save_exp: 
            save_exp.save_item(test_data, title + '_test_data')
            print_test_data(test_data, save_exp)
    
        test_data['LR_fit'], test_data['RL_fit'] = (None, None)
        
    rs_chunk = 5
    for i in range(int(n_resample/rs_chunk)):
        # logistic regression analysis.  
        test_data['LR_fit'] = model_fit_test(sessions_A, sessions_B, LR_agent, perm_type, rs_chunk,
                                             max_change_LR, parallel = parallel, mft = test_data['LR_fit'])
        if save_exp: 
            save_exp.save_item(test_data, title + '_test_data')
            print_test_data(test_data, save_exp)
        # RL model fitting analysis.
        test_data['RL_fit'] = model_fit_test(sessions_A, sessions_B, RL_agent, perm_type, rs_chunk,
                                             max_change_RL, parallel = parallel, mft = test_data['RL_fit'])
        if save_exp: 
            save_exp.save_item(test_data, title + '_test_data')
            print_test_data(test_data, save_exp)
    print_test_data(test_data)
    return test_data


def print_test_data(test_data, save_exp = None):
    if save_exp:
        f = open(save_exp.path + test_data['title'] +  '_test.txt', 'w+')
    else:
        f=sys.stdout
    if 'group_A_info' in test_data.keys():
        print('\nGroup A info:', file = f)
        print(test_data['group_A_info'], file = f)
        print('\nGroup B info:', file = f)
        print(test_data['group_B_info'], file = f)
    if 'trial_rate' in test_data.keys():
        print('\nP value for number of trials in first {} minutes: {}'
              .format(test_data['trial_rate']['test_time'],
                      test_data['trial_rate']['p_val']), file = f)
    if 'reversal_analysis' in test_data.keys():    
        print('\nReversal analysis P values: P_0: {}, tau: {}'
              .format(test_data['reversal_analysis']['block_end_P_value'],
                      test_data['reversal_analysis']['tau_P_value']), file = f)
    if 'LR_fit' in test_data.keys():   
        print('\nLogistic regression fit P values, {} permutations:'
              .format(test_data['LR_fit']['n_resample']), file = f)
        for param_name, p_val in zip(test_data['LR_fit']['fit_A']['param_names'],
                                     test_data['LR_fit']['means_data']['p_vals']):
            print('{} : {}'.format(param_name, p_val), file = f)
        if test_data['RL_fit']:       
            print('\nRL fit P values, {} permutations:'
                  .format(test_data['RL_fit']['n_resample']), file = f)
            for param_name, p_val in zip(test_data['RL_fit']['fit_A']['param_names'],
                                         test_data['RL_fit']['means_data']['p_vals']):
                print('{} : {}'.format(param_name, p_val), file = f)
            if 'pref_data' in test_data['RL_fit'].keys():
                print('\nPreference P values: MB: {},  TD: {}'
                      .format(test_data['RL_fit']['pref_data']['p_vals'][0],
                              test_data['RL_fit']['pref_data']['p_vals'][1]), file = f)
    if save_exp: f.close()

def plots(sessions_A, sessions_B, RL_agent, LR_agent = None, title = None,
                   max_change_LR = 0.001, max_change_RL = 0.01, 
                   test_time = 20, parallel = False, test_data = None):
    if test_data:
        RL_fit_A = test_data['RL_fit']['fit_A']
        RL_fit_B = test_data['RL_fit']['fit_B']
        LR_fit_A = test_data['LR_fit']['fit_A']
        LR_fit_B = test_data['LR_fit']['fit_B']
        title = test_data['title']
    else:
        RL_fit_A = mf.fit_population(sessions_A, RL_agent, max_change = max_change_RL, parallel = parallel)
        RL_fit_B = mf.fit_population(sessions_B, RL_agent, max_change = max_change_RL, parallel = parallel)
        LR_fit_A = mf.fit_population(sessions_A, LR_agent, max_change = max_change_LR, parallel = parallel)
        LR_fit_B = mf.fit_population(sessions_B, LR_agent, max_change = max_change_LR, parallel = parallel)

    trial_rate_comparison(sessions_A, sessions_B, test_time, 1, title)
    reversal_comparison(sessions_A, sessions_B,  2, title)
    rp.scatter_plot_comp(LR_fit_A, LR_fit_B, fig_no = 3)
    p.title(title)
    rp.pop_fit_comparison(RL_fit_A, RL_fit_B, fig_no = 4, normalize = False)
    p.suptitle(title)
    abs_preference_comparison(sessions_A, sessions_B, RL_fit_A, RL_fit_B, RL_agent, 5, title)


def estimate_test_time(sessions_A, sessions_B, RL_agent, LR_agent, perm_type,
                       max_change_LR = 0.01, max_change_RL = 0.05, n_test_perm = 3, parallel = False):
    '''Estimate time taken per permutation to run compare_groups.'''
    start_time = time.time()
    for i in range(n_test_perm):
        shuffled_ses_A, shuffled_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        mf.fit_population(shuffled_ses_A, RL_agent, eval_BIC = False, max_change = max_change_RL, parallel = parallel)
        mf.fit_population(shuffled_ses_B, RL_agent, eval_BIC = False, max_change = max_change_RL, parallel = parallel)
        mf.fit_population(shuffled_ses_A, LR_agent, eval_BIC = False, max_change = max_change_LR, parallel = parallel)
        mf.fit_population(shuffled_ses_B, LR_agent, eval_BIC = False, max_change = max_change_LR, parallel = parallel)
        pl.reversal_analysis(shuffled_ses_A, return_fits = True, by_type = False)
        pl.reversal_analysis(shuffled_ses_B, return_fits = True, by_type = False)
    print('Estimated time per permuation: ' + str((time.time() - start_time)/n_test_perm))

# -------------------------------------------------------------------------------------
# Group comparison plots.
# -------------------------------------------------------------------------------------

def fit_comparison(sessions_A, sessions_B, agent, fig_no = 1, title = None, max_change = 0.005):
    ''' Fit the two groups of sessions with the specified agent and plot the results on the same axis.
    '''
    fit_A = mf.fit_population(sessions_A, agent, max_change = max_change)
    fit_B = mf.fit_population(sessions_B, agent, max_change = max_change)
    rp.scatter_plot_comp(fit_A, fit_B, fig_no = fig_no)
    if title:p.title(title)

def trial_rate_comparison(sessions_A, sessions_B, test_time = None, fig_no = 1, title = None):
    '''
    Plot trials per minute for each group, and dashed vertical line at test time if specified.
    '''
    pl.trials_per_minute(sessions_A, col = 'b', fig_no = fig_no)
    pl.trials_per_minute(sessions_B, col = 'r', fig_no = fig_no, clf = False)
    if test_time:
        p.plot([test_time,test_time], p.ylim(),':k')
    if title:
        p.title(title)

def reversal_comparison(sessions_A, sessions_B,  fig_no = 1, title = None, groups = None):
    '''Plot choice trajectories around reversals for both groups.  
    '''
    pl.reversal_analysis(sessions_A, cols = 0, fig_no = fig_no, by_type = False)
    pl.reversal_analysis(sessions_B, cols = 1, fig_no = fig_no, by_type = False, clf = False)
    if title: p.title(title)

def p_correct_comparison(sessions_A, sessions_B, fig_no = 1, title = None):
    ''' Compare fraction of correct choices at end on non neutral blocks.  Plot shows 
    data point for each animal and population mean and SEM.
    '''
    p_corrects_A = pl.per_animal_end_of_block_p_correct(sessions_A, col = 'b', fig_no = fig_no)
    p_corrects_B = pl.per_animal_end_of_block_p_correct(sessions_B, col = 'r', fig_no = fig_no, clf = False)
    if set([s.subject_ID for s in sessions_A]) == set([s.subject_ID for s in sessions_B]):
        print('Paired t-test P value: {}'.format(ttest_rel(p_corrects_A, p_corrects_B)[1]))
    else:
        print('Independent t-test P value: {}'.format(ttest_ind(p_corrects_A, p_corrects_B)[1]))
 
def abs_preference_comparison(sessions_A, sessions_B, population_fit_A, population_fit_B, agent,
                               fig_no = 1, title = None):
    ''' Plot mean absolute preference of model based and model free system based on population fits.
    '''
    mean_preference_mb_A, mean_preference_td_A = rp.abs_preference_plot(sessions_A, population_fit_A, agent, to_plot = False)
    mean_preference_mb_B, mean_preference_td_B = rp.abs_preference_plot(sessions_B, population_fit_B, agent, to_plot = False)
    p.figure(fig_no)
    p.clf()
    p.bar([1  , 3],[mean_preference_mb_A, mean_preference_td_A])
    p.bar([1.8,3.8],[mean_preference_mb_B, mean_preference_td_B],color = 'r')
    p.xticks([1.8, 3.8], ['Model based', 'Model free'])
    p.xlim(0.8,4.8)
    p.ylabel('Mean abs. preference')
    if title:p.title(title)


# -------------------------------------------------------------------------------------
# Permutation tests.
# -------------------------------------------------------------------------------------


def model_fit_test(sessions_A, sessions_B, agent,  perm_type, n_resample = 100, 
                   max_change = 0.001, max_iter = 300, true_init = False, parallel = True, mft = None):
    '''Permutation test for significant differences in model fits between two groups of 
    sessions.  If a previous model_fit_test object (mft) is passed in, additional 
    permutations are performed and the results added to the current test.

    Outline of procedure:
    1. Perform model fitting seperately on both groups of sessions to give mean and standard
    devaiation of population level distributions for each group.
    2. Evaluate distance metric (KL divergence or difference of means) between these population
    level distibutions for each parameter.
    3. Generate population of resampled groups in which sessions are randomly allocated to 
    the A or B groups.  For more information on how permutations are created see _permuted_dataset doc.
    4. Perform model fitting and evalute distance metric for these resampled groups to get a 
    distribution of the distance metric under the null hypothesis that there is no difference 
    between groups.
    5. Compare the true distance metric for each parameter with the distribution for the 
    resampled groups to get a confidence value. 
    '''
    assert perm_type in ('within_subject', 'cross_subject', 'ignore_subject'), \
        'Invalid permutation type.'
        
    if true_init:
        comb_fit = mf.fit_population(sessions_A + sessions_B, agent, eval_BIC = False, parallel = parallel, max_change = max_change * 2, max_iter = max_iter)
        init_params = comb_fit['pop_params']
    else:
        init_params = None

    n_params = agent.n_params

    if not mft: # No previously calculated permutations passed in.

        true_model_fit_A = mf.fit_population(sessions_A, agent, eval_BIC = False, parallel = parallel, max_change = max_change, max_iter = max_iter, pop_init_params = init_params)
        true_model_fit_B = mf.fit_population(sessions_B, agent, eval_BIC = False, parallel = parallel, max_change = max_change, max_iter = max_iter, pop_init_params = init_params)

        true_distances_KL = _population_fit_distance(true_model_fit_A, true_model_fit_B, 'KL')
        true_distances_means = _population_fit_distance(true_model_fit_A, true_model_fit_B, 'means')

        if isinstance(agent, _RL_agent):  # Evaluate mean abs. preference.
            true_preferences_A = rp.abs_preference_plot(sessions_A, true_model_fit_A, agent, to_plot = False)
            true_preferences_B = rp.abs_preference_plot(sessions_B, true_model_fit_B, agent, to_plot = False)
            true_pref_dists = np.abs(np.array(true_preferences_A) - np.array(true_preferences_B))
            
    else: # Previously calculated permutation test passed in.

        n_resample_orig = mft['n_resample']
        n_resample = n_resample + n_resample_orig
        true_model_fit_A, true_model_fit_B  = mft['fit_A'], mft['fit_B']
        true_distances_KL, true_distances_means = mft['KL_data']['true_distances'], mft['means_data']['true_distances']      
        if isinstance(agent, _RL_agent):
            true_preferences_A, true_preferences_B = mft['pref_data']['true_preferences_A'], mft['pref_data']['true_preferences_B']
            true_pref_dists = mft['pref_data']['true_distances']
            
    # Creat structures to store permuted data.
    shuffled_distances_KL = np.zeros((n_resample, n_params))
    shuffled_distances_means = np.zeros((n_resample, n_params))
    shuffled_pref_dists = np.zeros((n_resample, 2))
    shuffled_fits = []

    if not mft:
        perm_indices = range(n_resample)

    else:  # fill first part of arrays with previously calculated data.
        perm_indices = range(n_resample_orig, n_resample)
        shuffled_distances_KL   [:n_resample_orig,:] = mft['KL_data']   ['shuffled_distances']
        shuffled_distances_means[:n_resample_orig,:] = mft['means_data']['shuffled_distances']
        shuffled_fits += mft['shuffled_fits']
        if isinstance(agent, _RL_agent):
            shuffled_pref_dists     [:n_resample_orig,:] = mft['pref_data'] ['shuffled_distances']

    for i in perm_indices:
        print('Fitting permuted sessions, round: {} of {}'.format(i+1, n_resample))

        shuffled_ses_A, shuffled_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
            
        shuffled_fit_A = mf.fit_population(shuffled_ses_A, agent, eval_BIC = False, max_change = max_change, max_iter = max_iter,
                                           pop_init_params = init_params, parallel = parallel)
        shuffled_fit_B = mf.fit_population(shuffled_ses_B, agent, eval_BIC = False, max_change = max_change, max_iter = max_iter,
                                           pop_init_params = init_params, parallel = parallel)
        shuffled_fits.append(({'means':shuffled_fit_A['pop_params']['means'],'SDs':shuffled_fit_A['pop_params']['SDs']},
                              {'means':shuffled_fit_B['pop_params']['means'],'SDs':shuffled_fit_B['pop_params']['SDs']}))
        shuffled_distances_KL[i,:]    = _population_fit_distance(shuffled_fit_A, shuffled_fit_B, 'KL')
        shuffled_distances_means[i,:] = _population_fit_distance(shuffled_fit_A, shuffled_fit_B, 'means')

        if isinstance(agent, _RL_agent): 
            shuffled_preferences_A = rp.abs_preference_plot(shuffled_ses_A, shuffled_fit_A, agent, to_plot = False)
            shuffled_preferences_B = rp.abs_preference_plot(shuffled_ses_B, shuffled_fit_B, agent, to_plot = False)
            shuffled_pref_dists[i,:] = np.abs(np.array(shuffled_preferences_A) -
                                       np.array(shuffled_preferences_B))

    dist_ranks_KL = sum(shuffled_distances_KL>=np.tile(true_distances_KL,(n_resample,1)),0)
    p_vals_KL = dist_ranks_KL / n_resample  # Should this be n_resample + 1?

    dist_ranks_means = sum(shuffled_distances_means>=np.tile(true_distances_means,(n_resample,1)),0)   
    p_vals_means = dist_ranks_means / n_resample  # Should this be n_resample + 1?

    mft =  {'fit_A': true_model_fit_A,
            'fit_B': true_model_fit_B,
            'n_resample': n_resample,
            'perm_type': perm_type,
            'shuffled_fits': shuffled_fits, 
            'KL_data':     {'true_distances': true_distances_KL,
                            'shuffled_distances': shuffled_distances_KL,
                            'dist_ranks': dist_ranks_KL,
                            'p_vals': p_vals_KL},
            'means_data':  {'true_distances': true_distances_means,
                            'shuffled_distances': shuffled_distances_means,
                            'dist_ranks': dist_ranks_means,
                            'p_vals': p_vals_means}
            }

    if isinstance(agent, _RL_agent): 
        dist_ranks_pref = sum(shuffled_pref_dists>=np.tile(true_pref_dists,(n_resample,1)),0)
        p_vals_pref = dist_ranks_pref / n_resample  # Should this be n_resample + 1?
        mft['pref_data'] = {'true_preferences_A' : true_preferences_A,
                            'true_preferences_B' : true_preferences_B,
                            'true_distances': true_pref_dists,
                            'shuffled_distances': shuffled_pref_dists,
                            'dist_ranks': dist_ranks_pref,
                            'p_vals': p_vals_pref}
    return mft

def _population_fit_distance(fit_A, fit_B, metric = 'KL'):
    '''Evaluate distance between distributions for each parameter of a pair of population fits.
    Distributions are assumed to be gaussians specified by mean and standard deviation.  
    Metric can be specified as'KL' for KL  divervence, or 'means' for absolute difference
    of means.  Used by model_fit_test. '''

    assert fit_A['param_names'] == fit_B['param_names'], \
    'Fits are not from same model, cannot evalate distance.'
    assert metric in ['KL', 'means'], 'Invalid distance metric.'

    means_A = fit_A['pop_params']['means']
    SDs_A   = fit_A['pop_params']['SDs']
    means_B = fit_B['pop_params']['means']
    SDs_B   = fit_B['pop_params']['SDs']
    if metric == 'KL':
        distances = np.log(SDs_B/SDs_A) + \
                   ((SDs_A**2 + (means_A-means_B)**2) / \
                   (2 * SDs_B**2)) - 0.5
    elif metric =='means':
        distances = np.abs(means_A - means_B)
    return distances

def MAP_fit_test(sessions_A, sessions_B, agent,  perm_type, n_resample = 1000,
                 max_change = 0.01, parallel = False, use_median = False):
    ''' A test for differences in model fits between two groups of subjects which fits a single
    population distribution to both sets of sessions combined and then looks for differences in the 
    distribution of MAP fits between the two groups.
    '''

    all_sessions = sessions_A + sessions_B

    all_sessions_fit =  mf.fit_population(all_sessions, agent, parallel = parallel, max_change = max_change)

    for i, MAP_fit in enumerate(all_sessions_fit['MAP_fits']):
        all_sessions[i].MAP_fit = MAP_fit

    true_MAP_fits_A = np.array([s.MAP_fit['params_T'] for s in sessions_A])
    true_MAP_fits_B = np.array([s.MAP_fit['params_T'] for s in sessions_B])

    if use_median:
        ave_func = np.median
    else:
        ave_func = np.mean

    true_fit_dists = np.abs(ave_func(true_MAP_fits_A, 0) - ave_func(true_MAP_fits_B, 0))

    shuffled_fit_dists = np.zeros([n_resample, agent.n_params])

    for i in range(n_resample):
        print('Evaluating permuted sessions, round: {} of {}'.format(i+1, n_resample))
        shuffled_ses_A, shuffled_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        shuffled_MAP_fits_A = np.array([s.MAP_fit['params_T'] for s in shuffled_ses_A])
        shuffled_MAP_fits_B = np.array([s.MAP_fit['params_T'] for s in shuffled_ses_B])
        shuffled_fit_dists[i,:] = np.abs(ave_func(shuffled_MAP_fits_A, 0) -
                                         ave_func(shuffled_MAP_fits_B, 0))

        dist_ranks = sum(shuffled_fit_dists>=np.tile(true_fit_dists,(n_resample,1)),0)
        p_vals = dist_ranks / n_resample

    return p_vals









def reversal_test(sessions_A, sessions_B, perm_type, n_resample = 1000, by_type = False, groups = None):
    ''' Permutation test for differences in the fraction correct at end of blocks and the time constant
    of adaptation to block transitions.
    '''
    fit_A = pl.reversal_analysis(sessions_A, return_fits = True, by_type = by_type)
    fit_B = pl.reversal_analysis(sessions_B, return_fits = True, by_type = by_type)
    true_reversal_fit_distances = _reversal_fit_distances(fit_A,fit_B)
    permuted_reversal_fit_distances = np.zeros([n_resample, 4])
    for i in range(n_resample):
        print('Fitting permuted sessions, round: {} of {}'.format(i+1, n_resample))

        shuffled_ses_A, shuffled_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type, groups)
        shuffled_fit_A = pl.reversal_analysis(shuffled_ses_A, return_fits = True, by_type = by_type)
        shuffled_fit_B = pl.reversal_analysis(shuffled_ses_B, return_fits = True, by_type = by_type)
        permuted_reversal_fit_distances[i,:] = _reversal_fit_distances(shuffled_fit_A, shuffled_fit_B)

    dist_ranks = sum(permuted_reversal_fit_distances>=np.tile(true_reversal_fit_distances,(n_resample,1)),0) 
    p_vals = dist_ranks / float(n_resample)
    print('Block end choice probability P value   : {}'.format(p_vals[0]))
    print('All reversals tau P value              : {}'.format(p_vals[1]))
    if by_type:
        print('Reward probability reversal tau P value: {}'.format(p_vals[2]))
        print('Trans. probability reversal tau P value: {}'.format(p_vals[3]))
    return {'block_end_P_value': p_vals[0], 'tau_P_value' : p_vals[1]}

def _reversal_fit_distances(fit_A, fit_B):
    '''Evaluate absolute difference in asymtotic choice probability and reversal time
    constants for pair of fits to reversal choice trajectories.  Used by reversal test.'''
    if fit_A['rew_rev']:  # Fit includes seperate analyses by reversal type.
        return np.abs([fit_A['p_1']              - fit_B['p_1'],
                       fit_A['both_rev']['tau']  - fit_B['both_rev']['tau'],
                       fit_A['rew_rev']['tau']   - fit_B['rew_rev']['tau'],
                       fit_A['trans_rev']['tau'] - fit_B['trans_rev']['tau']])
    else:
        return np.abs([fit_A['p_1']              - fit_B['p_1'],
                       fit_A['both_rev']['tau']  - fit_B['both_rev']['tau'], 0., 0.])


def trial_rate_test(sessions_A, sessions_B, perm_type, test_time = 120, n_resample = 1000): 
    ''' Evaluate whether number of trials per session in first test_time minutes is 
    different between groups.
    '''
    for session in sessions_A + sessions_B:
        session.n_trials_test = sum(session.trial_start_times < (60 * test_time))

    true_n_trials_diff = np.abs(sum([s.n_trials_test for s in sessions_A]) - \
                                sum([s.n_trials_test for s in sessions_B]))
    perm_n_trials_diff = np.zeros(n_resample)
    for i in range(n_resample):
        shuffled_ses_A, shuffled_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        perm_n_trials_diff[i] = np.abs(sum([s.n_trials_test for s in shuffled_ses_A]) - \
                                       sum([s.n_trials_test for s in shuffled_ses_B]))
    p_val = sum(perm_n_trials_diff>=true_n_trials_diff) /float(n_resample)
    print('Trial number difference P value: {}'.format(p_val))
    return {'test_time': test_time,
            'p_val'    : p_val} 


def _permuted_dataset(sessions_A, sessions_B, perm_type = 'ignore_subject', groups = None):
    ''' Generate permuted datasets by randomising assignment of sessions between groups A and B.
    perm_type argument controls how permutations are implemented:
    'within_subject' - Permute sessions within subject such that each permuted group has the same
                     number of session from each subject as the true datasets.
    'cross_subject' - All sessions from a given subject are assigned to one or other of the permuted datasets.
    'ignore_subject' - The identity of the subject who generated each session is ignored in the permutation.
    'within_group' - Permute subjects within groups that are subsets of all subjects.  
                     Animal assignment to groups is specified by groups argument which should be 
                     a list of lists of animals in each group.
    '''
    assert perm_type in ('within_subject', 'cross_subject', 'ignore_subject','within_group'), \
        'Invalid permutation type.'
    all_sessions = sessions_A + sessions_B
    all_subjects = list(set([s.subject_ID for s in all_sessions]))
    if perm_type == 'ignore_subject':  # Shuffle sessions ignoring which subject each session is from.        
        shuffle(all_sessions)
        shuffled_ses_A = all_sessions[:len(sessions_A)]
        shuffled_ses_B = all_sessions[len(sessions_A):]
    elif perm_type == 'cross_subject':  # Permute subjects across groups (used for cross subject tests.)
        n_subj_A     = len(set([s.subject_ID for s in sessions_A]))        
        shuffle(all_subjects)   
        shuffled_ses_A = [s for s in all_sessions if s.subject_ID in all_subjects[:n_subj_A]]
        shuffled_ses_B = [s for s in all_sessions if s.subject_ID in all_subjects[n_subj_A:]]
    elif perm_type == 'within_subject': # Permute sessions keeping number from each subject in each group constant.
        shuffled_ses_A = []
        shuffled_ses_B = []
        for subject in all_subjects:
            subject_sessions_A = [s for s in sessions_A if s.subject_ID == subject]
            subject_sessions_B = [s for s in sessions_B if s.subject_ID == subject]
            all_subject_sessions = subject_sessions_A + subject_sessions_B
            shuffle(all_subject_sessions)
            shuffled_ses_A += all_subject_sessions[:len(subject_sessions_A)]
            shuffled_ses_B += all_subject_sessions[len(subject_sessions_A):]
    elif perm_type == 'within_group':
        shuffled_ses_A, shuffled_ses_B = ([], [])
        for group in groups:
            group_sessions_A = [s for s in sessions_A if s.subject_ID in group]
            group_sessions_B = [s for s in sessions_B if s.subject_ID in group]
            group_shuffled_ses_A, group_shuffled_ses_B = _permuted_dataset(group_sessions_A, group_sessions_B, 'cross_subject')
            shuffled_ses_A += group_shuffled_ses_A
            shuffled_ses_B += group_shuffled_ses_B
    return (shuffled_ses_A, shuffled_ses_B)

#---------------------------------------------------------------------------------------------------
#  Plotting
#---------------------------------------------------------------------------------------------------

def plot_resampled_dists(mft, fig_title = 'Permutation test', fig_no = 1, x_offset = 0.1):
    n_resample = mft['n_resample']
    
    print('Permutations evaluated: {}'.format(mft['n_resample']))
    print('P values    KL: {}'.format(mft['KL_data']['p_vals']))
    print('P values means: {}'.format(mft['means_data']['p_vals']))
    if 'pref_data' in mft.keys():
        print('P values pref: {}'.format(mft['pref_data']['p_vals']))


    #Plotting
    p.figure(fig_no)
    p.clf()
    rp.pop_scatter_plot(mft['fit_A'], col = 'b', clf = True,  subplot = (3,1,1), x_offset = -x_offset)
    rp.pop_scatter_plot(mft['fit_B'], col = 'r', clf = False, subplot = (3,1,1), x_offset =  x_offset)
    
    if fig_title:
        p.suptitle(fig_title)

    p.subplot(3,1,2)
    _plot_dist(mft, 'KL')

    p.subplot(3,1,3)
    _plot_dist(mft, 'means')

def _plot_dist(mft, metric):
    perm_data = mft[metric + '_data']
    n_params =   len(perm_data['p_vals'])

    shuffled_distances = perm_data['shuffled_distances']
    true_distances = perm_data['true_distances']


    sorted_dists = np.sort(shuffled_distances,0)
    median_dists = np.median(shuffled_distances,0)
    min_dists = sorted_dists[0,:]
    max_dists = sorted_dists[-1,:]
    lower_95_conf = sorted_dists[int(np.floor(0.05*mft['n_resample'])),:]
    upper_95_conf = sorted_dists[int(np.ceil (0.95*mft['n_resample']))-1,:]
    p.errorbar(np.arange(n_params)+0.4, median_dists,
               yerr = (median_dists -min_dists, max_dists - median_dists),
               linestyle = '', linewidth = 2, color = 'k')
    p.errorbar(np.arange(n_params)+0.6, median_dists,
               yerr = (median_dists -lower_95_conf, upper_95_conf - median_dists),
               linestyle = '', linewidth = 2)
    p.plot(np.arange(n_params)+0.5, true_distances, linestyle = '', color = 'r', marker = '.')
    p.xlim(0,n_params)
    p.ylim(ymin = p.ylim()[1]*-0.05)
    p.xticks(np.arange(n_params)+0.5, mft['fit_A']['param_names'])
    p.ylabel('Distance (' + metric + ')')

def subject_fits_group_comparison(sub_fits_a, sub_fits_b, fig_no = 1):
    means_a = np.array([fit['pop_params']['means'] for fit in sub_fits_a])
    means_b = np.array([fit['pop_params']['means'] for fit in sub_fits_b])
    n_params = means_a.shape[1]
    pop_mean_a = np.mean(means_a,0)
    pop_mean_b = np.mean(means_b,0)
    pop_SEM_a = np.sqrt(np.var(means_a)/means_a.shape[0])
    pop_SEM_b = np.sqrt(np.var(means_b)/means_b.shape[0])
    p.figure(fig_no)
    p.clf()
    p.errorbar(np.arange(n_params), pop_mean_a, pop_SEM_a,linestyle = '', marker = '', linewidth = 2, color = 'b')
    p.errorbar(np.arange(n_params), pop_mean_b, pop_SEM_b,linestyle = '', marker = '', linewidth = 2, color = 'r')
    p.plot([0,n_params],[0,0],'k')
    p.xticks(np.arange(n_params), sub_fits_a[0]['param_names'])
