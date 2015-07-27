''' Plotting and analysis functions.

'''
import pylab as p
import numpy as np
import utility as ut
from scipy.stats import binom
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import ttest_rel
from human_session import human_session

p.ion()
p.rcParams['pdf.fonttype'] = 42

#----------------------------------------------------------------------------------
#  Plot utility functions used in plotting functions.
#----------------------------------------------------------------------------------

def setup_axis(plot_pos = []):

    ''' Set up figure and subplot.
    To set figure x and subplot(y1,y2,y3), pass plot_pos = (x, [y1,y2,y3])
    Pass any argument that evaluates to boolean false to plot to Fig 1 full pane.
    Pass 'current_axis' to leave current axis unchanged.
    Pass 'no_plot' to return boolean false.
    '''
    if plot_pos == 'no_plot':
        return False
    else:
        if not(plot_pos):
            plot_pos = (1, [1,1,1])
        if plot_pos != 'current_axis':
            p.figure(plot_pos[0])
            if plot_pos[1][2] == 1:
                p.clf()
            p.subplot(plot_pos[1][0],plot_pos[1][1],plot_pos[1][2])
        return True

def plot_event(event_name,y_pos,style, times):
    p.plot(times[event_name]/ 60.,y_pos * np.ones(times[event_name].shape),style)

def symplot(y,guard,symbol):
    x_ind = np.where(guard)[0]
    p.plot(x_ind,y[x_ind],symbol, markersize = 5)

#----------------------------------------------------------------------------------
# Various analyses.
#----------------------------------------------------------------------------------

def choice_mov_ave(session, plot_pos = [], show_TO = True):
    'Plot of choice moving average and reward block structure for single session.'
    setup_axis(plot_pos)
    choices, transitions, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, dtype = bool)
    second_steps = second_steps * 1.1-0.05
    mov_ave = ut.exp_mov_ave(choices)

    p.plot(mov_ave,'k.-', markersize = 3)

    if hasattr(session, 'blocks'):
        transitions = transitions == session.blocks['trial_trans_state'] # Convert transitions AB to transtions CR.
        for i in range(len(session.blocks['start_trials'])):
            y = [0.1,0.5,0.9][session.blocks['reward_states'][i]]  # y position coresponding to reward state.
            x = [session.blocks['start_trials'][i], session.blocks['end_trials'][i]]
            if session.blocks['transition_states'][i]:
                p.plot(x, [y,y], 'orange', linewidth = 2)
            else:
                y = 1 - y  # Invert y position if transition is inverted.
                p.plot(x, [y,y], 'purple', linewidth = 2)    
    if show_TO:
        symplot(second_steps,  transitions &  outcomes,'ob' )
        symplot(second_steps,  transitions & ~outcomes,'xb')
        symplot(second_steps, ~transitions &  outcomes,'og')
        symplot(second_steps, ~transitions & ~outcomes,'xg')  
    p.plot([0,len(choices)],[0.75,0.75],'--k')
    p.plot([0,len(choices)],[0.25,0.25],'--k')

    p.xlabel('Trial Number')
    p.yticks([0,0.5,1])
    p.ylim(-0.1, 1.1)
    p.xlim(0,len(choices))
    p.ylabel('Choice moving average')

def poke_poke_corrlations(IDs, event_codes, plot_pos = []):
    'Poke probability as function of previous poke.' 
    poke_IDs =  ut.get_IDs(IDs, ['low_poke','high_poke', 'left_poke', 'right_poke'])
    poke_sequence = event_codes[ut.array_contains(poke_IDs, event_codes)]
    poke_pairs = zip(poke_sequence[0:-1],poke_sequence[1::])
    k = {IDs['high_poke']: 0,IDs['low_poke']: 1, IDs['left_poke']: 2,IDs['right_poke']: 3}
    poke_counts = np.zeros([4,4])
    for poke, next_poke in poke_pairs:
            poke_counts[k[poke],k[next_poke]] += 1
    poke_probs = poke_counts / np.tile(np.sum(poke_counts,1)[np.newaxis].T,(1,4))
    if setup_axis(plot_pos):
        p.imshow(poke_probs,cmap=p.cm.copper,interpolation='nearest')
        p.colorbar()
        p.xticks([0,1,2,3],['High','Low','Left','Right'])
        p.yticks([0,1,2,3],['High','Low','Left','Right'])
        p.xlabel('Poke t + 1')
        p.ylabel('Poke t')
    else: return poke_probs

def runlength_analysis(sessions):
    'Histogram of length of runs of single choice.'
    run_lengths = []
    for session in sessions:
        choices = session.CTSO['choices']
        prev_choice = choices[0]
        run_length = 0
        for choice in choices[1:]:
            run_length += 1
            if not choice == prev_choice:
                run_lengths.append(run_length)
                run_length = 0
            prev_choice= choice
    counts,bins = np.histogram(run_lengths,range(1,41))
    p.plot(bins[:-1], counts/float(len(run_lengths)))
    p.ylabel('Fraction')
    p.xlabel('Run length')
    print('Mean run length: {}'.format(np.mean(run_lengths)))

#----------------------------------------------------------------------------------
# Longditudinal analyses through training.
#----------------------------------------------------------------------------------

def cumulative_trials(experiment, col = 'b'):
    'Cumulative trials as a function of session number with cross-animal SEM errorbars.'
    trials = np.zeros([experiment.n_subjects, experiment.n_days])
    blocks = np.zeros([experiment.n_subjects, experiment.n_days])
    for i, sID in enumerate(experiment.subject_IDs):
        for day in range(experiment.n_days):
            trials[i, day] =     experiment.get_sessions(sID, day + 1).n_trials
            blocks[i, day] = len(experiment.get_sessions(sID, day + 1).blocks['start_trials']) - 1
    cum_trials = np.cumsum(trials,1)
    mean_trials = np.mean(cum_trials,0)
    SEM_trials = np.sqrt(np.var(cum_trials,0)/float(experiment.n_subjects))
    cum_blocks = np.cumsum(blocks,1)
    mean_blocks = np.mean(cum_blocks,0)
    SEM_blocks = np.sqrt(np.var(cum_blocks,0)/float(experiment.n_subjects))
    days = np.arange(experiment.n_days) + 1
    p.figure(1)
    p.subplot(2,1,1)
    p.plot(days, mean_trials)
    p.fill_between(days, mean_trials - SEM_trials, mean_trials + SEM_trials, alpha = 0.2, facecolor = col)
    p.subplot(2,1,2)
    p.plot(days, mean_blocks)
    p.fill_between(days, mean_blocks - SEM_blocks, mean_blocks + SEM_blocks, alpha = 0.2, facecolor = col)

def trials_per_block(experiment, use_blocks = 'non_neutral', clf = True, fig_no = 1, last_n = 6):
    ' Number of trials taken to finish each block.'
    days = set([s.day for s in experiment.get_sessions('all', 'all')])   
    mean_tpb, sd_tpb = ([],[])
    residual_trials = np.zeros(len(experiment.subject_IDs)) # Number of trials in last (uncompleted) block of session.
    days_trials_per_block = []
    for day in days:
        day_trials_per_block = []
        sessions = experiment.get_sessions('all', day)
        for session in sessions:
            assert hasattr(session,'blocks'), 'Session do not have block info.'
            ax = experiment.subject_IDs.index(session.subject_ID) # Animal index used for residual trials array.
            block_lengths = np.subtract(session.blocks['end_trials'],
                                        session.blocks['start_trials'])
            block_lengths[0] += residual_trials[ax]
            residual_trials[ax] = block_lengths[-1]
            block_types = {'all': block_lengths[:-1],
                       'neutral': block_lengths[:-1][    np.equal(session.blocks['reward_states'],1)[:-1]],
                   'non_neutral': block_lengths[:-1][np.not_equal(session.blocks['reward_states'],1)[:-1]]}
            day_trials_per_block.append(np.mean(block_types[use_blocks]))
        mean_tpb.append(ut.nanmean(day_trials_per_block))
        sd_tpb.append(np.sqrt(ut.nanvar(np.array(day_trials_per_block))))
        days_trials_per_block.append(day_trials_per_block)
    days = np.array(list(days))-min(days) + 1
    p.figure(fig_no)
    if clf: p.clf()
    p.subplot(2,1,1)
    p.errorbar(days, mean_tpb, sd_tpb)
    p.xlim(0.5, max(days) + 0.5)
    p.xlabel('Day')
    p.ylabel('Trials per block')
    print 'Mean   trails per block for last {} sessions: {:.1f}'.format(last_n, np.mean  (mean_tpb[-last_n:]))
    print 'Median trails per block for last {} sessions: {:.1f}'.format(last_n, np.median(mean_tpb[-last_n:]))
    last_n_day_block_lengths = np.array(days_trials_per_block[-last_n:]).reshape([1,-1])[0]
    last_n_day_block_lengths = last_n_day_block_lengths[~np.isnan(last_n_day_block_lengths)]
    p.subplot(2,1,2)
    p.hist(last_n_day_block_lengths)


def blocks_per_trial(experiment, neutral_blocks = False, clf = True, fig_no = 1, last_n = 6):
    days = set([s.day for s in experiment.get_sessions('all', 'all')])  
    residual_trials = np.zeros(len(experiment.subject_IDs)) # Number of trials in last (uncompleted) block of session.
    mean_bpt, sd_bpt = ([],[]) # Lists to hold mean and standard deviation of blocks per trial for each day.
    for day in days:
        day_blocks_per_trial = []
        sessions = experiment.get_sessions('all', day)
        for session in sessions:
            assert hasattr(session,'blocks'), 'Session does not have block info.'
            ax = experiment.subject_IDs.index(session.subject_ID) # Index used for residual trials array.
            blocks_per_trial, residual_trials[ax] = _session_blocks_per_trial(session, residual_trials[ax], neutral_blocks)
            day_blocks_per_trial.append(blocks_per_trial)
        mean_bpt.append(ut.nanmean(day_blocks_per_trial))
        sd_bpt.append  (np.sqrt(ut.nanvar(np.array(day_blocks_per_trial))))
    days = np.array(list(days))-min(days) + 1
    p.figure(fig_no)
    if clf: p.clf()
    p.subplot(2,1,1)
    p.errorbar(days, mean_bpt, sd_bpt/np.sqrt(len(experiment.subject_IDs)))
    p.xlim(0.5, max(days) + 0.5)
    p.ylim(ymin = 0)
    p.ylabel('Blocks per trial')
    p.subplot(2,1,2)
    p.plot(days,1 / np.array(mean_bpt))
    p.xlabel('Day')
    p.ylabel('Trials per block')

def _session_blocks_per_trial(session, residual_trials = 0, neutral_blocks = False):
    ''' Evaluate average number of block tranistions per trial (<< 1) for a given session
    for neutral and non-neutral blocks.  Residual trials is the number of trials of the
    final block of the previous session.        
    '''
    blocks = session.blocks
    block_lengths = np.subtract(session.blocks['end_trials'],
                                session.blocks['start_trials'])
    block_lengths[0] += residual_trials  # Add residual trials from previous session to first block.
    residual_trials = block_lengths[-1]  # Residual trials for subsequent session.
    if neutral_blocks: # Analyse neutral blocks only.
        use_blocks = np.equal(blocks['reward_states'],1)
    else: # Analyse only non-neutral blocks.
        use_blocks =  ~np.equal(blocks['reward_states'],1)
    n_completed = sum(use_blocks[:-1])         # Number of blocks completed.
    n_trials = sum(block_lengths[use_blocks])  # Number of trials to complete blocks.
    try:
        blocks_per_trial = n_completed / float(n_trials)
    except ZeroDivisionError:
        blocks_per_trial = np.nan
    return (blocks_per_trial, residual_trials)


def rotation_analysis(sessions, cor_len = 200):
    ''' Evaluate auto and cross correlations for center->side 
    transitions and side->center choices over concatonated sessions for
    each subject.
    '''
    sIDs = list(set([s.subject_ID for s in sessions]))

    sub_trans_auto_corrs = np.zeros([len(sIDs), 2 * cor_len + 1])   # Transition autocorrelations for each subject.
    sub_choice_auto_corrs = np.zeros([len(sIDs), 2 * cor_len + 1])  # Choice (coded clockwise vs anticlockwise) autocorrelation for each subject.
    sub_cross_corrs = np.zeros([len(sIDs), 2 * cor_len + 1])        # Transition - choice cross correlations for each subject.
    for i, sID in enumerate(sIDs):
        a_sessions = sorted([s for s in sessions if s.subject_ID == sID],
                            key = lambda s:s.day)
        choices = []
        transitions = []
        for s in a_sessions:
            transitions += s.CTSO['transitions'][1:].tolist()
            choices += (s.CTSO['choices'][1:] == s.CTSO['second_steps'][:-1]).tolist()

        trans_autocor  = ut.norm_correlate(transitions, transitions)
        choice_autocor = ut.norm_correlate(choices, choices)
        cross_corr     = ut.norm_correlate(transitions, choices)
        cor_inds = range(cross_corr.size/2 - cor_len,
                         cross_corr.size/2 + cor_len + 1)
        sub_trans_auto_corrs[i,:] = trans_autocor[cor_inds]
        sub_choice_auto_corrs[i,:] = choice_autocor[cor_inds]
        sub_cross_corrs[i,:] = cross_corr[cor_inds]
    p.figure(1)
    p.clf()
    x = range(-cor_len, cor_len + 1)
    p.plot(x, np.mean(sub_trans_auto_corrs, 0))
    p.plot(x, np.mean(sub_choice_auto_corrs, 0))
    p.plot(x, np.mean(sub_cross_corrs, 0))
    p.ylim(0, sorted(np.mean(sub_trans_auto_corrs, 0))[-2])
    p.legend(('Trans. autocor.', 'Choice autocor.', 'Cross corr.'))
    p.xlabel('Lag (trials)')
    p.ylabel('Correlation')

#----------------------------------------------------------------------------------
# Temporal analyses.
#----------------------------------------------------------------------------------

def log_IPI(session, plot_pos = []):
    'Log inter-poke interval distribution.'
    setup_axis(plot_pos)
    poke_IDs =  ut.get_IDs(session.IDs, ['low_poke','high_poke', 'left_poke', 'right_poke'])
    poke_times = session.time_stamps[ut.array_contains(poke_IDs, session.event_codes)]
    log_IPIs = np.log(poke_times[1::]-poke_times[0:-1])
    poke_events = session.event_codes[ut.array_contains(poke_IDs, session.event_codes)]
    repeated_poke_log_IPIs = log_IPIs[poke_events[1::] == poke_events[0:-1]]   
    bin_edges = np.arange(-1.7,7.6,0.1)
    log_IPI_hist = np.histogram(log_IPIs, bin_edges)[0]
    rep_poke_log_IPI_hist = np.histogram(repeated_poke_log_IPIs, bin_edges)[0]
    mode_IPI = np.exp(bin_edges[np.argmax(log_IPI_hist)]+0.05)
    if setup_axis(plot_pos):
        p.plot(bin_edges[0:-1]+0.05,log_IPI_hist)
        p.plot(bin_edges[0:-1]+0.05,rep_poke_log_IPI_hist,'r')
        p.xlim(bin_edges[0],bin_edges[-1])
        p.xticks(np.log([0.25,1,4,16,64,256]),[0.25,1,4,16,64,256])
        p.xlabel('Inter-poke interval (sec)')
    else: return (mode_IPI, log_IPI_hist, bin_edges[0:-1]+0.05)

def log_ITI(sessions, boundary = 10.):
    'log inter - trial start interval distribution.'
    all_ITIs = []
    for session in sessions:
        trial_start_times = session.time_stamps[session.event_codes == session.IDs['trial_start']]
        ITIs = trial_start_times[1:] - trial_start_times[:-1]
        all_ITIs.append(ITIs)
    all_ITIs = np.concatenate(all_ITIs)
    fraction_below_bound = round(sum(all_ITIs < boundary) / float(len(all_ITIs)),3)
    print('{} of ITIs less than {} seconds.'.format(fraction_below_bound, boundary))
    p.figure(1)
    p.clf()
    p.hist(np.log(all_ITIs),100)
    p.xlim(0,5)
    p.xlabel('Log inter trial start interval (sec)')
    p.ylabel('Count')


def reaction_times_second_step(sessions, fig_no = 1):
    'Reaction times for second step pokes as function of common / rare transition.'
    sec_step_IDs = ut.get_IDs(sessions[0].IDs, ['right_active', 'left_active'])
    median_RTs_common = np.zeros(len(sessions))
    median_RTs_rare   = np.zeros(len(sessions))
    for i,session in enumerate(sessions):
        event_times = ut.get_event_times(session.time_stamps, session.event_codes, session.IDs)
        left_active_times = event_times['left_active']
        right_active_times = event_times['right_active']
        left_reaction_times  = _latencies(left_active_times,  event_times['left_poke'])
        right_reaction_times = _latencies(right_active_times, event_times['right_poke'])
        ordered_reaction_times = np.hstack((left_reaction_times,right_reaction_times))\
                                 [np.argsort(np.hstack((left_active_times,right_active_times)))]
        transitions = session.blocks['trial_trans_state'] == session.CTSO['transitions']  # common vs rare.                 
        median_RTs_common[i] = np.median(ordered_reaction_times[ transitions])
        median_RTs_rare[i]    = np.median(ordered_reaction_times[~transitions])
    mean_RT_common = 1000 * np.mean(median_RTs_common)
    mean_RT_rare   = 1000 * np.mean(median_RTs_rare)
    SEM_RT_common = 1000 * np.sqrt(np.var(median_RTs_common/len(sessions)))
    SEM_RT_rare   = 1000 * np.sqrt(np.var(median_RTs_rare  /len(sessions)))
    p.figure(fig_no)
    p.bar([1,2],[mean_RT_common, mean_RT_rare], yerr = [SEM_RT_common,SEM_RT_rare])
    p.xlim(0.8,3)
    p.ylim(mean_RT_common * 0.8, mean_RT_rare * 1.1)
    p.xticks([1.4, 2.4], ['Common', 'Rare'])
    p.title('Second step reaction times')
    p.ylabel('Reaction time (ms)')
    print('Paired t-test P value: {}'.format(ttest_rel(median_RTs_common, median_RTs_rare)[1]))

def reaction_times_first_step(sessions):
    median_reaction_times = np.zeros([len(sessions),4])
    all_reaction_times = []
    for i,session in enumerate(sessions):
        event_times = ut.get_event_times(session.time_stamps, session.event_codes, session.IDs)
        ITI_start_times = event_times['ITI_start']
        center_poke_times = sorted(np.hstack((event_times['high_poke'], event_times['low_poke'])))
        reaction_times = 1000 * _latencies(ITI_start_times,  center_poke_times)[1:-1]
        all_reaction_times.append(reaction_times)
        transitions = (session.blocks['trial_trans_state'] == session.CTSO['transitions'])[:len(reaction_times)] # Transitions common/rare.
        outcomes = session.CTSO['outcomes'][:len(reaction_times)].astype(bool)
        median_reaction_times[i, 0] = np.median(reaction_times[ transitions &  outcomes])  # Common transition, rewarded.
        median_reaction_times[i, 1] = np.median(reaction_times[~transitions &  outcomes])  # Rare transition, rewarded.
        median_reaction_times[i, 2] = np.median(reaction_times[ transitions & ~outcomes])  # Common transition, non-rewarded.
        median_reaction_times[i, 3] = np.median(reaction_times[~transitions & ~outcomes])  # Rare transition, non-rewarded.
    mean_RTs = np.mean(median_reaction_times,0)
    SEM_RTs  = np.sqrt(np.var(median_reaction_times,0)/len(sessions))
    p.figure(1)
    p.clf()
    p.title('First step reaction times')
    p.bar([1,2,3,4], mean_RTs, yerr = SEM_RTs)
    p.ylim(min(mean_RTs) * 0.8, max(mean_RTs) * 1.1)
    p.xticks([1.4, 2.4, 3.4, 4.4], ['Com. Rew.', 'Rare Rew.', 'Com. Non.', 'Rare. Non.'])
    p.xlim(0.8,5)
    p.ylabel('Reaction time (ms)')
    all_reaction_times = np.hstack(all_reaction_times)
    bin_edges = np.arange(0,3001)
    rt_hist = np.histogram(all_reaction_times, bin_edges)[0]
    cum_rt_hist = np.cumsum(rt_hist) / float(len(all_reaction_times))
    p.figure(2)
    p.clf()
    p.plot(bin_edges[:-1],cum_rt_hist)
    p.ylim(0,1)
    p.xlabel('Time from ITI start (ms)')
    p.ylabel('Cumumative fraction of first central pokes.')


def pokes_per_min(session, plot_pos = []):
    setup_axis(plot_pos)
    poke_IDs =  ut.get_IDs(session.IDs, ['low_poke','high_poke', 'left_poke', 'right_poke'])
    poke_times = session.time_stamps[ut.array_contains(poke_IDs, session.event_codes)]/60
    bin_edges = np.arange(0,np.ceil(poke_times.max())+1)
    pokes_per_min = np.histogram(poke_times, bin_edges)[0]
    p.plot(bin_edges[0:-1]+0.5,pokes_per_min)
    p.xlabel('Time (minutes)')

def trials_per_minute(sessions, smooth_SD = 5, ses_dur = 120, fig_no = 1, ebars = 'SEM',
                      clf = True, col = 'b', plot_cum = False):
    bin_edges = np.arange(ses_dur + 1)
    all_trials_per_minute = np.zeros((len(sessions),ses_dur))
    for i, session in enumerate(sessions):
        trial_start_ID = session.IDs['trial_start']
        trial_times = session.time_stamps[session.event_codes == trial_start_ID][1:] / 60
        trials_per_min = np.histogram(trial_times, bin_edges)[0]
        if smooth_SD: #Smooth by convolution with gaussian of specified standard deviation.
            trials_per_min = gaussian_filter1d(trials_per_min, smooth_SD)
        all_trials_per_minute[i,:] = trials_per_min
    mean_tpm = np.mean(all_trials_per_minute,0)
    sd_tpm = np.sqrt(np.var(all_trials_per_minute,0))
    sem_tpm = sd_tpm / np.sqrt(len(sessions))
    cumulative_tpm = np.cumsum(all_trials_per_minute,1)
    mean_ctpm = np.mean(cumulative_tpm,0)
    sd_ctpm = np.sqrt(np.var(cumulative_tpm,0))
    sem_ctpm = sd_ctpm / np.sqrt(len(sessions))    
    p.figure(fig_no)
    if clf: p.clf()
    if plot_cum:
        p.subplot(2,1,1)
    p.plot(bin_edges[1:], mean_tpm, color = col)
    if ebars == 'SD':
        p.fill_between(bin_edges[1:], mean_tpm-sd_tpm, mean_tpm+sd_tpm, alpha = 0.2, facecolor = col)
    elif ebars == 'SEM':
        p.fill_between(bin_edges[1:], mean_tpm-sem_tpm, mean_tpm+sem_tpm, alpha = 0.2, facecolor = col)
    p.ylabel('Trials per minute') 
    p.xlim(1,ses_dur)
    p.ylim(ymin = 0)
    if plot_cum:
        p.subplot(2,1,2)
        p.plot(bin_edges[1:], mean_ctpm, color = col)
        if ebars == 'SD':
            p.fill_between(bin_edges[1:], mean_ctpm-sd_ctpm, mean_ctpm+sd_ctpm, alpha = 0.2, facecolor = col)
        elif ebars == 'SEM':
            p.fill_between(bin_edges[1:], mean_ctpm-sem_ctpm, mean_ctpm+sem_ctpm, alpha = 0.2, facecolor = col)
        p.ylabel('Cum. trials per minute') 
        p.xlabel('Time (mins)')    
        p.xlim(1,ses_dur)
        p.ylim(ymin = 0)
    else:
        p.xlabel('Time (mins)')  

def ITI_poke_timings(sessions, fig_no = 1):
    'Plots the timing of central pokes relative to the inter-trail interval, averaged over multiple sessions.'
    center_poke_TS_hists,  first_poke_TS_hists,  ITI_poke_TS_hists, \
    center_poke_ITI_hists, first_poke_ITI_hists, ITI_poke_ITI_hists, cum_fp_hists = ([], [], [], [], [], [], [])
    for session in sessions:
        center_poke_TS_hist,  first_poke_TS_hist,  ITI_poke_TS_hist, \
        center_poke_ITI_hist, first_poke_ITI_hist, ITI_poke_ITI_hist, cum_fp_hist, bin_edges = _ITI_analysis(session)
        center_poke_TS_hists.append(center_poke_TS_hist)
        first_poke_TS_hists.append(first_poke_TS_hist)
        ITI_poke_TS_hists.append(ITI_poke_TS_hist)
        first_poke_ITI_hists.append(first_poke_ITI_hist)
        center_poke_ITI_hists.append(center_poke_ITI_hist)
        ITI_poke_ITI_hists.append(ITI_poke_ITI_hist)
        cum_fp_hists.append(cum_fp_hist)
    print 'Fraction ITI without poke: {}'.format(np.mean([successful_delay_fraction(s) for s in sessions]))
    p.figure(fig_no)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    p.clf()
    p.subplot(3,1,1)
    p.fill_between(bin_centers, np.mean(center_poke_TS_hists, 0) / np.sum(first_poke_TS_hists), color = 'b')
    p.fill_between(bin_centers, np.mean(first_poke_TS_hists,  0) / np.sum(first_poke_TS_hists), color = 'g')
    p.xlim(bin_centers[0], bin_centers[-1])
    p.ylim(ymin = 0)
    p.xlabel('Time relative to trial start (sec)')
    p.subplot(3,1,2)
    p.fill_between(bin_centers, np.mean(center_poke_ITI_hists, 0) / np.sum(first_poke_TS_hists), color = 'b')
    p.fill_between(bin_centers, np.mean(first_poke_ITI_hists,  0) / np.sum(first_poke_TS_hists), color = 'g')
    p.xlim(bin_centers[0], bin_centers[-1])
    p.ylim(ymin = 0)
    p.xlabel('Time relative to ITI start (sec)')
    p.subplot(3,2,5)
    p.plot(bin_centers, np.mean(cum_fp_hists, 0))
    p.xlim(0,bin_centers[-1])
    p.ylim(0,1)
    p.xlabel('Reaction time (sec)')
    p.ylabel('Fraction of trials')

def _ITI_analysis(session, max_delta_t = 2.5, resolution = 0.01):
    'Evaluates the timing of central pokes relative to the inter-trail interval for single session.'
    assert 'ITI_start' in session.IDs.keys(), 'Session does not have inter-trial interval'
    # Get time stamps for relevent events
    center_pokes = session.time_stamps[(session.event_codes == session.IDs['high_poke']) | \
                                            (session.event_codes == session.IDs['low_poke' ])]
    trial_starts = session.time_stamps[(session.event_codes == session.IDs['trial_start'])]
    ITI_starts   = session.time_stamps[(session.event_codes == session.IDs['ITI_start'])]
    first_pokes_of_trial = []
    for trial_start in trial_starts[:-1]:
        first_pokes_of_trial.append(center_pokes[np.argmax(center_pokes > trial_start)])
    pokes_during_ITI  = np.array([])
    ITI_poke_delta_ts = np.array([])
    for ITI_start, trial_start in zip(ITI_starts, trial_starts[1:]):
        pokes_during_this_ITI = center_pokes[(center_pokes > ITI_start) & (center_pokes < trial_start)]
        pokes_during_ITI = np.append(pokes_during_ITI, pokes_during_this_ITI)
        ITI_poke_delta_ts = np.append(ITI_poke_delta_ts, pokes_during_this_ITI[:1] - ITI_start)
    # histograms with respect to trial start.
    center_poke_TS_hist, bin_edges = _PETH(center_pokes, trial_starts, max_delta_t, resolution) #PETH of center pokes wrt trial starts.
    first_poke_TS_delta_ts = first_pokes_of_trial - trial_starts[:len(first_pokes_of_trial)]
    first_poke_TS_hist = np.histogram(first_poke_TS_delta_ts, bin_edges)[0]                     #PETH of first poke of trial wrt trial start.
    ITI_poke_TS_hist = _PETH(pokes_during_ITI, trial_starts, max_delta_t, resolution)[0]        #PETH of pokes during ITI wrt trial start.
    # histograms with respect to ITI start.
    center_poke_ITI_hist  = _PETH(center_pokes, ITI_starts, max_delta_t, resolution)[0]         #PETH of center pokes wrt ITI starts.
    first_poke_ITI_delta_ts = first_pokes_of_trial[1:] - ITI_starts[:len(first_pokes_of_trial)-1]
    first_poke_ITI_hist = np.histogram(first_poke_ITI_delta_ts, bin_edges)[0]                   #PETH of first poke of trial wrt ITI start.
    ITI_poke_ITI_hist = np.histogram(ITI_poke_delta_ts, bin_edges)[0]                           #PETH of pokes during ITI wrt ITI start.
    # Cumulative histogram of first pokes relative to trial start.    
    cum_fp_hist = np.cumsum(first_poke_ITI_hist) / float(session.n_trials)
    return (center_poke_TS_hist,  first_poke_TS_hist,  ITI_poke_TS_hist, \
            center_poke_ITI_hist, first_poke_ITI_hist, ITI_poke_ITI_hist, cum_fp_hist, bin_edges)


def _latencies(event_times_A, event_times_B):
    'Evaluate the latency between each event A and the first event B that occurs afterwards.'                
    latencies = np.outer(event_times_B, np.ones(len(event_times_A))) - \
                np.outer(np.ones(len(event_times_B)), event_times_A)
    latencies[latencies <= 0] = np.inf
    latencies = np.min(latencies,0)
    return latencies

def _PETH(time_stamps_1, time_stamps_2, max_delta_t, resolution):
    'Peri-event time histogram'
    delta_ts = np.tile(time_stamps_1,(len(time_stamps_2),1)).T - \
               np.tile(time_stamps_2,(len(time_stamps_1),1))
    delta_ts = delta_ts = delta_ts[abs(delta_ts) < max_delta_t]
    return np.histogram(delta_ts, np.arange(-max_delta_t, max_delta_t + resolution, resolution))
    
def microscope_trigger_test(sessions, sleep_time = 10., max_time = 20.):
    '''Evaluate number of trials that would be recorded for microscope triggering
    rule which turns on scope on trial start, turns off scope after sleep_time (seconds)
    without trial start, and allows max_time (minutes) of total recording.   
    '''
    n_trials_continuous = []
    n_trials_with_sleep = []
    for session in sessions:
        trial_start_times = session.time_stamps[session.event_codes == session.IDs['trial_start']]
        n_trials_continuous.append(sum(trial_start_times < (max_time * 60.)))
        ITIs = trial_start_times[1:] - trial_start_times[:-1]
        ITIs[ITIs > sleep_time] = sleep_time
        ITIs = ITIs[np.cumsum(ITIs) < (max_time * 60.)]
        n_trials_with_sleep.append(sum(~ (ITIs == sleep_time)))
    print('Average trials recorded continuous: {}'.format(np.mean(n_trials_continuous)))
    print('Average trials recorded with sleep: {}'.format(np.mean(n_trials_with_sleep)))


def successful_delay_fraction(session):
    'Evaluates fraction of trials on which animal witholds central poking during ITI'
    assert 'ITI_start' in session.IDs.keys(), 'Session does not have inter-trial interval'
    event_IDs =  ut.get_IDs(session.IDs, ['low_poke','high_poke', 'trial_start', 'ITI_start'])
    ev_seq = [ev for ev in session.event_codes if ev in event_IDs]
    ev_seq.append(-1) # To avoid index out of range errors at next line.
    events_following_ITI_start = [ev_seq[i+1] for i, x in enumerate(ev_seq) if x == session.IDs['ITI_start']]
    return np.mean(np.array(events_following_ITI_start) == session.IDs['trial_start'])


#----------------------------------------------------------------------------------
# Choice probability trajectory analyses.
#----------------------------------------------------------------------------------

def reversal_analysis(sessions, pre_post_trials = [-15,40], last_n = 15, fig_no = 3,
                      return_fits = False, clf = True, cols = 0, by_type = True):
    '''Analysis of choice trajectories around reversals in reward probability and
    transition proability.  Fits exponential decay to choice trajectories following reversals.'''
    p_1 = _end_of_block_p_correct(sessions, last_n)

    if by_type: # Analyse reversals in reward and transition probabilities seperately.
        choice_trajectories_rr = _get_choice_trajectories(sessions, 'reward_reversal', pre_post_trials)
        fit_rr = _fit_exp_to_choice_traj(choice_trajectories_rr, p_1, pre_post_trials, last_n)
        choice_trajectories_tr = _get_choice_trajectories(sessions, 'reward_unchanged', pre_post_trials)
        fit_tr = _fit_exp_to_choice_traj(choice_trajectories_tr, p_1,pre_post_trials, last_n)
    else:
        fit_rr, fit_tr = (None, None)

    choice_trajectories_br = _get_choice_trajectories(sessions, 'any_reversal', pre_post_trials)
    fit_br = _fit_exp_to_choice_traj(choice_trajectories_br, p_1, pre_post_trials, last_n)

    if return_fits:
        return {'p_1'      :p_1,
                'rew_rev'  :fit_rr,
                'trans_rev':fit_tr,
                'both_rev' :fit_br}
    else:
        colors = (('c','b'),('y','r'))[cols]
        p.figure(fig_no)
        if clf:p.clf()   
        if by_type:
            p.subplot(3,1,1)
            p.title('Reversal in reward probabilities', fontsize = 'small')
            _plot_mean_choice_trajectory(choice_trajectories_rr, pre_post_trials, colors[0])
            _plot_exponential_fit(fit_rr, p_1, pre_post_trials, last_n, colors[1])
            p.subplot(3,1,2)
            p.title('Reversal in transition probabilities', fontsize = 'small')
            _plot_mean_choice_trajectory(choice_trajectories_tr, pre_post_trials, colors[0])
            _plot_exponential_fit(fit_tr, p_1, pre_post_trials, last_n, colors[1])
            p.subplot(3,1,3)
            p.title('Both reversals combined', fontsize = 'small')
        _plot_mean_choice_trajectory(choice_trajectories_br, pre_post_trials, colors[0])
        _plot_exponential_fit(fit_br, p_1, pre_post_trials, last_n, colors[1])
        p.xlabel('Trials relative to block transition.')
        p.ylabel('Fraction of choices to pre-reversal correct side.')
        print('Average block end choice probability: {}'.format(p_1))
        if by_type:
            print('Reward probability reversal, tau: {}, P_0: {}'.format(fit_rr['tau'], fit_rr['p_0']))
            print('Trans. probability reversal, tau: {}, P_0: {}'.format(fit_tr['tau'], fit_tr['p_0']))
        print('Combined reversals,          tau: {}, P_0: {}'.format(fit_br['tau'], fit_br['p_0']))

def _block_index(blocks):
    '''Create dict of boolean arrays used for indexing block transitions,
    Note first value of index corresponds to second block of session.'''
    return {
    'transition_reversal' : np.array(blocks['transition_states'][:-1]) != np.array(blocks['transition_states'][1:]),
    'to_neutral'          : np.array(blocks['reward_states'][1:]) == 1,
    'from_neutral'        : np.array(blocks['reward_states'][:-1]) == 1,
    'reward_reversal'     : np.abs(np.array(blocks['reward_states'][:-1]) - np.array(blocks['reward_states'][1:])) == 2,
    'reward_unchanged'    : np.array(blocks['reward_states'][:-1]) == np.array(blocks['reward_states'][1:]),
    'any_reversal'        : (np.abs(np.array(blocks['reward_states'][:-1]) - np.array(blocks['reward_states'][1:])) == 2) | \
                            (np.array(blocks['reward_states'][:-1]) == np.array(blocks['reward_states'][1:]))}

def _get_choice_trajectories(sessions, trans_type, pre_post_trials):
    '''Evaluates choice trajectories around transitions of specified type. Returns float array
     of choice trajectories of size (n_transitions, n_trials). Choices are coded such that a 
    choice towards the option which is correct before the transition is 1, the other choice is 0,
    if the choice trajectory extends past the ends of the blocks before and after the transition
    analysed, it is padded with nans.'''
    choice_trajectories = []
    n_trans_analysed = 0
    n_trials = pre_post_trials[1] - pre_post_trials[0]
    for session in sessions:
        blocks = session.blocks
        selected_transitions = _block_index(blocks)[trans_type]
        n_trans_analysed +=sum(selected_transitions)
        start_trials = np.array(blocks['start_trials'][1:])[selected_transitions] # Start trials of blocks following selected transitions.
        end_trials = np.array(blocks['end_trials'][1:])[selected_transitions]     # End trials of blocks following selected transitions.
        prev_start_trials = np.array(blocks['start_trials'][:-1])[selected_transitions] # Start trials of blocks preceding selected transitions.
        transition_states = np.array(blocks['transition_states'][:-1])[selected_transitions] # Transition state of blocks following selected transitions.
        reward_states = np.array(blocks['reward_states'][:-1])[selected_transitions] # Reward state of blocks following selected transitions.

        for     start_trial,  end_trial,  prev_start_trial,  reward_state,  transition_state in \
            zip(start_trials, end_trials, prev_start_trials, reward_states, transition_states):

            trial_range = start_trial + np.array(pre_post_trials)
            if trial_range[0] < prev_start_trial:
                pad_start = prev_start_trial - trial_range[0] 
                trial_range[0] = prev_start_trial
            else:
                pad_start = 0
            if trial_range[1] > end_trial:
                pad_end = trial_range[1] - end_trial
                trial_range[1] = end_trial
            else:
                pad_end = 0
            choice_trajectory = session.CTSO['choices'][trial_range[0]:trial_range[1]].astype(bool)                        
            choice_trajectory = (choice_trajectory ^ bool(reward_state) ^ bool(transition_state)).astype(float)
            if pad_start:
                choice_trajectory = np.hstack((ut.nans(pad_start), choice_trajectory))
            if pad_end:
                choice_trajectory = np.hstack((choice_trajectory, ut.nans(pad_end)))
            choice_trajectories.append(choice_trajectory)
    return np.vstack(choice_trajectories)

def _plot_mean_choice_trajectory(choice_trajectories, pre_post_trials, col = 'b'):
    p.plot(range(pre_post_trials[0], pre_post_trials[1]), np.nanmean(choice_trajectories,0),col)
    p.plot([0,0],[0,1],'k--')
    p.plot([pre_post_trials[0], pre_post_trials[1]-1],[0.5,0.5],'k:')
    p.ylim(0,1)
    p.xlim(pre_post_trials[0],pre_post_trials[1])

def _plot_exponential_fit(fit, p_1, pre_post_trials, last_n, col = 'r'):
    t = np.arange(0,pre_post_trials[1])
    p_traj = np.hstack([ut.nans(-pre_post_trials[0]-last_n), np.ones(last_n) * fit['p_0'], \
                   _exp_choice_traj(fit['tau'], fit['p_0'], p_1, t)])
    p.plot(range(pre_post_trials[0], pre_post_trials[1]),p_traj, col, linewidth = 2)
    p.locator_params(nbins = 4)

def _end_of_block_p_correct(sessions, last_n = 15):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks.'
    n_correct, n_trials = (0, 0)
    for session in sessions:
        block_end_trials = session.select_trials('exc', last_n, block_type = 'non_neutral')
        n_trials += sum(block_end_trials)
        correct_choices = session.CTSO['choices'] ^ \
                          np.array(session.blocks['trial_rew_state'],   bool) ^ \
                          np.array(session.blocks['trial_trans_state'], bool)
        n_correct += sum(correct_choices[block_end_trials])
    p_correct = n_correct / float(n_trials)
    return p_correct

def _fit_exp_to_choice_traj(choice_trajectories, p_1, pre_post_trials,  last_n):
    '''Fit an exponential curve to the choice trajectroy following a block transition
    using maximum likelihood.  The only parameter that is adjusted is the time constant,
    the starting value is determined by the mean choice probability in the final last_n trials
    before the transition, and the asymptotic choice  probability is given by  (1 - p_1).
    '''

    n_traj = np.sum(~np.isnan(choice_trajectories),0)  # Number of trajectories at each timepoint.
    n_post = n_traj[-pre_post_trials[1]:]  # Section folowing transtion.
    if min(n_traj) == 0:
        return {'p_0':np.nan,'tau':np.nan}
    t = np.arange(0,pre_post_trials[1])
    sum_choices = np.nansum(choice_trajectories, 0)
    p_0 = np.sum(sum_choices[-pre_post_trials[1]-last_n:-pre_post_trials[1]]) / float( # Choice probability at end of previous block.
          np.sum(     n_traj[-pre_post_trials[1]-last_n:-pre_post_trials[1]]))
    q = sum_choices[-pre_post_trials[1]:]  # Number of choices to previously correct side at different timepoints relative to block transition.
    fits = np.zeros([3,2])
    for i, x_0 in enumerate([10.,20.,40.]):   # Multiple starting conditions for minimization. 
        minimize_output = minimize(_choice_traj_likelihood, np.array([x_0]),
                                    method = 'Nelder-Mead', args = (p_0, p_1, q, n_post, t))
        fits[i,0] = minimize_output['x'][0]
        fits[i,1] = minimize_output['fun']
    tau_est = fits[0,np.argmin(fits[1,:])] # Take fit with highest likelihood.
    return {'p_0':p_0,'tau':tau_est}

def _choice_traj_likelihood(tau, p_0, p_1, q, n, t):
    if tau < 0: return np.inf
    p_traj = _exp_choice_traj(tau, p_0, p_1, t)
    log_lik = binom.logpmf(q,n,p_traj).sum()
    return -log_lik

def _exp_choice_traj(tau, p_0, p_1, t):
    return (1. - p_1) + (p_0 + p_1 - 1.) * np.exp(-t/tau)

def per_animal_end_of_block_p_correct(sessions, last_n = 15, fig_no = 1, col = 'b', clf = True):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks on a per animals basis.'
    p_corrects = []
    for sID in sorted(set([s.subject_ID for s in sessions])):
        p_corrects.append(_end_of_block_p_correct([s for s in sessions if s.subject_ID == sID]))
    p.figure(fig_no)
    if clf: p.clf()
    n_sub = len(p_corrects)
    p.scatter(0.2*np.random.rand(n_sub),p_corrects, s = 8,  facecolor= col, edgecolors='none', lw = 0)
    p.errorbar(0.1, np.mean(p_corrects), np.sqrt(np.var(p_corrects)/n_sub),linestyle = '', marker = '', linewidth = 2, color = col)
    p.xlim(-1,1)
    p.xticks([])
    p.ylabel('Prob. correct choice')
    return p_corrects


def session_start_analysis(sessions, first_n = 40):
    'Analyses choice trajectories following session start.'
    choice_trajectories = []
    for session in sessions:
        reward_state = session.blocks['reward_states'][0]
        if not reward_state == 1: # don't analyse sessions that start in neutral block. 
            transition_state = session.blocks['transition_states'][0]
            choice_trajectory = session.CTSO['choices'][:first_n]
            choice_trajectory = choice_trajectory ^ bool(reward_state) ^ bool(transition_state)  
            choice_trajectories.append(choice_trajectory)
    p.figure(1)
    p.clf()
    p.plot(np.mean(choice_trajectories,0))
    p.xlabel('Trial number')
    p.ylabel('Choice Probability')


#----------------------------------------------------------------------------------
# Stay probability Analysis
#----------------------------------------------------------------------------------

def stay_probability_analysis(sessions, ebars = 'SEM', selection = 'inc', fig_no = 1, ymin = 0):
    '''Stay probability analysis for task version with reversals in transition matrix.
    '''
    assert ebars in [None, 'SEM', 'SD'], 'Invalid error bar specifier.'
    n_sessions = len(sessions)
    all_n_trials, all_n_stay = (np.zeros([n_sessions,12]), np.zeros([n_sessions,12]))
    for i, session in enumerate(sessions):
        trial_select = session.select_trials(selection)
        trial_select_A = trial_select &  session.blocks['trial_trans_state']
        trial_select_B = trial_select & ~session.blocks['trial_trans_state']
        #Eval total trials and number of stay trial for A and B blocks.
        all_n_trials[i,:4], all_n_stay[i,:4] = _stay_prob_analysis(session.CTSO, trial_select_A)
        all_n_trials[i,4:8], all_n_stay[i,4:8] = _stay_prob_analysis(session.CTSO, trial_select_B)
        # Evaluate combined data.
        all_n_trials[i,8:] = all_n_trials[i,:4] + all_n_trials[i,[5,4,7,6]]
        all_n_stay[i,8:] = all_n_stay[i,:4] + all_n_stay[i,[5,4,7,6]]
    if not ebars: # Don't calculate cross-animal error bars.
        mean_stay_probs = np.nanmean(all_n_stay / all_n_trials, 0)
        y_err  = np.zeros(12)
    else:
        session_sIDs = np.array([s.subject_ID for s in sessions])
        unique_sIDs = list(set(session_sIDs))
        n_subjects = len(unique_sIDs)
        per_animal_stay_probs = np.zeros([n_subjects,12])
        for i, sID in enumerate(unique_sIDs):
            session_mask = session_sIDs == sID # True for sessions with correct animal ID.
            per_animal_stay_probs[i,:] = sum(all_n_stay[session_mask,:],0) / sum(all_n_trials[session_mask,:],0)
        mean_stay_probs = np.nanmean(per_animal_stay_probs, 0)
        var_stay_probs  =  np.nanvar(per_animal_stay_probs, 0)
        if ebars == 'SEM':
            y_err = np.sqrt(var_stay_probs/n_subjects)
        else:
            y_err = np.sqrt(var_stay_probs)

    p.figure(fig_no)
    #p.clf()
    p.subplot(1,3,1)
    p.bar(np.arange(1,5), mean_stay_probs[:4], yerr = y_err[:4])
    p.ylim(ymin,1)
    p.xlim(0.75,5)
    p.title('A transitions normal.', fontsize = 'small')
    p.xticks([1.5,2.5,3.5,4.5],['1/A', '1/B', '0/A', '0/B'])
    p.ylabel('Stay Probability')
    p.subplot(1,3,2)
    p.bar(np.arange(1,5), mean_stay_probs[4:8], yerr = y_err[4:8])
    p.ylim(ymin,1)
    p.xlim(0.75,5)
    p.title('B transitions normal.', fontsize = 'small')
    p.xticks([1.5,2.5,3.5,4.5],['1/A', '1/B', '0/A', '0/B'])
    p.subplot(1,3,3)
    p.bar(np.arange(1,5), mean_stay_probs[8:], yerr = y_err[8:])
    p.ylim(ymin,1)
    p.xlim(0.75,5)
    p.title('Combined.', fontsize = 'small')
    p.xticks([1.5,2.5,3.5,4.5],['1/N', '1/R', '0/N', '0/R'])
    return mean_stay_probs

def _stay_prob_analysis(CTSO, trial_select):
    'Analysis for stay probability plots using binary mask to select trials.'
    choices, transitions, outcomes = ut.CTSO_unpack(CTSO, 'CTO', bool)
    stay = choices[1:] == choices[:-1]
    transitions, outcomes, trial_select = (transitions[:-1], outcomes[:-1], trial_select[:-1])
    stay_go_by_type = [stay[( outcomes &  transitions) & trial_select],  # A transition, rewarded.
                       stay[( outcomes & ~transitions) & trial_select],  # B transition, rewarded.
                       stay[(~outcomes &  transitions) & trial_select],  # A transition, not rewarded.
                       stay[(~outcomes & ~transitions) & trial_select]]  # B transition, not rewarded.
    n_trials_by_type = [len(s) for s in stay_go_by_type]
    n_stay_by_type =   [sum(s) for s in stay_go_by_type]
    return n_trials_by_type, n_stay_by_type

#----------------------------------------------------------------------------------
# Multi - pannel plots.
#----------------------------------------------------------------------------------

def plot_day(exp, day, full):
    if day < 0: day = exp.n_days + day + 1
    day_sessions = [s for s in exp.sessions if s.day == day]
    sID0 = exp.subject_IDs[0] - 1

    for i, s in enumerate(day_sessions):
        choice_mov_ave(s, (day, [exp.n_subjects,1,     i+1]))
        if full:
            log_IPI(s, (day+100, [exp.n_subjects, 4, 4*(i+1) - 1]))
            poke_poke_corrlations(s.IDs, s.event_codes, (day+100, [exp.n_subjects, 4, 4*(i+1)]))

            pokes_per_min(s, (day+200, [exp.n_subjects, 1, (i+1)]))
   
def plot_subject(exp, sID):
    subject_sessions =  exp.get_sessions(sID, 'all')
    session_numbers = [s.number for s in subject_sessions]
    sorted_sessions = [s for (n,s) in sorted(zip(session_numbers, subject_sessions))]
    for i,s in enumerate(sorted_sessions):
        choice_mov_ave(s, (sID, [len(sorted_sessions),1, i+1]))

def plot_session(session, fig_no = 1):
    'Plot data from a single session.'
    p.figure(fig_no)
    p.clf()
    if isinstance(session, human_session):
        choice_mov_ave(session, 'current_axis')
    else:
        if hasattr(session, 'reward_probs'): 
            n_rows = 3
            p.subplot2grid((n_rows,4), (1,0), colspan = 4)
            p.plot(np.arange(len(session.reward_probs)), session.reward_probs[:,0], 'k')
            p.plot(np.arange(len(session.reward_probs)), session.reward_probs[:,1], 'r')
            p.xlim(0,len(session.reward_probs))
        else:
            n_rows = 2
        p.subplot2grid((n_rows,4), (0,0), colspan = 4)
        choice_mov_ave(session, 'current_axis')
        if hasattr(session, 'IDs'): #Plots not available for simulated sessions.
            p.subplot2grid((n_rows,4), (n_rows - 1, 2))
            log_IPI(session, 'current_axis')
            p.subplot2grid((n_rows,4), (n_rows - 1, 3))
            poke_poke_corrlations(session.IDs, session.event_codes, 'current_axis')

