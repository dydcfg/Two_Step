import numpy as np
import datetime
from scipy import signal
from copy import deepcopy


def get_event_times(time_stamps, event_codes, IDs):
    'Return dictionary with times at which events in IDs occured.'
    event_times = {}
    for event, code in IDs.iteritems():
        times = time_stamps[event_codes == code]
        event_times[event] = times
    return event_times

def get_IDs(IDs, event_list):
    return [IDs[val] for val in event_list]

def event_name(event_code, IDs):
    'Get event name from event ID.'
    return IDs.keys()[IDs.values().index(event_code)]

def exp_mov_ave(data, tau = 8., initValue = 0.5):
    'Exponential Moving average for 1d data.'
    m = np.exp(-1./tau)
    i = 1 - m
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for k, sample in enumerate(data):
        mov_ave[k+1] = mov_ave[k] * m + i * sample 
    return mov_ave[1::]

def get_day_number(my_date,start_date):
    'Converts date strings to day numbers with start_date = 1.'
    d = datetime.datetime.strptime(my_date, '%Y-%m-%d')
    d_0 = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    return (d-d_0).days + 1

def array_contains(x,y):
    'Returns boolean array with shape of y that is true where x contains y.'
    return np.array([x.__contains__(e) for e in y])

def CTSO_unpack(CTSO, order = 'CTSO', dtype = int):
    'Return elements of CTSO dictionary in specified order and data type.'
    o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
    if dtype == int:
        return [CTSO[o_dict[i]] for i in order]
    else:
        return [CTSO[o_dict[i]].astype(dtype) for i in order]

def nans(shape, dtype=float):
    'return array of nans of specified shape.'
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def projection(u,v):
    '''For vectors (np.array) u and v, returns the projection of v along.
    '''
    u_dot_u = np.dot(u,u)
    if  u_dot_u == 0:
        return np.zeros(len(u))
    else:
        return u*np.dot(u,v)/u_dot_u

def check_common_transition_fraction(sessions):
    ''' Sanity check that common transitions are happening at correct frequency.
    '''
    sIDs = set([s.subject_ID for s in sessions])
    for sID in sIDs:
        a_sessions = [s for s in sessions if s.subject_ID == sID]
        transitions_CR = np.hstack([s.CTSO['transitions'] == s.blocks['trial_trans_state'] for s in a_sessions])
        print('Subject: {}  mean transitions: {}'.format(sID, np.mean(transitions_CR)))

def norm_correlate(a, v, mode='Full'):
    'Calls numpy correlate after normaising the inputs.'
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) /  np.std(v)
    return np.correlate(a, v, mode)


def select_trials(session, selection_type = 'inc', last_n = 20, first_n_mins = False,
                  block_type = 'all'):
    ''' Select specific trials for analysis.  
    The first selection step is specified by selection_type:
    'inc' :  Inclusive selection, only trials excluded are those from reversal in
             transition matrix to last_n trials before next block transition.
    'exc' : Exclusive selection, only last_n trials of each block are selected.
    'all' : All trials are included.

    The first_n_mins argument can be used to select only trials occuring within
    a specified number of minutes of the session start.

    Note - Function is placed outside of class so it can also be used by simulated session class.
'''

    assert selection_type in ['inc', 'exc', 'mor', 'all'], 'Invalid trial select type.'

    if selection_type == 'inc': # Select all trials except those from transition reversal to last_n 
                                # before end of next block.
        trials_to_use = np.ones(session.n_trials, dtype = bool)
        trans_change = np.hstack((True, ~np.equal(session.blocks['transition_states'][:-1], \
                                                  session.blocks['transition_states'][1:])))
        start_trials = session.blocks['start_trials'] + [session.blocks['end_trials'][-1] + last_n]
        for i in range(len(trans_change)):
            if trans_change[i]:
                trials_to_use[start_trials[i] - 1:start_trials[i+1]-last_n -1] = False

    if selection_type == 'mor': # Select all trials except nlast_n following transition reversal.
        trials_to_use = np.ones(session.n_trials, dtype = bool)
        trans_change = np.hstack((True, ~np.equal(session.blocks['transition_states'][:-1], \
                                                  session.blocks['transition_states'][1:])))
        start_trials = session.blocks['start_trials'] + [session.blocks['end_trials'][-1] + last_n]
        trials_to_use[0] = False
        for i in range(len(trans_change)):
            if trans_change[i]:
                trials_to_use[start_trials[i]:start_trials[i] + last_n] = False

    elif selection_type == 'exc':
        # Select only last_n trials before block transitions.
        trials_to_use = np.zeros(session.n_trials, dtype = bool)
        for b in session.blocks['start_trials'][1:]:
            trials_to_use[b - 1 - last_n:b -1] = True

    elif selection_type == 'all':
        # Use all trials.
        trials_to_use = np.ones(session.n_trials, dtype = bool)
        
    if first_n_mins:  #  Restrict analysed trials to only first n minutes. 
        time_selection = session.trial_start_times[:session.n_trials] < (60 * first_n_mins)
        trials_to_use = trials_to_use & time_selection

    if not block_type == 'all': #  Restrict analysed trials to blocks of certain types.
        block_selection = np.zeros(session.n_trials, dtype = bool)
        if block_type == 'neutral':       # Include trials only from neutral blocks.
            blocks_to_use = np.where( np.array(session.blocks['reward_states']) == 1)[0]
        elif block_type == 'non_neutral': # Include trials only from non-neutral blocks.
            blocks_to_use = np.where(~(np.array(session.blocks['reward_states']) == 1))[0]
        for n in blocks_to_use:
            block_selection[session.blocks['start_trials'][n]-1:session.blocks['end_trials'][n]] = True
        trials_to_use = trials_to_use & block_selection

    return trials_to_use