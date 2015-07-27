import numpy as np
import utility as ut
import plotting as pl
import os

class session:
    'Class containing data from a single session.'
    def __init__(self, file_name, data_path, IDs):
        # Session information.
        np.disp(file_name)
        self.file_name = file_name
        self.subject_ID =  int(file_name.split('-',1)[0][1:])
        self.date = file_name.split('-',1)[1].split('.')[0]
        self.IDs = IDs

        # Import data.

        data_file = open(os.path.join(data_path, file_name), 'r')

        self.pyControl_file = 'Run started at:' in data_file.read() # File is from pyControl system.
        
        data_file.seek(0)
        data_lines = [line.strip() for line in data_file if line[0].isdigit()
                      and len(line.split(' ')) == (2 + self.pyControl_file)]
        data_file.seek(0)
        reward_prob_lines = [line.strip() for line in data_file if line[0:3] == 'Rew']
        data_file.seek(0)
        block_start_lines = [line.strip() for line in data_file if line[0].isdigit() \
                                and len(line.split(' ')) > 1
                                and line.split(' ')[1 + self.pyControl_file] == '-1']
        data_file.close()

        #  Convert lines to numpy arrays of timestamps (in seconds) and event IDs.

        time_stamps = np.array([int(line.split(" ")[0])  for line in data_lines])
        event_codes = np.array([int(line.split(" ")[-1]) for line in data_lines])

        if self.pyControl_file: # 
            self.time_stamps = time_stamps / 1000.
            self.duration = time_stamps[-1]
            self.event_codes = event_codes
            start_time_stamp = 0

        else:
            self._raw_time_stamps = time_stamps

            if 'start_stop' in IDs.keys():
                start_stop_inds=np.where(event_codes == IDs['start_stop'])[0]
                start_ind = start_stop_inds[0]
                stop_ind  = start_stop_inds[1]
            else:
                start_ind = np.where(event_codes == IDs['session_start'])[0][0]
                stop_ind  = np.where(event_codes == IDs['session_stop' ])[0][0]

            start_time_stamp = time_stamps[start_ind]

            time_stamps = (time_stamps - start_time_stamp)  / 1000.

            self.duration = time_stamps[stop_ind]

        # Extract reward probs if available.

            if len(reward_prob_lines) > 0:
                self.reward_probs = np.array([(float(line.split()[2]), float(line.split()[4]))
                                              for line in reward_prob_lines])

            # Store data and summary info.

            self.time_stamps = time_stamps[start_ind:stop_ind]
            self.event_codes = event_codes[start_ind:stop_ind]

        self.rewards = sum ( (self.event_codes == IDs['left_reward']) |
                             (self.event_codes == IDs['right_reward']) )

        self.actions = sum ( (self.event_codes == IDs['left_poke'])  |
                             (self.event_codes == IDs['right_poke']) |
                             (self.event_codes == IDs['high_poke'])  |
                             (self.event_codes == IDs['low_poke']) )

        self.make_CTSO_representation()  # Boolean arrays of choice, transition, outcome information.

        self.fraction_rewarded = float(self.rewards) /  self.n_trials

        self.trial_start_times = self.time_stamps[self.event_codes == IDs['trial_start']]

        # Extract block tranistion information if available:
   
        if len(block_start_lines) > 0:
            start_times, start_trials, reward_states, transition_states = ([], [], [], [])
            for line in block_start_lines:
                line = line.split(' ')
                start_times.append((int(line[0]) - start_time_stamp) / 1000.)
                start_trials.append(np.argmax(self.trial_start_times>=start_times[-1]))
                start_trials[0] = 0 # Timestamp for first block info follows first trial start, subsequent block info time stamps
                                    # preceed first trial of new block.
                reward_states.append(int(line[-2]))
                transition_states.append(int(line[-1]))
            reward_states = np.array(reward_states)
            transition_states = np.array(transition_states)
            trial_trans_state = np.zeros(self.n_trials, dtype = bool) # Boolean array indicating state of tranistion matrix for each trial.
            trial_rew_state   = np.zeros(self.n_trials, dtype = int)  # Integer array indicating state of rewared probabilities for each trial.
            end_trials = start_trials[1:] + [self.n_trials]
            for start_trial,end_trial, trans_state, reward_state in \
                    zip(start_trials, end_trials, transition_states, reward_states):
                trial_trans_state[start_trial:end_trial] = trans_state   
                trial_rew_state[start_trial:end_trial]   = reward_state   
            if self.pyControl_file:
                # Invert reward states for consistency with animal data. # Note this hack is here because in the original datasets
                # recorded on the arduino setups the reward states where in fact the opposite of that intended and so the analysis 
                # currently is set up for inverted reward states.
                trial_rew_state = 2 - trial_rew_state
                reward_states   = 2 - reward_states

            self.blocks = {'start_times'       : start_times,
                           'start_trials'      : start_trials, # index of first trial of blocks, first trial of session is trial 0. 
                           'end_trials'        : start_trials[1:] + [self.n_trials],
                           'reward_states'     : reward_states,      # 0 for left good, 1 for neutral, 2 for right good.
                           'transition_states' : transition_states,  # 1 for A blocks, 0 for B blocks.
                           'trial_trans_state' : trial_trans_state,
                           'trial_rew_state'   : trial_rew_state}          

    def make_CTSO_representation(self):
        '''
        Create choice, transition, second_step, outcome representation of session used for stay probability analysis
        and various plotting functions.

        Algorithm works by keeping track of the state the task is in, updating this estimate when relevent
        events happen, and appending choices, tranistions and outcomes when these are detected.  

        It is possible for events to appear to occur out of sequence, e.g a second link state entry
        occuring imediately after a trial start event, without an intervening poke.  This can happen 
        due to event queing in the framework that runs the task as follows: 

        1. Timer event occurs and is placed in que.
        2. Poke occurs and is placed in que.
        3. Timer event is processed, causing state transition.
        4. Poke event is processed in new state.

        When this sequence of events occurs the poke appears in the event list before the state transition,
        but is processed in the state following the transition.  The signature of this happening is a 
        seemingly  impossible series of state transitions preceded at a very small time interval by a poke
        event.  
        '''

        choices = []      # True if high poke, flase if low poke.
        transitions = []  # True if high --> left or low --> right  (A type), 
                          # flase if high --> right or low --> left (B type).
        second_steps = [] # True if left, false if right.
        outcomes = []     # True if rewarded,  flase if no reward.

        event_list = ['trial_start', 'low_poke','high_poke', 'left_active',
                      'right_active', 'left_reward', 'right_reward']

        state = 'I'  # C = choose,  T = transition, O = outcome, I = inter-trial-interval.


        state_IDs = {'C': ut.get_IDs(self.IDs, ['low_poke', 'high_poke']),
                     'T': ut.get_IDs(self.IDs, ['left_active', 'right_active']),
                     'O': ut.get_IDs(self.IDs, ['left_reward', 'right_reward','trial_start']),
                     'I': ut.get_IDs(self.IDs, ['trial_start'])}

        self.trial_start_inds = []

        for i, ev in enumerate(self.event_codes):
            if ev in state_IDs[state]: # Relevent event has occured.
                if state == 'C':    
                    choices.append( ev == self.IDs['high_poke'] )
                    state = 'T'
                elif state == 'T':
                    transitions.append( choices[-1] == (ev == self.IDs['left_active']) )
                    second_steps.append(ev == self.IDs['left_active'])
                    state = 'O'
                elif state == 'O':
                    outcomes.append( (ev == self.IDs['left_reward']) or (ev == self.IDs['right_reward']) )
                    if ev == self.IDs['trial_start']:
                        state = 'C'
                        self.trial_start_inds.append(i)
                    else:
                        state = 'I'      
                elif state == 'I':
                    state = 'C'
                    self.trial_start_inds.append(i)

            elif ev == self.IDs['trial_start']: 
                # Trial start event occurred out of sequence, print error message but ignore event.
                print('Out of sequence trial start, current state: {}, event #: {}, TS: {}'.format(state, i, 1000*self._raw_time_stamps[i]))
            elif ev in ut.get_IDs(self.IDs, ['left_active', 'right_active']):
                # Second link state entry occured out of sequence.
                if (ignored_event in ut.get_IDs(self.IDs, ['low_poke', 'high_poke'])) & (state == 'C'): 
                    # Trial start was logged after central poke that triggered transition to second link state.            
                    choices.append( ignored_event == self.IDs['high_poke'] )
                    transitions.append( choices[-1] == (ev == self.IDs['left_active']) )
                    second_steps.append(ev == self.IDs['left_active'])
                    state = 'O'
                else:
                    print('Out of sequence second link state, current state: {}, event #: {}, TS: {}'.format(state, i, 1000*self._raw_time_stamps[i]))
            elif ev in ut.get_IDs(self.IDs, ['left_reward', 'right_reward']):
                # Reward occured out of sequence.
                print('Out of sequence reward, current state: {}, event #: {}, TS: {}'.format(state, i, 1000*self._raw_time_stamps[i]))
            else:
                ignored_event = ev 
                ingnored_event_time = self.time_stamps[i]

        self.CTSO = {'choices'      : np.array(choices[0:len(outcomes)],int), # Store as integer arrays.
                     'transitions'  : np.array(transitions[0:len(outcomes)], int),
                     'second_steps' : np.array(second_steps[0:len(outcomes)], int),
                     'outcomes'     : np.array(outcomes, int)
                     }

        n_trial_starts = sum(self.event_codes == self.IDs['trial_start'])

        assert n_trial_starts - 1 <= len(outcomes) <= n_trial_starts, \
            'Incorrect number of trials found by make_CTSO_representation.'

        self.n_trials = len(outcomes)

    def plot(self): pl.plot_session(self)

    def remove_trials(self, first_n_mins):
        '''Remove trials occuring after first n minutes, used for analysing only begining
        of sessions.'''
        self.n_trials = sum(self.trial_start_times[:self.n_trials] < (60 * first_n_mins))
        for key in self.CTSO.keys():
            self.CTSO[key] = self.CTSO[key][:self.n_trials]
        for key in self.CSO.keys():
            self.CSO[key] = self.CSO[key][:self.n_trials]
        if hasattr(self,'blocks'): 
            n_blocks = sum(self.blocks['start_trials'] <= self.n_trials)
            for key in self.blocks.keys():
                if key[:5] == 'trial':
                    self.blocks[key] = self.blocks[key][:self.n_trials]
                else:
                    self.blocks[key] = self.blocks[key][:n_blocks]
            self.blocks['end_trials'][-1] = self.n_trials + 1

    def select_trials(self, selection_type = 'inc', last_n = 20, first_n_mins = False,
                      block_type = 'all'):
        return ut.select_trials(self, selection_type, last_n, first_n_mins, block_type)



class concatenated_session:
    '''Concatinates a set of consecutive sessions into a single 
    long session'''

    def __init__(self, subject_sessions):
        assert len(set([s.subject_ID for s in subject_sessions])) == 1, \
            'All sessions must be from a single subject to concatenate.'
        subject_sessions = sorted(subject_sessions, key = lambda s:s.day)
        choices, transitions, second_steps, outcomes, trial_trans_state, trial_rew_state = ([],[],[],[],[],[]) 
        self.n_trials = sum([s.n_trials for s in subject_sessions])
        session_start_trials = np.zeros(self.n_trials, bool)
        session_start_trial = 0
        for i, session in enumerate(subject_sessions):
                choices      += session.CTSO['choices'].tolist()
                transitions  += session.CTSO['transitions'].tolist()
                second_steps += session.CTSO['second_steps'].tolist()
                outcomes     += session.CTSO['second_steps'].tolist()
                trial_trans_state += session.blocks['trial_trans_state'].tolist() 
                trial_rew_state   += session.blocks['trial_rew_state'].tolist()            
                session_start_trials[session_start_trial] = True
                session_start_trial += session.n_trials

        self.CTSO = {'choices'      : np.array(choices),
                     'transitions'  : np.array(transitions),
                     'second_steps' : np.array(second_steps),
                     'outcomes'     : np.array(outcomes)
                     }

        self.blocks = {'trial_trans_state': np.array(trial_trans_state),
                       'trial_rew_state'  : np.array(trial_rew_state),
                       'session_start_trials': session_start_trials}

        self.IDs = subject_sessions[0].IDs
        self.subject_ID = subject_sessions[0].subject_ID
        self.n_trials = sum([s.n_trials for s in subject_sessions])