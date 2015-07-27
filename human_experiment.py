 # Class containing data from all sessions in one experiment.

import os
import pickle
import imp
import scipy.io as sio
import numpy as np
import human_session as hs
import utility as ut
import plotting as pl

class experiment:
    def __init__(self, exp_name, rebuild_sessions = False):
        '''
        Instantiate an experiment object for specified group number.  Tries to load previously 
        saved sessions,  then loads sessions from data folder if they were not in
        the list of loaded sessions and are from animals in the group.  rebuild sessions argument
        forces the sessions to be created directly from the data files rather than loaded.
        '''

        self.name = exp_name
        self.start_date = exp_name[:10]  

        data_sets = 'C:\\Users\\Thomas\\Documents\\Dropbox\Work\\Behavioural Tasks\\Two_step_experiments\\data sets\\'

        self.path = data_sets + exp_name + '\\'

        self.data_path = self.path + 'data\\'

        info = imp.load_source('info', self.path + 'info.py')

        self.IDs = info.IDs
        self.info = info.info
        self.subject_IDs = info.subject_IDs       

        self.sessions = []
        
        if not rebuild_sessions:
            try:
                exp_file = open(self.path + 'sessions.pkl','rb')
                self.sessions = pickle.load(exp_file)
                exp_file.close()
                print 'Saved sessions loaded from: sessions.pkl'
            except IOError:
               pass

        self.import_data()

        if rebuild_sessions:
            self.save()

    def save(self):
        'Save sessions from experiment.'
        exp_file = open(self.path + 'sessions.pkl','wb')
        pickle.dump(self.sessions, exp_file)
        exp_file.close()

    def save_item(self, item, file_name):
        'save an item to experiment folder using pickle.'
        f = open(self.path + file_name + '.pkl', 'wb')
        pickle.dump(item, f)
        f.close()

    def load_item(self, item_name):
        'Unpickle and return specified item from experiment folder.'
        f = open(self.path + item_name + '.pkl', 'rb')
        out = pickle.load(f)
        f.close()
        return out

    def import_data(self):
        '''Load new sessions as session class instances.'''

        old_files = [session.file_name for session in self.sessions]
        files = os.listdir(self.data_path)
        new_files = [f for f in files if f not in old_files and f[0].isdigit()]

        if len(new_files) > 0:
            print 'Loading new data files...'
            new_sessions = [hs.human_session(file_name,self.data_path,self.IDs) 
                            for file_name in new_files]

            self.sessions = self.sessions + new_sessions  

        self.n_subjects = len(np.unique([session.subject_ID for session in self.sessions]))

    def get_sessions(self, sIDs, numbers = []):
        '''Return list of sessions which match specified animal IDs and session numbers.
        All days or animals can be selected with input 'all'.
        The last n days can be selected with days = -n .
        '''
        if isinstance(sIDs, int): sIDs = [sIDs]
        if isinstance(numbers, int): numbers = [numbers]
        if numbers == 'all':
            numbers = list(set([s.number for s in self.sessions]))
        if sIDs == 'all':
            sIDs = self.subject_IDs
        valid_sessions = [s for s in self.sessions if s.number in numbers and s.subject_ID in sIDs]
        if len(valid_sessions) == 1: 
            valid_sessions = valid_sessions[0] # Don't return list for single session.
        return valid_sessions                
                 

    # Plotting.
    def plot_subject(self, sID): pl.plot_subject(self, sID) 




