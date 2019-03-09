# utilitary functions to create the expert and volunteers oracles from the taskruns dataset

import pandas as pd

from modules.utils import aux_functions
from modules.utils import firefox_dataset_p2 as fd

class Br_Feat_Oracle_Creator:
    
    def __init__(self, bugreports, features):
        self.bugreports = bugreports
        self.features = features
    
    def __create_exp_feat_br_matrix(self, expert_taskruns):
        taskruns_expert = aux_functions.shift_taskruns_answers(expert_taskruns)
        taskruns_expert.sort_values(by='bug_id', inplace=True)

        feat_br_matrix = pd.DataFrame(columns=self.features.feat_name.values, 
                            index=self.bugreports.Bug_Number)
        feat_br_matrix.index.names = ['bug_number']
        
        for idx,row in taskruns_expert.iterrows():
            ans = row.new_answers.split(" ")
            for i in range(len(ans)):
                feat_name = feat_br_matrix.columns[i]
                feat_br_matrix.at[row.bug_id, feat_name] = int(ans[i])
        
        return feat_br_matrix
    
    ## TESTS PURPOSES -------------
    def create_br_feat_expert_matrix(self, expert_taskruns):
        feat_br_matrix = self.__create_exp_feat_br_matrix(expert_taskruns)
        fd.Feat_BR_Oracles.write_feat_br_expert_df(feat_br_matrix)
    
    
    def create_br_feat_expert_2_matrix(self, expert_taskruns):
        feat_br_matrix = self.__create_exp_feat_br_matrix(expert_taskruns)
        fd.Feat_BR_Oracles.write_feat_br_expert_2_df(feat_br_matrix)
        
    
    def create_br_feat_volunteers_matrix(self, taskruns_volunteers_1, taskruns_volunteers_2):
        ignored_taskruns = [154,  155,  156,  157,  169,  170,  171,  172,  183,  184,  196,  
                    197,  198,  199,  200,  201,  202,  203,  204,  206,  241,  242,  
                    253,  264,  265,  266,  267,  268,  269,  270]
        
        taskruns_volunteers_1 = aux_functions.shift_taskruns_answers(taskruns_volunteers_1)
        taskruns_volunteers_2 = aux_functions.shift_taskruns_answers(taskruns_volunteers_2)
        
        taskruns = pd.concat([taskruns_volunteers_1, taskruns_volunteers_2])
        taskruns.sort_values(by='bug_id', inplace=True)
        
        not_ignored_taskruns = [t_id for t_id in taskruns.id.values if t_id not in ignored_taskruns]
        taskruns = taskruns[taskruns.id.isin(not_ignored_taskruns)]
        
        feat_br_matrix = pd.DataFrame(columns=self.features.feat_name.values, 
                    index=self.bugreports.Bug_Number)
        feat_br_matrix.index.names = ['bug_number']        

        for idx,row in taskruns.iterrows():
            ans = row.new_answers.split(" ")
            for i in range(len(ans)):
                feat_name = feat_br_matrix.columns[i]
                feat_br_matrix.at[row.bug_id, feat_name] = int(ans[i])

        fd.Feat_BR_Oracles.write_feat_br_volunteers_df_2(feat_br_matrix)
        