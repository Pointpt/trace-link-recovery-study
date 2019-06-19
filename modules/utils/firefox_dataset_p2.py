# functions for firefox_p2 (empirical study) ------

import pandas as pd
from enum import Enum

BASE_PATH = '/home/guilherme/anaconda3/envs/trace-link-recovery-study'

FEAT_X_BR_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/feat_br/'
TC_X_BR_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/tc_br/'

BUGREPORTS_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/'
TESTCASES_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/TC/'
FEATURES_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/'

TASKRUNS_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/taskruns/'

class FilePath(Enum):
    ORACLE_EXPERT_VOLUNTEERS_UNION = TC_X_BR_M_PATH + 'oracle_expert_volunteers_union.csv'
    ORACLE_EXPERT_VOLUNTEERS_INTERSEC = TC_X_BR_M_PATH + 'oracle_expert_volunteers_intersec.csv'
    ORACLE_EXPERT = TC_X_BR_M_PATH + 'oracle_expert.csv'
    ORACLE_VOLUNTEERS = TC_X_BR_M_PATH + 'oracle_volunteers.csv'
    
    FEAT_BR_EXPERT_VOLUNTEERS_UNION = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert_volunteers_union.csv'
    FEAT_BR_EXPERT_VOLUNTEERS_INTERSEC = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert_volunteers_intersec.csv'
    
    FEAT_BR_EXPERT = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert.csv'
    FEAT_BR_EXPERT_2 = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert_2.csv'
    FEAT_BR_VOLUNTEERS = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_volunteers.csv'
    
    FEAT_BR_MATRIX_FINAL = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_final.csv'
    
    TESTCASES = TESTCASES_M_PATH + 'testcases_final.csv'
    BUGREPORTS = BUGREPORTS_M_PATH + 'selected_bugreports_final.csv'
    FEATURES = FEATURES_M_PATH + 'features_final.csv'

    ORIG_FEATURES = FEATURES_M_PATH + 'features.csv'
    ORIG_BUGREPORTS = BUGREPORTS_M_PATH + 'bugreports_final.csv'
    
    EXPERT_TASKRUNS = TASKRUNS_M_PATH + 'taskruns_expert.csv'
    EXPERT_TASKRUNS_2 = TASKRUNS_M_PATH + 'taskruns_expert_2.csv'
    VOLUNTEERS_TASKRUNS_1 = TASKRUNS_M_PATH + 'taskruns_volunteers.csv'
    VOLUNTEERS_TASKRUNS_2 = TASKRUNS_M_PATH + 'taskruns_volunteers_2.csv'
    AUX_2_TASKRUNS = TASKRUNS_M_PATH + 'taskruns_aux_2.csv'

    
# TC_BR ORACLES --------------------------

class Tc_BR_Oracles:   
    def read_oracle_expert_volunteers_union_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_EXPERT_VOLUNTEERS_UNION.value)
        oracle_df.set_index('TC_Number', inplace=True, drop=True)
        oracle_df.columns.name = 'Bug_Number'
        oracle_df.rename(columns=lambda col: int(col), inplace=True)
        print('OracleExpertVolunteers_UNION.shape: {}'.format(oracle_df.shape))
        return oracle_df
    
    def read_oracle_expert_volunteers_intersec_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_EXPERT_VOLUNTEERS_INTERSEC.value)
        oracle_df.set_index('TC_Number', inplace=True, drop=True)
        oracle_df.columns.name = 'Bug_Number'
        oracle_df.rename(columns=lambda col: int(col), inplace=True)
        print('OracleExpertVolunteers_INTERSEC.shape: {}'.format(oracle_df.shape))
        return oracle_df
    
    
    def write_oracle_expert_volunteers_union_df(oracle_exp_vol_union_df):
        oracle_exp_vol_union_df.to_csv(FilePath.ORACLE_EXPERT_VOLUNTEERS_UNION.value)
        print('OracleExpertVolunteers_UNION.shape: {}'.format(oracle_exp_vol_union_df.shape))
    
    def write_oracle_expert_volunteers_intersec_df(oracle_exp_vol_intersec_df):
        oracle_exp_vol_intersec_df.to_csv(FilePath.ORACLE_EXPERT_VOLUNTEERS_INTERSEC.value)
        print('OracleExpertVolunteers_INTERSEC.shape: {}'.format(oracle_exp_vol_intersec_df.shape))
        
        
    def read_oracle_expert_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_EXPERT.value)
        oracle_df.set_index('TC_Number', inplace=True, drop=True)
        oracle_df.columns.name = 'Bug_Number'
        oracle_df.rename(columns=lambda col: int(col), inplace=True)
        print('OracleExpert.shape: {}'.format(oracle_df.shape))
        return oracle_df

    def write_oracle_expert_df(oracle_expert_df):
        oracle_expert_df.to_csv(FilePath.ORACLE_EXPERT.value)
        print('OracleExpert.shape: {}'.format(oracle_expert_df.shape))
    
    
    def read_oracle_volunteers_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_VOLUNTEERS.value)
        oracle_df.set_index('TC_Number', inplace=True, drop=True)
        oracle_df.columns.name = 'Bug_Number'
        oracle_df.rename(columns=lambda col: int(col), inplace=True)
        print('OracleVolunteers.shape: {}'.format(oracle_df.shape))
        return oracle_df
    
    def write_oracle_volunteers_df(oracle_volunteers_df):
        oracle_volunteers_df.to_csv(FilePath.ORACLE_VOLUNTEERS.value)
        print('OracleVolunteers.shape: {}'.format(oracle_volunteers_df.shape))

        
# FEAT_BR ORACLES ------------------------

class Feat_BR_Oracles:   
    def read_feat_br_expert_volunteers_union_df():
        feat_br_trace_df = pd.read_csv(FilePath.FEAT_BR_EXPERT_VOLUNTEERS_UNION.value)
        feat_br_trace_df.set_index('Bug_Number', inplace=True)
        print('Expert and Volunteers Matrix UNION.shape: {}'.format(feat_br_trace_df.shape))
        return feat_br_trace_df
    
    def read_feat_br_expert_volunteers_intersec_df():
        feat_br_trace_df = pd.read_csv(FilePath.FEAT_BR_EXPERT_VOLUNTEERS_INTERSEC.value)
        feat_br_trace_df.set_index('Bug_Number', inplace=True)
        print('Expert and Volunteers Matrix INTERSEC.shape: {}'.format(feat_br_trace_df.shape))
        return feat_br_trace_df
    
    
    def write_feat_br_expert_volunteers_union_df(feat_br_expert_volunteers_matrix):
        feat_br_expert_volunteers_matrix.to_csv(FilePath.FEAT_BR_EXPERT_VOLUNTEERS_UNION.value, index=True)
        print('Feat_BR Expert and Volunteers Matrix UNION.shape: {}'.format(feat_br_expert_volunteers_matrix.shape))
        
    def write_feat_br_expert_volunteers_intersec_df(feat_br_expert_volunteers_matrix):
        feat_br_expert_volunteers_matrix.to_csv(FilePath.FEAT_BR_EXPERT_VOLUNTEERS_INTERSEC.value, index=True)
        print('Feat_BR Expert and Volunteers Matrix INTERSEC.shape: {}'.format(feat_br_expert_volunteers_matrix.shape))
        

    def read_feat_br_expert_2_df():
        expert_matrix = pd.read_csv(FilePath.FEAT_BR_EXPERT_2.value)
        expert_matrix.set_index('bug_number', inplace=True)
        expert_matrix.sort_index(inplace=True)
        print('Feat_BR Expert 2 Matrix shape: {}'.format(expert_matrix.shape))
        return expert_matrix
    
    def write_feat_br_expert_2_df(feat_br_expert_matrix):
        feat_br_expert_matrix.to_csv(FilePath.FEAT_BR_EXPERT_2.value, index=True)
        print('Feat_BR Expert 2 Matrix shape: {}'.format(feat_br_expert_matrix.shape))
    
    
    def read_feat_br_expert_df():
        expert_matrix = pd.read_csv(FilePath.FEAT_BR_EXPERT.value)
        expert_matrix.set_index('bug_number', inplace=True)
        expert_matrix.sort_index(inplace=True)
        print('Feat_BR Expert Matrix shape: {}'.format(expert_matrix.shape))
        return expert_matrix
    
    def write_feat_br_expert_df(feat_br_expert_matrix):
        feat_br_expert_matrix.to_csv(FilePath.FEAT_BR_EXPERT.value, index=True)
        print('Feat_BR Expert Matrix shape: {}'.format(feat_br_expert_matrix.shape))
        
       
    def read_feat_br_volunteers_df():
        vol_matrix = pd.read_csv(FilePath.FEAT_BR_VOLUNTEERS.value)
        vol_matrix.set_index('bug_number', inplace=True)
        vol_matrix.sort_index(inplace=True)
        print('Feat_BR Volunteers Matrix shape: {}'.format(vol_matrix.shape))
        return vol_matrix
    
    def write_feat_br_volunteers_df(feat_br_volunteers_matrix):
        feat_br_volunteers_matrix.to_csv(FilePath.FEAT_BR_VOLUNTEERS.value, index=True)
        print('Feat_BR Volunteers Matrix shape: {}'.format(feat_br_volunteers_matrix.shape))
    
    
    ## selected_bug_reports_2 with the related features after the empirical study
    def read_br_2_features_matrix_final_df():
        br_2_feat_matrix = pd.read_csv(FilePath.FEAT_BR_MATRIX_FINAL.value, dtype=object)
        br_2_feat_matrix.set_index('Bug_Number', inplace=True)
        br_2_feat_matrix.sort_index(inplace=True)
        br_2_feat_matrix.fillna("", inplace=True)
        br_2_feat_matrix.drop(columns='Unnamed: 0', inplace=True)
        print('BR_2_Features Matrix Final.shape: {}'.format(br_2_feat_matrix.shape))
        return br_2_feat_matrix

    def write_br_2_features_matrix_final_df(br_2_feat_matrix):
        br_2_feat_matrix.to_csv(FilePath.FEAT_BR_MATRIX_FINAL.value)
        print('BR_2_Features Matrix Final.shape: {}'.format(br_2_feat_matrix.shape))

    
# DATASETS: TESTCASES, BUGREPORTS, FEATURES (FORMATTED) -----------

class Datasets:
    def read_testcases_df():
        testcases_df = pd.read_csv(FilePath.TESTCASES.value)
        testcases_df = testcases_df[(testcases_df.Feature_ID != 20) & (testcases_df.Feature_ID != 21)] # drop testcases from branch 65
        print('TestCases.shape: {}'.format(testcases_df.shape))
        return testcases_df


    def read_selected_bugreports_df():
        bugreports_df = pd.read_csv(FilePath.BUGREPORTS.value)
        bugreports_df = bugreports_df[(bugreports_df.Bug_Number != 1181835) & (bugreports_df.Bug_Number != 1315514)] # drop bug_reports lost during empirical study execution
        print('SelectedBugReports.shape: {}'.format(bugreports_df.shape))
        return bugreports_df
    
    def write_selected_bug_reports_df(bugreports):
        bugreports.to_csv(FilePath.BUGREPORTS.value)
        print('SelectedBugReports.shape: {}'.format(bugreports_df.shape))

    def read_features_df():
        features_df = pd.read_csv(FilePath.FEATURES.value)
        features_df = features_df[(features_df.Feature_Number != 20) & (features_df.Feature_Number != 21)] # drop features from branch 65
        print('Features.shape: {}'.format(features_df.shape))
        return features_df

    def write_features_df(features):
        features.to_csv(FilePath.FEATURES.value)
        print('Features.shape: {}'.format(features_df.shape))
    

# ORIGINAL DATASETS: FEATURES, BUGREPORTS (NOT FORMATTED) ---------

class OrigDatasets:
    def read_orig_features_df():
        orig_features_df = pd.read_csv(FilePath.ORIG_FEATURES.value)
        print('OrigFeatures.shape: {}'.format(orig_features_df.shape))
        return orig_features_df

    def read_orig_bugreports_df():
        orig_bugreports_df = pd.read_csv(FilePath.ORIG_BUGREPORTS.value)
        print('OrigBugReports.shape: {}'.format(orig_bugreports_df.shape))
        return orig_bugreports_df


# TASKRUNS: EXPERT AND VOLUNTEERS TASKRUNS ------------------

class TaskRuns:
    def read_expert_taskruns_df():
        taskruns = pd.read_csv(FilePath.EXPERT_TASKRUNS.value)
        taskruns.sort_values(by='bug_id', inplace=True)
        print('TaskRuns shape: {}'.format(taskruns.shape))
        return taskruns
    
    def read_expert_taskruns_2_df():
        taskruns = pd.read_csv(FilePath.EXPERT_TASKRUNS_2.value)
        taskruns.sort_values(by='bug_id', inplace=True)
        print('TaskRuns shape: {}'.format(taskruns.shape))
        return taskruns
    
    def read_volunteers_taskruns_1_df():
        taskruns_1 = pd.read_csv(FilePath.VOLUNTEERS_TASKRUNS_1.value)
        print('TaskRuns_1 shape: {}'.format(taskruns_1.shape))
        return taskruns_1
    
    def read_volunteers_taskruns_2_df():
        taskruns_2 = pd.read_csv(FilePath.VOLUNTEERS_TASKRUNS_2.value)      
        print('TaskRuns_2 shape: {}'.format(taskruns_2.shape))
        return taskruns_2
    
    def read_aux2_taskruns_df():
        taskruns = pd.read_csv(FilePath.AUX_2_TASKRUNS.value)
        taskruns.sort_values(by='bug_id', inplace=True)
        return taskruns
    