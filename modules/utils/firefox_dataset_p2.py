# functions for firefox_p2 (empirical study) ------

import pandas as pd
from enum import Enum

BASE_PATH = '/home/guilherme/anaconda3/envs/trace-link-recovery-study'

FEAT_X_BR_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/feat_br/'
TC_X_BR_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/tc_br/'

BUGREPORTS_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/'
TESTCASES_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/TC/'
FEATURES_M_PATH = BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/'


class FilePath(Enum):
    #ORACLE_EXPERT_VOLUNTEERS = TC_X_BR_M_PATH + 'oracle_expert_volunteers.csv'
    ORACLE_EXPERT_VOLUNTEERS_UNION = TC_X_BR_M_PATH + 'oracle_expert_volunteers_union.csv'
    ORACLE_EXPERT_VOLUNTEERS_INTERSEC = TC_X_BR_M_PATH + 'oracle_expert_volunteers_intersec.csv'
    ORACLE_EXPERT = TC_X_BR_M_PATH + 'oracle_expert.csv'
    ORACLE_VOLUNTEERS = TC_X_BR_M_PATH + 'oracle_volunteers.csv'
    
    #FEAT_BR_EXPERT_VOLUNTEERS = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert_volunteers.csv'
    FEAT_BR_EXPERT_VOLUNTEERS_UNION = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert_volunteers_union.csv'
    FEAT_BR_EXPERT_VOLUNTEERS_INTERSEC = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert_volunteers_intersec.csv'
    FEAT_BR_EXPERT = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_expert.csv'
    FEAT_BR_VOLUNTEERS = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_volunteers.csv'
    FEAT_BR_VOLUNTEERS_2 = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_volunteers_2.csv'
    FEAT_BR_MATRIX_FINAL = FEAT_X_BR_M_PATH + 'br_2_feature_matrix_final.csv'
    
    TESTCASES = TESTCASES_M_PATH + 'testcases_final.csv'
    BUGREPORTS = BUGREPORTS_M_PATH + 'selected_bugreports_final.csv'
    FEATURES = FEATURES_M_PATH + 'features_final.csv'

    ORIG_FEATURES = FEATURES_M_PATH + 'features.csv'
    ORIG_BUGREPORTS = BUGREPORTS_M_PATH + 'bugreports_final.csv'

    
# TC_BR ORACLES --------------------------

class Tc_BR_Oracles:   
    def read_oracle_expert_volunteers_union_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_EXPERT_VOLUNTEERS_UNION.value)
        oracle_df.set_index('tc_name', inplace=True, drop=True)
        print('OracleExpertVolunteers_UNION.shape: {}'.format(oracle_df.shape))
        return oracle_df
    
    def read_oracle_expert_volunteers_intersec_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_EXPERT_VOLUNTEERS_INTERSEC.value)
        oracle_df.set_index('tc_name', inplace=True, drop=True)
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
        oracle_df.set_index('tc_name', inplace=True, drop=True)
        print('OracleExpert.shape: {}'.format(oracle_df.shape))
        return oracle_df

    def write_oracle_expert_df(oracle_expert_df):
        oracle_expert_df.to_csv(FilePath.ORACLE_EXPERT.value)
        print('OracleExpert.shape: {}'.format(oracle_expert_df.shape))
    
    
    def read_oracle_volunteers_df():
        oracle_df = pd.read_csv(FilePath.ORACLE_VOLUNTEERS.value)
        oracle_df.set_index('tc_name', inplace=True, drop=True)
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
        print('Expert and Volunteers Matrix UNION.shape: {}'.format(feat_br_expert_volunteers_matrix.shape))
        
    def write_feat_br_expert_volunteers_intersec_df(feat_br_expert_volunteers_matrix):
        feat_br_expert_volunteers_matrix.to_csv(FilePath.FEAT_BR_EXPERT_VOLUNTEERS_INTERSEC.value, index=True)
        print('Expert and Volunteers Matrix INTERSEC.shape: {}'.format(feat_br_expert_volunteers_matrix.shape))
        

    def read_feat_br_expert_df():
        expert_matrix = pd.read_csv(FilePath.FEAT_BR_EXPERT.value)
        expert_matrix.set_index('bug_number', inplace=True)
        print('Expert Matrix shape: {}'.format(expert_matrix.shape))
        return expert_matrix

    def read_feat_br_volunteers_df():
        volunteers_matrix_1 = pd.read_csv(FilePath.FEAT_BR_VOLUNTEERS.value)
        volunteers_matrix_1.set_index('bug_number', inplace=True)
        #print('Volunteers Matrix 1 shape: {}'.format(volunteers_matrix_1.shape))
        
        volunteers_matrix_2 = pd.read_csv(FilePath.FEAT_BR_VOLUNTEERS_2.value)
        volunteers_matrix_2.set_index('bug_number', inplace=True)
        #print('Volunteers Matrix 2 shape: {}'.format(volunteers_matrix_2.shape))
               
        volunteers_matrix =  pd.concat([volunteers_matrix_1, volunteers_matrix_2])
        print('Volunteers Matrix shape: {}'.format(volunteers_matrix.shape))
        return volunteers_matrix

    
    ## selected_bug_reports_2 with the related features after the empirical study
    def read_br_2_features_matrix_final_df():
        br_2_feat_matrix = pd.read_csv(FilePath.FEAT_BR_MATRIX_FINAL.value, dtype=object)
        br_2_feat_matrix.set_index('Bug_Number', inplace=True)
        br_2_feat_matrix.fillna("", inplace=True)
        br_2_feat_matrix.drop(columns='Unnamed: 0', inplace=True)
        print('BR_2_Features Matrix Final.shape: {}'.format(br_2_feat_matrix.shape))
        return br_2_feat_matrix

    def write_br_2_features_matrix_final_df(br_2_feat_matrix):
        br_2_feat_matrix.to_csv(FilePath.FEAT_BR_MATRIX_FINAL.value, dtype=object)
        print('BR_2_Features Matrix Final.shape: {}'.format(br_2_feat_matrix.shape))

    
# DATASETS: TESTCASES, BUGREPORTS, FEATURES (FORMATTED) -----------

class Datasets:
    def read_testcases_df():
        testcases_df = pd.read_csv(FilePath.TESTCASES.value)
        print('TestCases.shape: {}'.format(testcases_df.shape))
        return testcases_df


    def read_selected_bugreports_df():
        bugreports_df = pd.read_csv(FilePath.BUGREPORTS.value)
        print('SelectedBugReports.shape: {}'.format(bugreports_df.shape))
        return bugreports_df

    def write_selected_bug_reports_df(bugreports):
        bugreports.to_csv(FilePath.BUGREPORTS.value)
        print('SelectedBugReports.shape: {}'.format(bugreports_df.shape))

    def read_features_df():
        features_df = pd.read_csv(FilePath.FEATURES.value)
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


