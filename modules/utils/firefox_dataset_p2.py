import pandas as pd

# functions for firefox_p2 (empirical study) ------

BASE_PATH = '/home/guilherme/anaconda3/envs/trace-link-recovery-study'

# TC_BR ORACLES --------------------------

def read_oracle_expert_volunteers_df():
    oracle_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/tc_br/oracle_expert_volunteers.csv')
    oracle_df.set_index('tc_name', inplace=True, drop=True)
    print('OracleExpertVolunteers.shape: {}'.format(oracle_df.shape))
    return oracle_df

def read_oracle_expert_df():
    oracle_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/tc_br/oracle_expert.csv')
    oracle_df.set_index('tc_name', inplace=True, drop=True)
    print('OracleExpert.shape: {}'.format(oracle_df.shape))
    return oracle_df

def read_oracle_volunteers_df():
    oracle_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/tc_br/oracle_volunteers.csv')
    oracle_df.set_index('tc_name', inplace=True, drop=True)
    print('OracleVolunteers.shape: {}'.format(oracle_df.shape))
    return oracle_df


# FEAT_BR ORACLES ------------------------

def read_feat_br_expert_volunteers_df():
    feat_br_trace_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/feat_br/br_2_feature_matrix_expert_volunteers.csv')
    feat_br_trace_df.set_index('feat_name', inplace=True)
    print('Expert and Volunteers Matrix.shape: {}'.format(feat_br_trace_df.shape))
    return feat_br_trace_df

def read_feat_br_expert_df():
    expert_matrix = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/feat_br/br_2_feature_matrix_expert.csv')
    expert_matrix.set_index('bug_number', inplace=True)
    print('Expert Matrix shape: {}'.format(expert_matrix.shape))
    return expert_matrix

def read_feat_br_volunteers_df():
    volunteers_matrix = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/feat_br/br_2_feature_matrix_volunteers.csv')
    volunteers_matrix.set_index('bug_number', inplace=True)
    print('Volunteers Matrix shape: {}'.format(volunteers_matrix.shape))
    return volunteers_matrix


# AUXILIARY DATAFRAME WITH TRACELINKS BETWEEN BUG REPORTS AND FEATURES

def read_aux_df():
    aux_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/oracle/output/firefox_v2/aux_data/aux_df.csv')
    print('AuxDf.shape: {}'.format(aux_df.shape))
    return aux_df


# DATASETS: TESTCASES, BUGREPORTS, FEATURES (FORMATTED) -----------

def read_testcases_df():
    testcases_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/TC/testcases_final.csv')
    print('TestCases.shape: {}'.format(testcases_df.shape))
    return testcases_df

def read_bugreports_df():
    bugreports_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/selected_bugreports_final.csv')
    print('BugReports.shape: {}'.format(bugreports_df.shape))
    return bugreports_df

def read_features_df():
    features_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/features_final.csv')
    print('Features.shape: {}'.format(features_df.shape))
    return features_df


# ORIGINAL DATASETS: FEATURES, BUGREPORTS (NOT FORMATTED) ---------

def read_orig_features_df():
    orig_features_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/features.csv')
    print('OrigFeatures.shape: {}'.format(orig_features_df.shape))
    return orig_features_df

def read_orig_bugreports_df():
    orig_bugreports_df = pd.read_csv(BASE_PATH + '/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')
    print('OrigBugReports.shape: {}'.format(orig_bugreports_df.shape))
    return orig_bugreports_df


