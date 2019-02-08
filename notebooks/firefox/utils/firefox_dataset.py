import pandas as pd

def read_trace_df():
    oracle_df = pd.read_csv('../../data/mozilla_firefox_v2/firefoxDataset/oracle/output/trace_matrix_final.csv')
    print('Oracle.shape: {}'.format(oracle_df.shape))
    return oracle_df
    
def read_testcases_df():
    testcases_df = pd.read_csv('../../data/mozilla_firefox_v2/firefoxDataset/docs_english/TC/testcases_final.csv')
    print('TestCases.shape: {}'.format(testcases_df.shape))
    return testcases_df

def read_bugreports_df():
    bugreports_df = pd.read_csv('../../data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')
    print('BugReports.shape: {}'.format(bugreports_df.shape))
    return bugreports_df