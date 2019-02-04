import pandas as pd

def read_trace_df():
    return pd.read_csv('../../data/mozilla_firefox_v2/firefoxDataset/oracle/output/trace_matrix_final.csv')

def read_testcases_df():
    return pd.read_csv('../../data/mozilla_firefox_v2/firefoxDataset/docs_english/TC/testcases_final.csv')

def read_bugreports_df():
    return pd.read_csv('../../data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')