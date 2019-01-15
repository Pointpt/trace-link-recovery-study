import pandas as pd

def read_trace_df():
    return pd.read_csv('../../data/mozilla_firefox/firefoxDataset/oracle/output/trace_matrix.csv')

def read_artfs_desc_df():
    return pd.read_csv('../../data/mozilla_firefox/firefoxDataset/oracle/output/artifacts_descriptions.csv', sep="|")