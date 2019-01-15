import pandas as pd

def read_trace_df():
    return pd.read_csv('../../data/jEdit/jEditDataset/oracle/output/trace_matrix.csv')

def read_artfs_desc_df():
    return pd.read_csv('../../data/jEdit/jEditDataset/oracle/output/artifacts_descriptions.csv', sep="|")