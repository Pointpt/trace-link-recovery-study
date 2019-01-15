import pandas as pd
import numpy as np

class OracleLoader:
    def __init__(self, rows_names, columns_names):
        self.oracle = None
        self._columns_names = columns_names
        self._rows_names = rows_names
    
    def load(self, trace_df):
        self.oracle = pd.DataFrame(columns=list(self._columns_names), 
                                   data=np.zeros(shape=(len(self._rows_names), len(self._columns_names)), 
                                                 dtype='int64'))
        self.oracle.insert(0, 'artf_name', list(self._rows_names))
        
        for index, row in trace_df.iterrows():
            idx = self.oracle[self.oracle.artf_name == row['trg_artf']].index
            self.oracle.at[idx, row['src_artf']] = row['link']

        self.oracle.set_index('artf_name', inplace=True)