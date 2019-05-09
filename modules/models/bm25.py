import pandas as pd
import numpy as np

from gensim.summarization.bm25 import BM25

from enum import Enum

from modules.models.generic_model import GenericModel
from modules.models.model_hyperps import BM25_Model_Hyperp

from modules.utils import tokenizers as tok

from sklearn.preprocessing import MinMaxScaler

"""
params_dict = {
    'bm25__k' : 1.2,
    'bm25__b' : 0.75,
    'bm25__epsilon' : 0.25,
    'bm25__name' : 'BM25',
    'bm25__tokenizer' : Tokenizer(),
    'bm25__min_threshold' : 3
}
"""
class BM_25(GenericModel):
    # k = 1.2, b = 0.75 (default values)
    def __init__(self, **kwargs):
        self.k = None
        self.b = None
        self.epsilon = None
        self.tokenizer = None
        self._sim_matrix = None
               
        super().__init__()
        
        self.set_basic_params(**kwargs)
        self.set_tokenizer(**kwargs)
    
    def set_name(self, name):
        super().set_name(name)
    
    def set_model_gen_name(self, gen_name):
        super().set_model_gen_name(gen_name)
        
    def set_basic_params(self, **kwargs):
        self.set_name('BM25' if BM25_Model_Hyperp.NAME.value not in kwargs.keys() else kwargs[BM25_Model_Hyperp.NAME.value])
        self.set_model_gen_name('bm25')
        
        self.k = 1.2 if BM25_Model_Hyperp.K.value not in kwargs.keys() else kwargs[BM25_Model_Hyperp.K.value]
        self.b = 0.75 if BM25_Model_Hyperp.B.value not in kwargs.keys() else kwargs[BM25_Model_Hyperp.B.value]
        self.epsilon = 0.25 if BM25_Model_Hyperp.EPSILON.value not in kwargs.keys() else kwargs[BM25_Model_Hyperp.EPSILON.value]
        
        
    def set_tokenizer(self, **kwargs):
        self.tokenizer = tok.WordNetBased_LemmaTokenizer() if BM25_Model_Hyperp.TOKENIZER.value not in kwargs.keys() else kwargs[BM25_Model_Hyperp.TOKENIZER.value]
        
        #tokenizer_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__tokenizer__' in key}
        #self.tokenizer.set_params(**tokenizer_params)
        
    def recover_links(self, corpus, query, test_cases_names, bug_reports_names):
        bm25 = BM25([self.tokenizer.__call__(doc) for doc in corpus])
        average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
        query = [self.tokenizer.__call__(doc) for doc in query]
        
        self._sim_matrix = pd.DataFrame(index = test_cases_names, 
                                           columns = bug_reports_names,
                                           data=np.zeros(shape=(len(test_cases_names), len(bug_reports_names)),dtype='float64'))
        
        for bug_id, bug_desc in zip(bug_reports_names, query):
            scores = bm25.get_scores(bug_desc, average_idf=average_idf)
            for tc_id, sc in zip(test_cases_names, scores):
                self._sim_matrix.at[tc_id, bug_id] = sc
        
        self._sim_matrix =  super().normalize_sim_matrix(self._sim_matrix)
        self._sim_matrix = pd.DataFrame(self._sim_matrix, index=test_cases_names, columns=bug_reports_names)
        
        
    def model_setup(self):
        return {"Setup" : 
                  [
                      {"Name" : self.get_name()},
                      {"K" : self.k},
                      {"B" : self.b},
                      {"Epsilon" : self.epsilon},
                      {"Tokenizer Type" : type(self.tokenizer)}
                  ]
               }
    
    def get_name(self):
        return super().get_name()
    
    def get_model_gen_name(self):
        return super().get_model_gen_name()
    
    def get_sim_matrix(self):
        return super().get_sim_matrix()
    
    def get_tokenizer_type(self):
        return type(self.tokenizer)
        
    def save_sim_matrix(self):
        super().save_sim_matrix()
    
    
    