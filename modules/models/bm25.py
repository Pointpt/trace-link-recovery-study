import pandas as pd
import numpy as np

import math

from gensim.summarization.bm25 import BM25
from gensim.summarization.bm25 import get_bm25_weights

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
    
    def get_bm25_weights(self, corpus):
        return get_bm25_weights(corpus)
    
    def recover_links(self, corpus, query, test_cases_names, bug_reports_names):
        self.bm25 = BM25([self.tokenizer.__call__(doc) for doc in corpus])
        #self.average_idf = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())
        query = [self.tokenizer.__call__(doc) for doc in query]
        
        self._sim_matrix_origin = pd.DataFrame(index = test_cases_names, 
                                           columns = bug_reports_names,
                                           data=np.zeros(shape=(len(test_cases_names), len(bug_reports_names)),dtype='float64'))
        
        for bug_id, bug_desc in zip(bug_reports_names, query):
            scores = self.bm25.get_scores(bug_desc)
            for tc_id, sc in zip(test_cases_names, scores):
                self._sim_matrix_origin.at[tc_id, bug_id] = sc
        
        self._sim_matrix = super().normalize_sim_matrix(self._sim_matrix_origin)
        self._sim_matrix = pd.DataFrame(self._sim_matrix, index=test_cases_names, columns=bug_reports_names)
        
        self._recover_docs_feats(corpus, query, test_cases_names, bug_reports_names)
    
    
    def _recover_docs_feats(self, corpus, query, test_cases_names, bug_reports_names):
        self.mrw_tcs = self._recover_mrw_list(test_cases_names, corpus)
        self.mrw_brs = self._recover_mrw_list(bug_reports_names, query)
        
        self.dl_tcs = self._recover_dl_list(test_cases_names, corpus)
        self.dl_brs = self._recover_dl_list(bug_reports_names, query)
        
        index = list(test_cases_names) + list(bug_reports_names)
        self.docs_feats_df = pd.DataFrame(index=index,
                                         columns=['mrw','dl'])
        
        for tc_name, mrw in self.mrw_tcs:
            self.docs_feats_df.at[tc_name, 'mrw'] = mrw

        for tc_name, dl in self.dl_tcs:
            self.docs_feats_df.at[tc_name, 'dl'] = dl
            
        for br_name, mrw in  self.mrw_brs:
            self.docs_feats_df.at[br_name, 'mrw'] = mrw
        
        for br_name, dl in self.dl_brs:
            self.docs_feats_df.at[br_name, 'dl'] = dl
        
    def _recover_dl_list(self, artf_names, artf_descs):
        dl_list = []
        for idx, (artf_name, artf_desc) in enumerate(zip(artf_names, artf_descs)):
            dl_list.append( (artf_name, sum(self.bm25.doc_freqs[idx].values())) )
        return dl_list
    
    def _recover_mrw_list(self, artfs_names, artfs_descs):
        N_REL_WORDS = 6
        mrw_list = [] # list of tuples (artf_name, mrw_list={})
        
        for idx,(artf_name,artf_desc) in enumerate(zip(artfs_names, artfs_descs)):
            t_w_list = []
            for token in self.bm25.doc_freqs[idx].keys():
                Tf = self.bm25.doc_freqs[idx][token]
                DL = len(self.bm25.doc_freqs[idx])
                AVGDL = self.bm25.avgdl
                N = self.bm25.corpus_size
                NDt = np.sum([1 if token in d.keys() else 0 for d in self.bm25.doc_freqs])
                K1 = self.k                
                b = self.b
                eps = self.epsilon
                print(math.log(N/NDt))
                t_weight = (Tf * (K1 + 1) / K1*((1-b) + b*DL/AVGDL) ) / (eps + math.log(N/NDt))
                
                t_w_list.append((token, t_weight))
            
            df_tokens = pd.DataFrame(columns=['token','token_weight'])
            df_tokens['token'] = [tok for tok,tok_w in t_w_list]
            df_tokens['token_weight'] = [tok_weight for tok,tok_weight in t_w_list]
            df_tokens.sort_values(by='token_weight', ascending=False, inplace=True)
            
            mrw = list(df_tokens.iloc[0:N_REL_WORDS, 0].values)
            mrw_list.append((artf_name, mrw))
            
        return mrw_list
        
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
    
    
    