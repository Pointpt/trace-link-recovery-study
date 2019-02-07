import pandas as pd
import numpy as np
import spacy

from sklearn.pipeline import Pipeline

from enum import Enum

from models.generic_model import GenericModel
from models.model_hyperps import WordVec_Model_Hyperp


class SimilarityMeasure(Enum):
    COSINE = 'cosine'
    
    

"""
params_dict = {
    'wordvec__sim_measure_min_threshold' : ('cosine',.9),
    'wordvec__name' : 'WordVec',
    'wordvec__top' : 3
    'wordvec_tokenizer' : WordNetBased_LemmaTokenizer()
}
"""
class WordVec_BasedModel(GenericModel):
    def __init__(self, **kwargs):
        self._nlp_model = None
        self.tokenizer = None
        
        super().__init__()
        
        self.set_basic_params(**kwargs)
        self.set_nlp_model()
    
    def set_name(self, name):
        super().set_name(name)
    
    def set_model_gen_name(self, gen_name):
        super().set_model_gen_name(gen_name)
    
    def set_top(self, top):
        super().set_top(top)
    
    def set_sim_measure_min_threshold(self, threshold):
        super().set_sim_measure_min_threshold(threshold)
    
    def set_basic_params(self, **kwargs):
        self.set_name('WordVec' if WordVec_Model_Hyperp.NAME.value not in kwargs.keys() else kwargs[WordVec_Model_Hyperp.NAME.value])
        self.set_sim_measure_min_threshold((SimilarityMeasure.COSINE.value,.80) if WordVec_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value not in kwargs.keys() else kwargs[WordVec_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value])
        self.set_top(3 if WordVec_Model_Hyperp.TOP.value not in kwargs.keys() else kwargs[WordVec_Model_Hyperp.TOP.value])
        self.set_model_gen_name('wordvector')
        
        self.tokenizer = tok.WordNetBased_LemmaTokenizer() if WordVec_Model_Hyperp.TOKENIZER.value not in kwargs.keys() else kwargs[WordVec_Model_Hyperp.TOKENIZER.value]
        
    
    def set_nlp_model(self):
        """
            WordVec based on GloVe 1.1M keys x 300 dim
            300-dimensional word vectors trained on Common Crawl with GloVe.
        """
        self._nlp_model = spacy.load('en_vectors_web_lg')
    
    def __getstate__(self):
        """to pickle object serialization/deserialization"""
        d = dict(self.__dict__)
        del d['_nlp_model']
        return d
    
    def __setstate__(self, d):
        """to pickle object serialization/deserialization"""
        self.__dict__.update(d)
    
    def recover_links(self, corpus, query, use_cases_names, bug_reports_names):
        return self._recover_links_cosine(corpus, query, use_cases_names, bug_reports_names)
    
    def _recover_links_cosine(self, corpus, query, use_cases_names, bug_reports_names):
        list_corpus_tokens = [self.tokenizer.__call__(doc) for doc in corpus]
        list_query_tokens = [self.tokenizer.__call__(doc) for doc in query]
        
        corpus = [' '.join(tok_list) for tok_list in list_corpus_tokens]
        query = [' '.join(tok_list) for tok_list in list_query_tokens]
        
        self._sim_matrix = pd.DataFrame(index = use_cases_names, 
                                           columns = bug_reports_names,
                                           data=np.zeros(shape=(len(use_cases_names), len(bug_reports_names)),dtype='float64'))
        
        for bug_id, bug_desc in zip(bug_reports_names, query):
            for uc_id, uc_desc in zip(use_cases_names, corpus):
                doc1 = self._nlp_model(bug_desc)
                doc2 = self._nlp_model(uc_desc)
                self._sim_matrix.at[uc_id, bug_id] = doc1.similarity(doc2)  # cosine similarity is default
        
        self._sim_matrix = pd.DataFrame(self._sim_matrix, index=use_cases_names, columns=bug_reports_names)
        super()._fillUp_traceLinksDf(use_cases_names, bug_reports_names, self._sim_matrix)        
    
    def model_setup(self):
        return {"Setup" : 
                  [
                      {"Name" : self.get_name()},
                      {"Similarity Measure and Minimum Threshold" : self.get_sim_measure_min_threshold()},
                      {"Top Value" : self.get_top_value()},
                      {"Tokenizer" : self.tokenizer}
                  ]
               }
    
    def get_name(self):
        return super().get_name()
    
    def get_model_gen_name(self):
        return super().get_model_gen_name()
    
    def get_top_value(self):
        return super().get_top_value()
    
    def get_sim_measure_min_threshold(self):
        return super().get_sim_measure_min_threshold()
    
    def get_sim_matrix(self):
        return super().get_sim_matrix()
    
    def get_tokenizer_type(self):
        return type(self.tokenizer)
    
    def get_trace_links_df(self):
        return super().get_trace_links_df()
    
    def save_sim_matrix(self):
        super().save_sim_matrix()
    
    def get_model_dump_path(self):
        return super().get_model_dump_path()


