import pandas as pd
import numpy as np
import spacy

from sklearn.pipeline import Pipeline

from modules.models.generic_model import GenericModel
from modules.models.model_hyperps import WordVec_Model_Hyperp

from modules.utils import tokenizers as tok
from modules.utils import similarity_measures as sm


"""
params_dict = {
    'wordvec__name' : 'WordVec',
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
    
        
    def set_basic_params(self, **kwargs):
        self.set_name('WordVec' if WordVec_Model_Hyperp.NAME.value not in kwargs.keys() else kwargs[WordVec_Model_Hyperp.NAME.value])
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
    
    def recover_links(self, corpus, query, test_cases_names, bug_reports_names):
        return self._recover_links_cosine(corpus, query, test_cases_names, bug_reports_names)
    
    def _recover_links_cosine(self, corpus, query, test_cases_names, bug_reports_names):
        list_corpus_tokens = [self.tokenizer.__call__(doc) for doc in corpus]
        list_query_tokens = [self.tokenizer.__call__(doc) for doc in query]
        
        corpus = [' '.join(tok_list) for tok_list in list_corpus_tokens]
        query = [' '.join(tok_list) for tok_list in list_query_tokens]
        
        self._sim_matrix = pd.DataFrame(index = test_cases_names, 
                                           columns = bug_reports_names,
                                           data=np.zeros(shape=(len(test_cases_names), len(bug_reports_names)),dtype='float64'))
        
        for bug_id, bug_desc in zip(bug_reports_names, query):
            for uc_id, uc_desc in zip(test_cases_names, corpus):
                doc1 = self._nlp_model(bug_desc)
                doc2 = self._nlp_model(uc_desc)
                self._sim_matrix.at[uc_id, bug_id] = doc1.similarity(doc2)  # cosine similarity is default
        
        self._sim_matrix = pd.DataFrame(self._sim_matrix, index=test_cases_names, columns=bug_reports_names)      
    
    
    def model_setup(self):
        return {"Setup" : 
                  [
                      {"Name" : self.get_name()},
                      {"Tokenizer" : self.tokenizer}
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
    
    


