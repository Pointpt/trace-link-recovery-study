import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, pairwise_distances, pairwise
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

from scipy.stats import entropy

from enum import Enum

from modules.models.generic_model import GenericModel
from modules.models.model_hyperps import LDA_Model_Hyperp


class SimilarityMeasureName(Enum):
    JSD = 'jsd'

class SimilarityMeasure:
    def __init__(self):
        self.name = SimilarityMeasureName.JSD
    
    # static method
    def jsd(p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        # normalize
        #p /= p.sum()
        #q /= q.sum()
        m = (p + q) / 2
        return (entropy(p, m) + entropy(q, m)) / 2


"""
params_dict = {
    'lda__name' : 'LDA',
    'lda__top_value' : 3,
    'lda__sim_measure_min_threshold' : ('cosine',.9),
    'lda__vectorizer' : TfidfVectorizer(),
    'lda__vectorizer__stop_words' : 'english',
    'lda__vectorizer__tokenizer' : Tokenizer(),
    'lda__vectorizer__use_idf' : True,          # optional if type(Vectorizer) == TfidfVectorizer
    'lda__vectorizer__smooth_idf' : True,       # optional if type(Vectorizer) == TfidfVectorizer
    'lda__vectorizer__ngram_range' : (1,2),
    'lda__lda_model' : TruncatedSVD(),
    'lda__lda_model__n_components' : 5
}
"""
class LDA(GenericModel):
    def __init__(self, **kwargs):
        self._corpus_matrix = None
        self._query_vector = None
        
        self.vectorizer = None
        self.lda_model = LatentDirichletAllocation(n_jobs=-1)
        
        super().__init__()
        
        self.set_basic_params(**kwargs)
        
        self.set_vectorizer(**kwargs)
        self.set_lda_model(**kwargs)
    
    def set_name(self, name):
        super().set_name(name)
    
    def set_model_gen_name(self, gen_name):
        super().set_model_gen_name(gen_name)
    
    def set_top(self, top):
        super().set_top(top)
    
    def set_sim_measure_min_threshold(self, threshold):
        super().set_sim_measure_min_threshold(threshold)
    
    def set_basic_params(self, **kwargs):
        self.set_name('LDA' if LDA_Model_Hyperp.NAME.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.NAME.value])
        self.set_sim_measure_min_threshold((SimilarityMeasureName.JSD.value, .3) if LDA_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value]       )
        self.set_top(3 if LDA_Model_Hyperp.TOP.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.TOP.value])
        self.set_model_gen_name('lda')
        
    def set_vectorizer(self, **kwargs):
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                             use_idf=True, 
                                             smooth_idf=True) if LDA_Model_Hyperp.VECTORIZER.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.VECTORIZER.value]
        vec_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__vectorizer__' in key}
        self.vectorizer.set_params(**vec_params)
    
    def set_lda_model(self, **kwargs):      
        lda_model_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__lda_model__' in key}
        self.lda_model.set_params(**lda_model_params)
    
    def recover_links(self, corpus, query, use_cases_names, bug_reports_names):
        self._corpus_matrix = self.vectorizer.fit_transform(corpus)
        self._query_vector = self.vectorizer.transform(query)
        
        out_1 = self.lda_model.fit_transform(self._corpus_matrix)
        out_2 = self.lda_model.transform(self._query_vector)
        
        metric = self.sim_measure_min_threshold[0]
        if metric == 'cosine':
            self._sim_matrix = pairwise.cosine_similarity(X=out_1, Y=out_2)
        elif metric == 'jsd':
            self._sim_matrix = pairwise_distances(X=out_1, Y=out_2, metric=SimilarityMeasure.jsd)
        
        self._sim_matrix = pd.DataFrame(data=self._sim_matrix, index=use_cases_names, columns=bug_reports_names)
        super()._fillUp_traceLinksDf(use_cases_names, bug_reports_names, self._sim_matrix)
                   
    def model_setup(self):
        return {"Setup" : 
                  [
                      {"Name" : self.get_name()},
                      {"Similarity Measure and Minimum Threshold" : self.get_sim_measure_min_threshold()},
                      {"Top Value" : self.get_top_value()},
                      {"LDA Model" : self.lda_model.get_params()},
                      {"Vectorizer" : self.vectorizer.get_params()},
                      {"Vectorizer Type" : type(self.vectorizer)}
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
    
    def save_trace_matrix(self):
        super().save_trace_matrix()
    
    def get_model_dump_path(self):
        return super().get_model_dump_path()
    
    def get_query_vector(self):
        return self._query_vector
    
    def get_corpus_matrix(self):
        return self._corpus_matrix
    
    def get_vectorizer_type(self):
        return type(self.vectorizer)
    

