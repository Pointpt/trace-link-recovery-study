import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer, normalize

from modules.utils.similarity_measures import SimilarityMeasure

from modules.models.generic_model import GenericModel
from modules.models.model_hyperps import LSI_Model_Hyperp


    
"""
params_dict = {
    'lsi__similarity_measure' : SimilarityMeasure.COSINE,
    'lsi__name' : 'LSI',
    'lsi__vectorizer' : TfidfVectorizer(),
    'lsi__vectorizer__stop_words' : 'english',
    'lsi__vectorizer__tokenizer' : Tokenizer(),
    'lsi__vectorizer__use_idf' : True,          # optional if type(Vectorizer) == TfidfVectorizer
    'lsi__vectorizer__smooth_idf' : True,       # optional if type(Vectorizer) == TfidfVectorizer
    'lsi__vectorizer__ngram_range' : (1,2),
    'lsi__svd_model' : TruncatedSVD(),
    'lsi__svd_model__n_components' : 5
}
"""
class LSI(GenericModel):
    def __init__(self, **kwargs):
        self._svd_matrix = None
        self._query_vector = None
        
        self.vectorizer = None
        self.svd_model = None
        
        super().__init__()
        
        self.similarity_measure = None
        
        self.set_basic_params(**kwargs)
        self.set_vectorizer(**kwargs)
        self.set_svd_model(**kwargs)
    
    def set_name(self, name):
        super().set_name(name)
    
    def set_model_gen_name(self, gen_name):
        super().set_model_gen_name(gen_name)
       
    def set_basic_params(self, **kwargs):
        self.set_name('LSI' if LSI_Model_Hyperp.NAME.value not in kwargs.keys() else kwargs[LSI_Model_Hyperp.NAME.value])
        self.set_similarity_measure(SimilarityMeasure.COSINE)
        self.set_model_gen_name('lsi')
    
    def set_similarity_measure(self, sim_measure):
        self.similarity_measure = sim_measure
    
    def set_vectorizer(self, **kwargs):
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                             use_idf=True, 
                                             smooth_idf=True) if LSI_Model_Hyperp.VECTORIZER.value not in kwargs.keys() else kwargs[LSI_Model_Hyperp.VECTORIZER.value]
        
        vec_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__vectorizer__' in key}
        self.vectorizer.set_params(**vec_params)
    
    def set_svd_model(self, **kwargs):
        self.svd_model = TruncatedSVD(n_components = 100, 
                                         algorithm = 'randomized',
                                         n_iter = 10, 
                                         random_state = 42) if LSI_Model_Hyperp.SVD_MODEL.value not in kwargs.keys() else kwargs[LSI_Model_Hyperp.SVD_MODEL.value]
        
        svd_model_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__svd_model__' in key}
        self.svd_model.set_params(**svd_model_params)
        
    
    def recover_links(self, corpus, query, test_cases_names, bug_reports_names):
        
        if self.similarity_measure == SimilarityMeasure.COSINE:
            return self._recover_links_cosine(corpus, query, test_cases_names, bug_reports_names)
        
        elif self.similarity_measure == SimilarityMeasure.JACCARD_INDEX:
            return self._recover_links_jaccard(corpus, query, test_cases_names, bug_reports_names)
        
        elif self.similarity_measure == SimilarityMeasure.EDIT_DISTANCE:
            return self._recover_links_edit(corpus, query, test_cases_names, bug_reports_names)
    
    def _recover_links_cosine(self, corpus, query, test_cases_names, bug_reports_names):
        svd_transformer = Pipeline([('vec', self.vectorizer), 
                            ('svd', self.svd_model)])

        self._svd_matrix = svd_transformer.fit_transform(corpus)
        self._query_vector = svd_transformer.transform(query)
        self._sim_matrix = pairwise.cosine_similarity(X=self._svd_matrix, Y=self._query_vector)
        self._sim_matrix = pd.DataFrame(data=self._sim_matrix, index=test_cases_names, columns=bug_reports_names)

    
    def _recover_links_jaccard(self, corpus, query, test_cases_names, bug_reports_names):
        tokenizer = self.vectorizer.tokenizer
                
        corpus_tokens = [tokenizer.__call__(doc) for doc in corpus]        
        query_tokens = [tokenizer.__call__(doc) for doc in query]
        
        self._sim_matrix = pd.DataFrame(index = test_cases_names, 
                                       columns = bug_reports_names,
                                       data = np.zeros(shape=(len(test_cases_names), len(bug_reports_names)), dtype='int8'))
        
        for br_id, doc_query_tset in zip(bug_reports_names, query_tokens):
            for tc_id, doc_corpus_tset in zip(test_cases_names, corpus_tokens):
                self._sim_matrix.at[tc_id, br_id] = nltk.jaccard_distance(set(doc_corpus_tset), set(doc_query_tset))
                
    
    def _recover_links_edit(self, corpus, query, test_cases_names, bug_reports_names):
        self._sim_matrix = pd.DataFrame(index = test_cases_names, 
                                       columns = bug_reports_names,
                                       data = np.zeros(shape=(len(test_cases_names), len(bug_reports_names)), dtype='int8'))
                
        for br_id, doc_query in zip(bug_reports_names, query):
            for tc_id, doc_corpus in zip(test_cases_names, corpus):
                self._sim_matrix.at[tc_id, br_id] = nltk.edit_distance(doc_corpus, doc_query)
        
        normalizer = Normalizer(copy=False).fit(self._sim_matrix.values)
        self._sim_matrix = pd.DataFrame(data=normalizer.transform(self._sim_matrix.values), index=test_cases_names, columns=bug_reports_names)
        
    
    def model_setup(self):
        return {"Setup" : 
                  [
                      {"Name" : self.get_name()},
                      {"Similarity Measure" : self.get_similarity_measure()},
                      {"SVD Model" : self.svd_model.get_params()},
                      {"Vectorizer" : self.vectorizer.get_params()},
                      {"Vectorizer Type" : type(self.vectorizer)}
                  ]
               }
        
    def get_query_vector(self):
        return self._query_vector
    
    def get_svd_matrix(self):
        return self._svd_matrix
    
    def get_vectorizer_type(self):
        return type(self.vectorizer)
    
    def get_tokenizer_type(self):
        return type(self.vectorizer.tokenizer)    
        
    def get_name(self):
        return super().get_name()

    def get_model_gen_name(self):
        return super().get_model_gen_name()
    
    def get_similarity_measure(self):
        return self.similarity_measure
    
    def get_sim_matrix(self):
        return super().get_sim_matrix()
        
    def save_sim_matrix(self):
        super().save_sim_matrix()
    
    