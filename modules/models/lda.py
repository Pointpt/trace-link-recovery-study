import time
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, pairwise_distances, pairwise
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

from scipy.stats import entropy

from modules.models.generic_model import GenericModel
from modules.models.model_hyperps import LDA_Model_Hyperp

from modules.utils import similarity_measures as sm
from modules.utils.tokenizers import PorterStemmerBased_Tokenizer

class SimilarityMeasure:
    def __init__(self):
        self.name = sm.SimilarityMeasure.JSD
    
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
    'lda__similarity_measure' : SimilarityMeasure.COSINE,
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
        
        self.similarity_measure = None
        self.set_basic_params(**kwargs)
        
        self.set_vectorizer(**kwargs)
        self.set_lda_model(**kwargs)
    
    def set_name(self, name):
        super().set_name(name)
    
    def set_model_gen_name(self, gen_name):
        super().set_model_gen_name(gen_name)
    
    def set_basic_params(self, **kwargs):
        self.set_name('LDA' if LDA_Model_Hyperp.NAME.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.NAME.value])
        self.set_model_gen_name('lda')
        self.set_similarity_measure(sm.SimilarityMeasure.COSINE if LDA_Model_Hyperp.SIMILARITY_MEASURE.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.SIMILARITY_MEASURE.value])
    
    def set_similarity_measure(self, sim_measure):
        self.similarity_measure = sim_measure
    
    def set_vectorizer(self, **kwargs):
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                             use_idf=True, 
                                             smooth_idf=True) if LDA_Model_Hyperp.VECTORIZER.value not in kwargs.keys() else kwargs[LDA_Model_Hyperp.VECTORIZER.value]
        vec_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__vectorizer__' in key}
        self.vectorizer.set_params(**vec_params)
    
    def set_lda_model(self, **kwargs):      
        lda_model_params = {key.split('__')[2]:kwargs[key] for key,val in kwargs.items() if '__lda_model__' in key}
        self.lda_model.set_params(**lda_model_params)
    
    def recover_links(self, corpus, query, test_cases_names, bug_reports_names):
        starttime = time.time()
        
        self._corpus_matrix = self.vectorizer.fit_transform(corpus)
        self._query_vector = self.vectorizer.transform(query)
        
        self.out_1 = self.lda_model.fit_transform(self._corpus_matrix)
        self.out_2 = self.lda_model.transform(self._query_vector)
        
        metric = self.similarity_measure
        if metric == sm.SimilarityMeasure.COSINE:
            self._sim_matrix = pairwise.cosine_similarity(X=self.out_1, Y=self.out_2)
        elif metric == sm.SimilarityMeasure.JSD:
            self._sim_matrix = pairwise_distances(X=self.out_1, Y=self.out_2, metric=SimilarityMeasure.jsd)
        elif metric == sm.SimilarityMeasure.EUCLIDIAN_DISTANCE:
            self._sim_matrix = pairwise_distances(X=self.out_1, Y=self.out_2, metric='euclidean')
        
        #self._sim_matrix =  super().normalize_sim_matrix(self._sim_matrix)
        self._sim_matrix = pd.DataFrame(data=self._sim_matrix, index=test_cases_names, columns=bug_reports_names)

        self._record_docs_feats(corpus, query, test_cases_names, bug_reports_names)
        
        endtime = time.time()
        
        print(f' ..Total processing time: {round(endtime-starttime,2)} seconds')
    
    
    def _record_docs_feats(self, corpus, query, test_cases_names, bug_reports_names):
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
        tokenizer = PorterStemmerBased_Tokenizer()
        dl_list = []
        for artf_name, artf_desc in zip(artf_names, artf_descs):
            dl_list.append((artf_name, len(tokenizer.__call__(artf_desc))))
        return dl_list
    
    def _recover_mrw_list(self, artf_names, artf_descs):
        N_REL_WORDS = 6
        mrw_list = [] # list of tuples (artf_name, mrw_list={})
        
        for artf_name, artf_desc in zip(artf_names, artf_descs):
            X = self.vectorizer.transform([artf_desc])
            df1 = pd.DataFrame(X.T.toarray())
            df1['token'] = self.vectorizer.get_feature_names()
            df1.sort_values(by=0, ascending=False, inplace=True)
            mrw = list(df1.iloc[0:N_REL_WORDS,1].values)
            mrw_list.append((artf_name, mrw))
            
        return mrw_list
    
        
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
    
    def get_similarity_measure(self):
        return self.similarity_measure
    
    def get_sim_matrix(self):
        return super().get_sim_matrix()
    
    def get_tokenizer_type(self):
        return type(self.tokenizer)
    
    def save_sim_matrix(self):
        super().save_sim_matrix()
        
    def get_query_vector(self):
        return self._query_vector
    
    def get_corpus_matrix(self):
        return self._corpus_matrix
    
    def get_vectorizer_type(self):
        return type(self.vectorizer)
    
    def print_topics(self):
        feature_names = self.vectorizer.get_feature_names()
        n_top_words = 10

        for topic_idx, topic in enumerate(self.lda_model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
