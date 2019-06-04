import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from modules.utils import plots
from modules.utils import firefox_dataset_p2 as fd
from modules.utils import tokenizers as tok
from modules.utils import aux_functions

from modules.models.lda import LDA
from modules.models.lsi import LSI
from modules.models.bm25 import BM_25
from modules.models.wordvec import WordVec_BasedModel
from modules.models.zeror import ZeroR_Model

import modules.models.model_hyperps as mh

class Feat_BR_Models_Hyperp:
    
    @staticmethod
    def get_lsi_model_hyperp():
        return {
            mh.LSI_Model_Hyperp.SVD_MODEL_N_COMPONENTS.value: 20,
            mh.LSI_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            mh.LSI_Model_Hyperp.VECTORIZER_MAX_FEATURES.value: 400,
            mh.LSI_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            mh.LSI_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.WordNetBased_LemmaTokenizer()
        }
    
    @staticmethod
    def get_lda_model_hyperp():
        return {
            mh.LDA_Model_Hyperp.LDA_MODEL_N_COMPONENTS.value: 20,
            mh.LDA_Model_Hyperp.LDA_MODEL_RANDOM_STATE.value : 2,
            mh.LDA_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            mh.LDA_Model_Hyperp.VECTORIZER_MAX_FEATURES.value: 200,
            mh.LDA_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            mh.LDA_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_bm25_model_hyperp():
        return {
            mh.BM25_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_w2v_hyperp():
        return {
            mh.WordVec_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

class Feat_BR_Models_Runner:
    def __init__(self, features=pd.DataFrame(), bugreports=pd.DataFrame()):
        self.features_df = None
        self.bug_reports_df = None
        self.corpus = None
        self.query = None
        self.features_names = None
        self.bug_reports_names = None
        
        self.set_basic_params(features, bugreports)
    
    def set_basic_params(self, features, bugreports):
        if features.empty:
            self.features_df = fd.Datasets.read_features_df()
        else:
            self.features_df = features
            
        if bugreports.empty:
            self.bug_reports_df = fd.Datasets.read_selected_bugreports_df()
        else:
            self.bug_reports_df = bugreports
        
        self.corpus = self.features_df.feat_desc
        self.query = self.bug_reports_df.br_desc

        self.features_names = self.features_df.feat_name
        self.bug_reports_names = self.bug_reports_df.Bug_Number
    
    def run_lsi_model(self, lsi_hyperp=None):
        print("Running LSI model -----")
        
        if lsi_hyperp == None:
            lsi_hyperp = Feat_BR_Models_Hyperp.get_lsi_model_hyperp()

        lsi_model = LSI(**lsi_hyperp)
        lsi_model.set_name('LSI_Model_Feat_BR')
        lsi_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)
       
        return lsi_model
    
    def run_lda_model(self, lda_hyperp=None):
        print("Running LDA model -----")
        
        if lda_hyperp == None:
            lda_hyperp = Feat_BR_Models_Hyperp.get_lda_model_hyperp()

        lda_model = LDA(**lda_hyperp)
        lda_model.set_name('LDA_Model_Feat_BR')
        lda_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)
        
        return lda_model
    
    def run_bm25_model(self, bm25_hyperp=None):
        print("Running BM25 model -----")
        
        if bm25_hyperp == None:
            bm25_hyperp = Feat_BR_Models_Hyperp.get_bm25_model_hyperp()

        bm25_model = BM_25(**bm25_hyperp)
        bm25_model.set_name('BM25_Model_Feat_BR')
        bm25_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)
        
        return bm25_model
    
    def run_word2vec_model(self, wv_hyperp=None):
        print("Running W2V model -----")
        
        if wv_hyperp == None:
            wv_hyperp = Feat_BR_Models_Hyperp.get_w2v_hyperp()

        wv_model = WordVec_BasedModel(**wv_hyperp)
        wv_model.set_name('WordVec_Model_Feat_BR')
        wv_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)

        return wv_model
    
    def run_zeror_model(self, zeror_hyperp=None):
        print("Running ZeroR model -----")
        
        oracle = fd.Feat_BR_Oracles.read_feat_br_expert_volunteers_intersec_df().T
        
        zeror_model = ZeroR_Model(oracle)
        zeror_model.set_name('ZeroR_Model_Feat_BR')
        zeror_model.recover_links()
        
        return zeror_model