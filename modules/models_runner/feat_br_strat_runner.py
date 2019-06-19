from modules.models import model_hyperps as mh
from modules.utils import firefox_dataset_p2 as fd
from modules.utils import model_evaluator as m_eval
from modules.utils import similarity_measures as sm
from modules.utils import tokenizers as tok

from sklearn.feature_extraction.text import TfidfVectorizer

from modules.models_runner.feat_br_models_runner import Feat_BR_Models_Runner

from abc import ABCMeta, abstractmethod

import warnings; warnings.simplefilter('ignore')

class Feat_BR_Generic_Strat_Runner(metaclass=ABCMeta):
    def __init__(self, oracle):
        self.lsi_model = None
        self.lda_model = None
        self.bm25_model = None
        self.wv_model = None
        self.cust_wv_model = None
        self.zeror_model = None
        
        self.oracle = oracle
        
        self.models_runner = Feat_BR_Models_Runner()
        self.evals_df = None
    
    def get_lsi_model(self):
        return self.lsi_model
    
    def get_lda_model(self):
        return self.lda_model
    
    def get_bm25_model(self):
        return self.bm25_model
    
    def get_word2vec_model(self):
        return self.wv_model
    
    def get_cust_word2vec_model(self):
        return self.cust_wv_model
    
    def get_zeror_model(self):
        return self.zeror_model
    
    def get_oracle(self):
        return self.oracle
    
    def get_evaluator(self):
        return self.evaluator
    
    def get_evals_df(self):
        return self.evals_df
        
    def __run_models(self):
        self.lsi_model = self.models_runner.run_lsi_model()
        self.lda_model = self.models_runner.run_lda_model()
        self.bm25_model = self.models_runner.run_bm25_model()
        self.cust_wv_model = self.models_runner.run_cust_word2vec_model()
        self.wv_model = self.models_runner.run_word2vec_model()
        self.zeror_model = self.models_runner.run_zeror_model()

    def __evaluate_models(self):
        self.evaluator = m_eval.ModelEvaluator(self.get_oracle())
        self.evals_df = self.evaluator.run_evaluator(models=[self.get_lsi_model(),
                                                        self.get_lda_model(),
                                                        self.get_bm25_model(),
                                                        self.get_word2vec_model(),
                                                        self.get_cust_word2vec_model(),
                                                        self.get_zeror_model()],
                                               top_values=[1,3,5], 
                                               sim_thresholds=[(sm.SimilarityMeasure.COSINE, x/10) for x in range(0,10)])
    
    def execute(self):
        self.__run_models()
        self.__evaluate_models()

    
class Feat_BR_Vol_Strat_Runner(Feat_BR_Generic_Strat_Runner):
    def __init__(self):
        super().__init__(fd.Feat_BR_Oracles.read_feat_br_volunteers_df().T)
    
    def get_lsi_model(self):
        return super().get_lsi_model()
    
    def get_lda_model(self):
        return super().get_lda_model()
    
    def get_bm25_model(self):
        return super().get_bm25_model()
    
    def get_cust_word2vec_model(self):
        return super().get_cust_word2vec_model()
    
    def get_word2vec_model(self):
        return super().get_word2vec_model()
    
    def get_zeror_model(self):
        return super().get_zeror_model()
    
    def get_oracle(self):
        return super().get_oracle()
    
    def get_evaluator(self):
        return super().get_evaluator()
    
    def get_evals_df(self):
        return super().get_evals_df()
        
    def execute(self):
        super().execute()

        
class Feat_BR_Exp_Strat_Runner(Feat_BR_Generic_Strat_Runner):
    def __init__(self):
        super().__init__(fd.Feat_BR_Oracles.read_feat_br_expert_df().T)
    
    def get_lsi_model(self):
        return super().get_lsi_model()
    
    def get_lda_model(self):
        return super().get_lda_model()
    
    def get_bm25_model(self):
        return super().get_bm25_model()
    
    def get_word2vec_model(self):
        return super().get_word2vec_model()
    
    def get_cust_word2vec_model(self):
        return super().get_cust_word2vec_model()
    
    def get_zeror_model(self):
        return super().get_zeror_model()
    
    def get_oracle(self):
        return super().get_oracle()
    
    def get_evaluator(self):
        return super().get_evaluator()
    
    def get_evals_df(self):
        return super().get_evals_df()
        
    def execute(self):
        super().execute()

        
class Feat_BR_Exp_Vol_Union_Strat_Runner(Feat_BR_Generic_Strat_Runner):
    def __init__(self):
        super().__init__(fd.Feat_BR_Oracles.read_feat_br_expert_volunteers_union_df().T)
    
    def get_lsi_model(self):
        return super().get_lsi_model()
    
    def get_lda_model(self):
        return super().get_lda_model()
    
    def get_bm25_model(self):
        return super().get_bm25_model()
    
    def get_word2vec_model(self):
        return super().get_word2vec_model()
    
    def get_cust_word2vec_model(self):
        return super().get_cust_word2vec_model()
    
    def get_zeror_model(self):
        return super().get_zeror_model()
    
    def get_oracle(self):
        return super().get_oracle()
    
    def get_evaluator(self):
        return super().get_evaluator()
    
    def get_evals_df(self):
        return super().get_evals_df()
        
    def execute(self):
        super().execute()

class Feat_BR_Exp_Vol_Intersec_Strat_Runner(Feat_BR_Generic_Strat_Runner):
    def __init__(self):
        super().__init__(fd.Feat_BR_Oracles.read_feat_br_expert_volunteers_intersec_df().T)
    
    def get_lsi_model(self):
        return super().get_lsi_model()
    
    def get_lda_model(self):
        return super().get_lda_model()
    
    def get_bm25_model(self):
        return super().get_bm25_model()
    
    def get_word2vec_model(self):
        return super().get_word2vec_model()
    
    def get_cust_word2vec_model(self):
        return super().get_cust_word2vec_model()
    
    def get_zeror_model(self):
        return super().get_zeror_model()
    
    def get_oracle(self):
        return super().get_oracle()
    
    def get_evaluator(self):
        return super().get_evaluator()
    
    def get_evals_df(self):
        return super().get_evals_df()
        
    def execute(self):
        super().execute()