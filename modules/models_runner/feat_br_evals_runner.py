import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from modules.utils import plots
from modules.utils import firefox_dataset_p2 as fd
from modules.utils import tokenizers as tok
from modules.utils import aux_functions
from modules.utils import model_evaluator as m_eval
from modules.utils import similarity_measures as sm

from modules.models.lda import LDA
from modules.models.lsi import LSI
from modules.models.bm25 import BM_25
from modules.models.wordvec import WordVec_BasedModel

from modules.models.model_hyperps import LDA_Model_Hyperp
from modules.models.model_hyperps import LSI_Model_Hyperp
from modules.models.model_hyperps import BM25_Model_Hyperp
from modules.models.model_hyperps import WordVec_Model_Hyperp



class Feat_BR_Evals_Runner:
    def __init__(self, oracle):
        self.orc = oracle
        
    def run_evaluator(self, model, verbose=False):
        evals = []
        evaluator = m_eval.ModelEvaluator(self.orc, model)
        
        for top_value in [1,3,5,10]:
            for cos_value in [.0]:
                ref_name = "top_{}_cos_{}".format(top_value, cos_value)
                evals.append(evaluator.evaluate_model(verbose=verbose, 
                                                      top_value=top_value, 
                                                      sim_threshold=(sm.SimilarityMeasure.COSINE, cos_value), 
                                                      ref_name=ref_name))
        return evals
    
    