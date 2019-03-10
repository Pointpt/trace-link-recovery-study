from enum import Enum

class SimilarityMeasure(Enum):
    # LSI and others -------
    COSINE = 'cosine'
    JACCARD_INDEX = 'jaccard'
    EDIT_DISTANCE = 'edit'
    
    # LDA ------------
    JSD = 'jsd'