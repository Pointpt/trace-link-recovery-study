import nltk

"""
Others stemmers are not relevant for our analysis:
 . RSLP Stemmer: portuguese language
 . ISRIS Stemmer: returns Arabic root for the given token 
 . Regexp Stemmer: uses regulax expressions to identify morphological affixes
 
Relevant Stemmers/Lemmatizers are implemented below. 
"""

class GenericTokenizer(object):
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
    def __call__(self, doc):
        tokens = [self.stemmer.stem(token) for token in nltk.word_tokenize(doc)]
        #return [token.lower() for token in tokens if token.isalpha() and token not in self.stopwords and len(token) > 1]
        return [token.lower() for token in tokens if token not in self.stopwords]
        
class WordNetBased_LemmaTokenizer(GenericTokenizer):
    def __init__(self):
        super().__init__()
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(token) for token in nltk.word_tokenize(doc)]
        return [token.lower() for token in tokens if token.isalpha() and token not in self.stopwords]

class LancasterStemmerBased_Tokenizer(GenericTokenizer):
    def __init__(self):
        super().__init__()
        self.stemmer = nltk.stem.LancasterStemmer()
    def __call__(self, doc):
        return super().__call__(doc)

class PorterStemmerBased_Tokenizer(GenericTokenizer):
    def __init__(self):
        super().__init__()
        self.stemmer = nltk.stem.PorterStemmer()
    def __call__(self, doc):
        return super().__call__(doc)
    
class SnowballStemmerBased_Tokenizer(GenericTokenizer):    
    def __init__(self):
        super().__init__()
        self.stemmer = nltk.stem.SnowballStemmer('english')    
    def __call__(self, doc):
        return super().__call__(doc)
        