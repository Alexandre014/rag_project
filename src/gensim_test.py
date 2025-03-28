# Initialize logging.
import logging
from nltk.corpus import stopwords
from nltk import download
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import gensim.downloader as api
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
first_sentence = 'The tree is green'
second_sentence = 'A dog is barking'

# Import and download stopwords from NLTK.

download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

first_sentence = preprocess(first_sentence)
second_sentence = preprocess(second_sentence)



documents = [first_sentence, second_sentence]
dictionary = Dictionary(documents)

first_sentence = dictionary.doc2bow(first_sentence)
second_sentence = dictionary.doc2bow(second_sentence)

documents = [first_sentence, second_sentence]
tfidf = TfidfModel(documents)

# first_sentence = tfidf[first_sentence]
# second_sentence = tfidf[second_sentence]


model = api.load('word2vec-google-news-300')

termsim_index = WordEmbeddingSimilarityIndex(model)
termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

similarity = termsim_matrix.inner_product(first_sentence, second_sentence, normalized=(True, True))
print('similarity = %.4f' % similarity)