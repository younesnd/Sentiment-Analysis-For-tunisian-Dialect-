from nltk.tokenize import word_tokenize
import pickle
import nltk
nltk.download('punkt')
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

pkl_file = r'/home/younesnd/Downloads/tfidf_model.pkl'
with open(pkl_file, 'rb') as file:
    td_if_load = pickle.load(file)
