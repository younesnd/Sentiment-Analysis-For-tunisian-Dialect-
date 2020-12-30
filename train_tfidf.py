import pickle
from sklearn import svm
import pandas as pd 
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
#upload your dataset that you had cross validated 
#df= pd.read_csv('data_path')
fold=6
train_df = df[df.kfold != fold].reset_index(drop=True)
val_df = df[df.kfold == fold].reset_index(drop=True)
tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize,token_pattern=None)
tfidf_vec.fit(train_df.text)
xtrain = tfidf_vec.transform(train_df.text)
x_val = tfidf_vec.transform(val_df.text)
# save tfidf_vec as pickle file to reuse it later 
pkl_fil5 = "tfidf_model.pkl"
with open(pkl_fil5, 'wb') as file:
    pickle.dump(tfidf_vec, file)
# train the model "SVC" support vector classifier 
# you can use any classifier you want  
model = svm.SVC()
model.fit(xtrain, train_df.label)
#save the SVC model as pickle file
pkl_fil6 = "pickle_model.pkl"
with open(pkl_fil6, 'wb') as file:
    pickle.dump(pickle_model, file)
