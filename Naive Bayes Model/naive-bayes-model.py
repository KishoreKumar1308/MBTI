import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as es
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
import pickle


nltk.download(["wordnet","punkt","stopwords"])
es_stop_words = set(stopwords.words("english"))

data_mbti = pd.read_csv("mbti_1.csv")

# Cleaning Data by removing URLs, ||| and other non alphebatic symobls which will have no impact on model building
mbti_posts = data_mbti.posts.replace('+', ' ').replace('.', " ").replace(',', ' ').replace(':', ' ')
for index,post in enumerate(mbti_posts.values):
    mbti_posts.values[index] = re.sub('http\S+', '',post)

for index,post in enumerate(mbti_posts.values):
    mbti_posts.values[index] = re.sub(r'\|\|\|', r' ',post)

for index,post in enumerate(mbti_posts.values):
    mbti_posts.values[index] = re.sub(r'[^a-zA-Z]', r' ',post)

for index,post in enumerate(mbti_posts.values):
    mbti_posts.values[index] = re.sub(r' +', r' ',post).lower()

mbti_posts = pd.DataFrame(mbti_posts)
data_mbti["Cleaned Posts"] = mbti_posts

X = data_mbti["Cleaned Posts"]
Y = data_mbti["type"]

lemmatizer = WordNetLemmatizer()
snow_stemmer = SnowballStemmer(language='english')

for idx,temp in enumerate(X):
    token_list = word_tokenize(temp)
    filtered_list = [word for word in token_list if not word in es_stop_words]
    temp = " ".join(filtered_list)

    temp = lemmatizer.lemmatize(temp)
    temp = snow_stemmer.stem(temp)
    X[idx] = temp

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25,random_state = 268)

vectorizer = CountVectorizer(tokenizer = str.split, stop_words = es)
X_train_vect = vectorizer.fit(X_train)
pickle.dump(vectorizer,open("vect_fit.pickle","wb"))

vector = CountVectorizer(tokenizer = str.split, stop_words = es)
X_train_vect = vectorizer.fit_transform(X_train)
pickle.dump(vector, open("vect_transform.pickle","wb"))
X_test_vect = vectorizer.transform(X_test)


nb = MultinomialNB() 
nb.fit(X_train_vect,y_train)


pickle.dump(nb,"mbti-model-nb.sav")
