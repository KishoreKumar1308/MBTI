import numpy as np
import pandas as pd
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk

nltk.download(["wordnet","punkt","stopwords"])
es_stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
snow_stemmer = SnowballStemmer(language='english')

def scrape(user):
    os.system(f"snscrape --jsonl --max-results 100 twitter-search 'from:{user}' > user-json-tweets.json")

    user_json = pd.read_json("user-json-tweets.json",lines = True)
    user_json.to_csv("user-tweets.csv")
    df = pd.read_csv("user-tweets.csv")

    tweet_list = []
    for tweet in df["renderedContent"]:
        tweet_list.append(tweet)

    user_string = "|||".join(tweet_list)

    clean_string = user_string.replace("+"," ").replace('.', " ").replace(',', ' ').replace(':', ' ').replace('@', '')
    clean_string = re.sub(r'\|\|\|',r' ',clean_string)
    clean_string = re.sub('http\S+','',clean_string)
    clean_string = re.sub(r"@\S+","",clean_string)
    clean_string = re.sub(r'[^a-zA-Z]',r' ',clean_string)
    clean_string = re.sub(r' +',r' ',clean_string).lower()
    
    return clean_string