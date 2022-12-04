#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 03:09:45 2022

@author: aloudra
"""

#importing necessary libraries
from afinn import Afinn
import pandas as pd
import numpy as np
import pandas as pd

#instantiate afinn
afn = Afinn()

df = pd.read_csv('youtube-comments.csv')  
#creating list sentences
news_df = df["Comment"]

# compute scores (polarity) and labels
scores = [afn.score(article) for article in news_df]
sentiment = ['positive' if score > 0
                        else 'negative' if score < 0
                            else 'neutral'
                                for score in scores]

# dataframe creation
df = pd.DataFrame()
df['topic'] = news_df
df['scores'] = scores
df['sentiments'] = sentiment
print(df)

df.to_csv('Sentiment Analysis.csv')


###NLTK

from nltk.sentiment import SentimentIntensityAnalyzer
import operator
sia = SentimentIntensityAnalyzer()
df["sentiment_score"] = df["Comment"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])

##TextBlob

from textblob import TextBlob
df["sentiment_score"] = df["Comment"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])

##Flair

from flair.models import TextClassifier
from flair.data import Sentence
sia = TextClassifier.load('en-sentiment')
def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"
df["sentiment"] = df["Comment"].apply(flair_prediction)




