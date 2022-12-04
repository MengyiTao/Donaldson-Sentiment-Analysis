#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 21:23:33 2022

@author: aloudra
"""





import re
import numpy as np
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv("youtube-comments-final.csv", encoding='latin1')

df['Comment'] = df['Comment'].str.encode('ascii','ignore')
df['Comment'] = df['Comment'].str.decode('ascii','ignore')

df['Comment'] = df['Comment'].str.replace('[^\w\s]','')
df['Comment'] = df['Comment'].str.replace('[\n]','')

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


df['Comment'] = df['Comment'].apply(str)
df['Comment'] = df['Comment'].apply(lambda x: remove_emoji(x))

with open('/content/Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

def convert_emojis_to_word(text):
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
    return text

df['Comment'] = df['Comment'].apply(str)
df['Comment'] = df['Comment'].apply(lambda x: convert_emojis_to_word(x))

df['Comment'] = df['Comment'].str.replace('[_]','')

df.to_csv('youtube-comments.csv', index=False, header=False)

