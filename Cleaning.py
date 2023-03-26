# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:39:09 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize

# Reading in text file
with open ('Complete_data .txt') as f:
    txt = f.readlines()

# Parsing out numbers of paragraphs
txt_df = pd.DataFrame(txt, columns=['text'])

txt_df['num'] = txt_df.text.apply(lambda x: x if len(x) < 8 else 0)
d_ind = txt_df[txt_df.num != 0].index

txt_df = txt_df.replace(0, np.NaN)
txt_df = txt_df.ffill(axis=0)

txt_df = txt_df.drop(d_ind,axis=0)
txt_df.num = txt_df.num.astype('float')

txt_df = txt_df.reset_index().drop('index',axis=1)

#LP
string = txt_df.text[0]
sent_string = sent_tokenize(string)
word_string = word_tokenize(string)

#stopwords
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

filtered_list = [word for word in word_string if word.casefold() not in stopwords]

#stemming
# from nltk.stem import PorterStemmer
# stremmer = PorterStemmer()

# stemmed = [stremmer.stem(word) for word in filtered_list]

#pos tagging
tagged = pd.DataFrame(nltk.pos_tag(word_string), columns=['words','tag'])

punc = ['.',',','`','?',':',';','``',"''",'(',')']
tagged['punc'] = tagged.tag.apply(lambda x: 1 if x in punc else 0)
tagged = tagged.drop(tagged[tagged.punc == 1].index).drop('punc',axis=1)

tagged.tag.value_counts()

#lemmatizing
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

tagged['lemmatized'] = [lemmatizer.lemmatize(word) for word in tagged['words']]

tagged.loc[tagged['lemmatized'] != tagged['words']].shape[0]

#chunking
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(list(zip(tagged['lemmatized'],tagged['tag'])))
tree.draw()

#chinking
grammar = """
    Chunk: {<.>+.}
           }<JJ>{"""
chink_parser = nltk.RegexpParser(grammar)

tree = chink_parser.parse(list(zip(tagged['lemmatized'],tagged['tag'])))
tree.draw()

#using name entity recognition
nltk.download("maxent_ne_chunker")
nltk.download("words")

tree = nltk.ne_chunk(list(zip(tagged['lemmatized'],tagged['tag'])))
tree.draw()


























