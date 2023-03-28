# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:39:09 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

txt_df.to_csv('Cleaned df.csv', index=None)




















