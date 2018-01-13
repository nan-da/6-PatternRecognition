#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:46:15 2017

@author: dt
"""

from HaikuClassificationFunctions import *
import pandas as pd

# customized training set
def excel_corpus(excel_name, max_length, trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d, short_name):
    short_labeled = []
    lemma = 'y' #raw_input("Lemmatize? y/n?")
    
    if type(excel_name) == str:
        outsource = pd.read_excel(excel_name)
    elif type(excel_name) == pd.DataFrame:
        outsource = excel_name
    else:
        raise Exception("Unexpected type: ", type(excel_name))
    
    for filenum, raw in tqdm(outsource.values):
        if len(raw) <= max_length:
            if count_syls == 1:
                raw = clean_poetry(raw, 0)
                text_words = nltk.word_tokenize(raw.lower())
                poem_syl = count_syl(text_words, -1, text_syl_dict, temp_syl_dict, d)            #get syllable count of poem

            raw = clean_poetry(raw, 1)                  #clean again, but this time do not exclude "!"    
            text_words = nltk.word_tokenize(raw.lower())
            if lemma == "y":
                wnl = nltk.WordNetLemmatizer()
                text_words = [wnl.lemmatize(w) for w in text_words] 
            feature = bag_of_non_stopwords(text_words, stopfile='english')
            if count_syls == 1:
                if trans_or_adapt == "trans":
                    if poem_syl <= 18:                          #use <= 20 for pre-1914 corpus; use <= 22 for post-1914 corpus
                        feature["less_than_18_syl"] = True
                    if poem_syl > 18:
                        feature["less_than_18_syl"] = False
                else:
                    if poem_syl <= 19:                          #use <= 20 for pre-1914 corpus; use <= 22 for post-1914 corpus
                        feature["less_than_19_syl"] = True
                    if poem_syl > 19:
                        feature["less_than_19_syl"] = False
                    if poem_syl > 19 and poem_syl <= 31:       #this captures a second bin; only use for post-1914 hokku
                        feature["btw_19_and_31"] = True
                    if poem_syl < 19 and poem_syl > 31:
                        feature["btw_19_and_31"] = False 
            label = 'not-haiku'
            short_labeled.append(['{}{}'.format(short_name, filenum), [feature, label]])  
    return short_labeled    