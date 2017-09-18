#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:08:48 2017

@author: dt
"""

import pandas as pd

couplets = pd.read_excel("Chinese couplets_TC Lai.xlsx")

couplets['text_filter'] = couplets.Text.apply(lambda x: x.replace("/",'').replace(",",'').replace(":",'').replace(";",'').replace("?",'').replace(".",'').replace("-",' '))
couplets['Character count'] = couplets['text_filter'].apply(len)
couplets['Word Count'] = couplets['text_filter'].apply(lambda x: len(x.split()))


#==============================================================================
# reference: nsyl in HaikuClassificationFunctions.py
#==============================================================================

from HaikuClassificationFunctions import nsyl, count_syl
from nltk.corpus import cmudict
d= cmudict.dict()
couplets['syllables'] = couplets['text_filter'].apply(lambda x: count_syl(x.split(), 0, {}, {}, d))
del couplets['text_filter']

couplets.to_excel("Chinese couplets_TC Lai Count.xlsx", index=False)