# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:07:29 2020

@author: ksrikond
"""

# !/usr/bin/env python

import sys
from string import punctuation
from operator import itemgetter
import csv
import nltk
import pandas as pd
# import re
from nltk.corpus import stopwords
from nltk.corpus import brown
from pandas import DataFrame

stop = stopwords.words('english')
stop.extend(["hsd-es", "server", "bugeco", "sprsp", "subject", "thanks", " ", ""])
english_words = set(nltk.corpus.words.words())
brown_words = set(brown.words())
# print(english_words)
stop.extend(english_words)
stop.extend(brown_words)
# words = set(nltk.corpus.words.words())
# print (stop)
# Check command line inputs
# if len(sys.argv) == 1:
#    print ('Pass the input text file as the first argument.')
#    sys.exit()
# elif len(sys.argv) == 2:
#    infile = sys.argv[1]
#    outfile = '%s.html' % (infile.split('.')[0],)
# else:
infile = "W:\hsd-es-study\hsd_query1.csv"
outfile = "W:\hsd-es-study\demo1_out.txt"

print(infile, outfile)
total_words = 0
N = 10
words = {}  # Dict of word frequencies
pos = {}  # Dict of word positions
scores = []  # List of word repeatedness scores
d_scores = {}

# articles = ['the', 'a', 'of', 'and', 'in', 'et', 'al','for','is','at','on','this'] # Common articles to ignore

# Build lists

words_gen = (word.strip(punctuation).lower() for line in open(infile, encoding='utf-8', errors='ignore')
             for word in line.split())

for word in words_gen:
    if word not in stop and "intel" not in word:
        total_words += 1
print(total_words)
words_gen = (word.strip(punctuation).lower() for line in open(infile, encoding='utf-8', errors='ignore')
             for word in line.split())

i = 0

for word in words_gen:
    if word not in stop and "intel" not in word:
        # total_words+=1
        words[word] = words.get(word, 0) + 1
        d_scores[word] = (words[word] / total_words) * 100
        # Build a list of word positions
        if words[word] == 1:
            pos[word] = [i]
        else:
            pos[word].append(i)

    i += 1

# Calculate scores

words_gen = (word.strip(punctuation).lower() for line in open(infile, encoding='utf-8', errors='ignore')
             for word in line.split())

# =============================================================================
# for word in words_gen:
#     if words[word]>10:
#         print (word)
#         print (total_words)
#         d_score[word]
# =============================================================================
###Relative Postion  --- Needs fixing
# =============================================================================
# i = 0
# for word in words_gen:
#     scores.append(0)
# #    scores[i] = -1 + sum([pow(2, -abs(d-i)) for d in pos[word]]) # The -1 accounts for the 2^0 for self words
#     if word not in articles and len(word) > 1:
#         for d in pos[word]:
#             if d != i and abs(d-i) < 50:
#                 scores[i] += 1.0/abs(d-i)
#     i += 1
# =============================================================================

# scores = [score*1.0/max(scores) for score in scores] # Scale from 0 to 1

# Write colored output


f = open(outfile, 'w', encoding='utf-8');
i = 0
newDict_words = {key: value for (key, value) in sorted(d_scores.items(), key=lambda item: item[1], reverse=True)}
# print (newDict_words)
df = DataFrame(list(newDict_words.items()), columns=['Words', 'Weight'])
df.to_csv("W:\hsd-es-study\demo_out_1.csv", encoding='utf-8')
f.write(str(newDict_words))
# =============================================================================
# for line in open(infile,encoding='utf-8',errors='ignore'):
#     for word in line.split():
#         #f.write('<span style="background: rgb(%i, 255, 255)">%s</span> ' % ((1-scores[i])*255, word))
#         #i += 1
#          f.write(word+":[  ")
#          f.write(str(scores[i])+"]  ")
#          i+=1
#     f.write('<br /><br />')
# f.close()
# 
# =============================================================================

print('Output saved to %s' % (outfile,))
