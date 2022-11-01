# # -*- coding: utf-8 -*-
# """
# Created on Thu May  7 00:40:21 2020
#
# @author: ksrikond
# """
# import csv
# import nltk
# import pandas as pd
# import re
# from nltk.corpus import stopwords
# stop = stopwords.words('english')
# words = set(nltk.corpus.words.words())
#
# def extract_numbers(string):
#     r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
#     phone_numbers = r.findall(string)
#     return [re.sub(r'\D', '', number) for number in phone_numbers]
#
# def extract_email_addresses(string):
#     r = re.compile(r'[\w\.-]+@[\w\.-]+')
#     return r.findall(string)
#
# def ie_preprocess(document):
#     document = ' '.join([i for i in document.split() if i not in stop])
#     sentences = nltk.sent_tokenize(document)
#     sentences = [nltk.word_tokenize(sent) for sent in sentences]
#     sentences = [nltk.pos_tag(sent) for sent in sentences]
#     return sentences
#
# def extract_names(document):
#     names = []
#     sentences = ie_preprocess(document)
#     for tagged_sentence in sentences:
#         for chunk in nltk.ne_chunk(tagged_sentence):
#             if type(chunk) == nltk.tree.Tree:
#                 if chunk.label() == 'PERSON':
#                     names.append(' '.join([c[0] for c in chunk]))
#     return names
#
# #df=pd.read_csv("W:\hsd-es-study\hsd_query1.csv")
# #for row in df.iterrows():
# #print (row)
# f=open('W:\hsd-es-study\demo1.csv',"w")
# with open('W:\hsd-es-study\hsd_query1.csv',encoding='utf-8') as file:
#     for line in file.readlines():
#         #print (line)
#         #print("next line")
#         #x=" ".join(w for w in nltk.wordpunct_tokenize(line) \
#         #     if w.lower() not in words)
#         line=re.sub("\<(.*?)\>","",line)
#         for w in nltk.wordpunct_tokenize(line):
#             if w.lower() not in words:
#                f.write(w)
#
# f.close()
#         #    print(w)
#         #print (x)
#     #print (x)
#     #file_contents = file.read()
#     #print (file_contents)
#     #for row in file_contents:
#     #    print(row)
#     #x=" ".join(w for w in nltk.wordpunct_tokenize(row) \
#     #     if w.lower() not in words or not w.isalpha())
#     #print (x)
#
#
# #sent = "Io andiamo to the beach with my amico."
#
# #x=" ".join(w for w in nltk.wordpunct_tokenize(sent) \
# #         if w.lower() not in words or not w.isalpha())
# #print (x)
# print("good")
#
#
#
#
