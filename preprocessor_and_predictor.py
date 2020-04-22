import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from bs4 import BeautifulSoup
import urllib
import re
from ast import literal_eval
import os
import glob
import pandas as pd
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import pandas as pd
import pandas as pd
import numpy as np

import re
import nltk
# nltk.download('all')
import json

from collections import Counter
# from textblob import TextBlob
# !pip install contractions
import contractions
from autocorrect import spell
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from keras_preprocessing.text import tokenizer_from_json

import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from wordcloud import WordCloud, STOPWORDS 
def scrape(link):

    link=[link]
    question=" " 
    comments=[]
    explanation=" "

    for i in link:

            try:

                req = urllib.request.Request(str(i), headers={'User-Agent': 'Mozilla/5.0'})
                html = urllib.request.urlopen(req).read()
                bs = BeautifulSoup(html, "html.parser")
                for script in bs(["script", "style"]): # remove all javascript and stylesheet code
                    script.extract()
                titles = bs.find('h1')
#                 print("Question::",count)
                print(titles.text)
                
                


            except Exception as e:
                print(e)
                print("link is broken")
                continue;
            question=titles.text
            coms=[]
            try:
                print("Explanation:")
                exp=bs.find('div',{"data-click-id":"text"}).text
#                 explanation.append(exp)
#                 print(stripTage(exp))
                explanation=exp
                print(exp)
        
                print(explanation)

            except Exception as e:
                print(e)
                explanation=" "



            try:
                print("Comments:::")
                comms=bs.find_all('div',{"data-test-id":"comment"})
    #             print(comms)
                c=1
                for i in comms:
                    print(c,"::",i.text)
                    coms.append(i.get_text())

                    c+=1
                comments.append(coms)
            except:
                comments.append([0])

            print("*"*50)
    
    return [question,explanation,comments]


def preprocess(text):

    text = text.replace("(<br/>)", "")
    text = text.replace('(<a).*(>).*(</a>)', '')
    text = text.replace('(&amp)', '')
    text = text.replace('(&gt)', '')
    text = text.replace('(&lt)', '')
    text = text.replace('(\xa0)', ' ') 
    
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text=re.sub(r'\b\w{1,2}\b', '', text)
    text=text.lower().replace("indian","india")
    return ' '.join(text.split()).replace(r"\s*\([^()]*\)","").strip()
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def get_combined(question,explanation,comments):
    question=preprocess(question)
    explanation=preprocess(explanation)
    cleaned_comments_single=" "
    if comments==[[]]:
        comments=" "
    else:
        c=literal_eval(str(comments[0]).strip())
        l1=list(map(preprocess, c))
    #     cleaned_comments.append(l1)
        cleaned_comments_single=" ".join(l1)
    
    no_contract_question=' '.join(map(str,[contractions.fix(word) for word in question.split()]))
    no_contract_explanation=' '.join(map(str,[contractions.fix(word) for word in explanation.split()]))
    no_contract_comments=' '.join(map(str,[contractions.fix(word) for word in cleaned_comments_single.split()]))
    lower_question=[word.lower() for word in word_tokenize(no_contract_question)]
    lower_explanation=[word.lower() for word in word_tokenize(no_contract_explanation)]
    lower_comments=[word.lower() for word in word_tokenize(no_contract_comments)]
    punc = string.punctuation
    stopwords = set(STOPWORDS) 
    stopwords_removed_question=[word for word in lower_question if word not in punc and word not in stopwords]
    stopwords_removed_explanation=[word for word in lower_explanation if word not in punc and word not in stopwords]
    stopwords_removed_comments=[word for word in lower_comments if word not in punc and word not in stopwords]
    # nltk.tag.pos_tag(stopwords_removed_question)
    wnl = WordNetLemmatizer()
    question_lemma=' '.join(map(str, [wnl.lemmatize(word, tag) for word, tag in [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in nltk.tag.pos_tag(stopwords_removed_question)]]))
    explanation_lemma=' '.join(map(str, [wnl.lemmatize(word, tag) for word, tag in [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in nltk.tag.pos_tag(stopwords_removed_explanation)]]))
    comments_lemma=' '.join(map(str, [wnl.lemmatize(word, tag) for word, tag in [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in nltk.tag.pos_tag(stopwords_removed_comments)]]))
    combined=question_lemma+" "+explanation_lemma+" "+comments_lemma
    combined=" ".join(combined.split())
    return combined

def get_data(link):
    question,explanation,comments=scrape(link)
    combined=get_combined(question,explanation,comments)
    model1=joblib.load("pipeline1.pkl")
    model2=joblib.load("pipeline.pkl")
    model3=tf.keras.models.load_model("lstm1.h5")

    with open('tokenizer.json') as f:
    	data = json.load(f)
    	tokenizer = tokenizer_from_json(data)

    lb1=joblib.load("label_biarizer.pkl")

    answer=[]
    answer.append(model1.predict([combined])[0])
    answer.append(model2.predict([combined])[0])

    max_length=100

    trunc_type='post'
    padding_type='post'
    tokens=tokenizer.texts_to_sequences([combined])

    padded=pad_sequences(tokens, maxlen=max_length, padding=padding_type, truncating=trunc_type)
                                        
    answer.append(lb1.inverse_transform(model3.predict([padded]))[0])


    print(answer)




    return Counter(answer).most_common(1)[0]
    
