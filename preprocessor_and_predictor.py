import warnings
warnings.filterwarnings("ignore")

import re

import pandas as pd

import contractions



import numpy as np


import nltk

import json

from collections import Counter



# from keras_preprocessing.text import tokenizer_from_json

import joblib
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from wordcloud import STOPWORDS
import praw

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


reddit = praw.Reddit(client_id = "VmSi9YqPqeslPQ",
					client_secret = "aVxcXEnFX6UEuc0zBq4Lboi32Go",
					user_agent = "detector1",
					username = "saireddyavs",
					password = "1234567890")



def get_data_from_link(link):
    submission = reddit.submission(url = link)
    question=str(submission.title)
    explanation=str(submission.selftext)
    submission.comments.replace_more(limit=None)
    comment=""
    count = 0

    for top_level_comment in submission.comments:
            comment+=str(top_level_comment.body)
            count+=1
            if(count > 10):
                break

    return " ".join(str(question+" "+explanation+" "+comment).split())




def get_data(links):



    combined=list(map(get_data_from_link,links))






    df=pd.DataFrame({'combined':combined})


    df.combined=df.combined.map(lambda x:contractions.fix(str(x)))

    url_reg  = r'[a-z]*[:.]+\S+'

    df.combined=df.combined.map(lambda x:re.sub(url_reg,"",x))
    df.combined=df.combined.map(lambda x:" ".join(re.sub("([^A-Za-z0-9])|(\b\w{1,2}\b)"," ",x).split()))

    stopwords = set(STOPWORDS)

    porter_stemmer = nltk.stem.PorterStemmer()
    df['combined']=df.combined.map(lambda x:" ".join([porter_stemmer.stem(i)  for i in x.split() if i not in stopwords]))















    model1=joblib.load("pipeline1.pkl")
    # model2=joblib.load("pipeline.pkl")
    # model3=tf.keras.models.load_model("lstm1.h5")
	#
    # with open('tokenizer.json') as f:
    # 	data = json.load(f)
    # 	tokenizer = tokenizer_from_json(data)
	#
    # lb1=joblib.load("label_biarizer.pkl")

    # answer=[]
    arr1=model1.predict(df.combined)
#     arr2=model2.predict(df.combined)
#
#     max_length=100
#
#     trunc_type='post'
#     padding_type='post'
#     tokens=tokenizer.texts_to_sequences(df.combined)
#
#     padded=pad_sequences(tokens, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#
#     arr3=lb1.inverse_transform(model3.predict(padded))
#
#
# #     print(answer)
#
#
#     m = np.vstack([arr1,arr2,arr3])
#
#
#
#
#
#     modes_counter = []
#
#     for c in m.T:
#         modes_counter.append(Counter(c).most_common(1)[0][0])



    res = dict(zip(links,list(arr1)))


    print(reddit.auth.limits)


#     resp = jsonify(res)

#     resp.status_code = 200

    return res
