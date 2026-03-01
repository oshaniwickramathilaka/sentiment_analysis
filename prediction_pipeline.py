import pandas as pd
import numpy as np
import re
import string
import pickle
#load model
with open('static/model/model.pickle','rb') as f:
    model=pickle.load(f)

#load stopwords
with open('static/model/corpora/stopwords/english','r') as file:
    sw=file.read().splitlines()

#load vocabulary
vocab=pd.read_csv('static/model/vocabulary.txt',header=None)
tokens=vocab[0].tolist()

from nltk.stem import PorterStemmer
ps=PorterStemmer()


def preprocessing(text):
    data=pd.DataFrame([text],columns=["tweet"])
    
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(word.lower() for word in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: re.sub(r"http\S+|www\S+", " ", x))
    data["tweet"] = data["tweet"].apply(lambda x: re.sub(r"[^a-z\s]", " ", x))
    data["tweet"] = data["tweet"].apply(
        
        lambda x: " ".join(word for word in x.split() if word not in sw)
)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorizer(ds):
    vectorized_list=[]
    for sentence in ds:
        sentence_list=np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_list[i]=1

        vectorized_list.append(sentence_list)

    vectorized_list_new=np.asarray(vectorized_list,dtype=np.float32)
    return vectorized_list_new

def get_prediction(vectorized_txt):
    prediction=model.predict(vectorized_txt)
    if prediction==1:
        
        return 'negative'
    else:
        return 'positive'