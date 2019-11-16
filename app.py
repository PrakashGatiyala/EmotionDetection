from flask import Flask,render_template,url_for,request
#from bootstrap_flask import Bootstrap
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
#from bs4 import BeautifulSoup
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("train_chat.csv")
    df['chat'] = df['turn1'] + " " + df['turn2']+ " " + df['turn3']
    df['label'] = df['label'].map({'happy':1, 'stress':2,'angry':3,'others':4})
    import string
    string.punctuation
    def remove_punct(stng):
         text_nonpunct = "".join([i for i in stng if i not in string.punctuation])
         return text_nonpunct
    df['chat']=df['chat'].apply(lambda x:remove_punct(x))
    #tokenising
    import re
    def tokenize(strg):
        tokens=re.split('\W+',strg)
        a=[]
        for i in tokens:
            if i.isdigit()==False:
                a.append(i)
        return a 
    x =['chat']
    for col in x :
        df[col]=df[col].apply(lambda x:tokenize(x))

    stopwords=nltk.corpus.stopwords.words('english')
    def remove_stopwords(tokn_lt):
        text=[w for w in tokn_lt if w not in stopwords]
        return text
    for col in x:    
        df[col]=df[col].apply(lambda x:remove_stopwords(x))
    
    ps=nltk.PorterStemmer()
    wn=nltk.WordNetLemmatizer()

    def stemming(tok_lt):
        text=[ps.stem(word) for word in tok_lt]
        return text

    def lemmatizing(tok_lt):
        text=[wn.lemmatize(word) for word in tok_lt]
        return text

    for col in x:    
        df[col]=df[col].apply(lambda x:stemming(x))
        df[col]=df[col].apply(lambda x:lemmatizing(x))

    #joining all text
    def join_text(text):
         a=" ".join(text)
         return a

    for col in x :
        df[col]=df[col].apply(lambda x:join_text(x))


    df_x = df['chat']
    df_y = df['label']
    vect = TfidfVectorizer()
    x_vect = vect.fit_transform(df['chat'])
    X_train, X_test, y_train, y_test = train_test_split(x_vect, df_y, test_size=0.20, random_state=42)
    clf  = LinearSVC()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        Vect = vect.fit_transform(data).toarray()
        my_prediction = clf.predict(Vect)
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
