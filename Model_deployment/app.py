from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import re
import string
import requests
from string import digits
from sklearn import linear_model
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
  

#Import Flask modules
from flask import Flask, request, render_template
import joblib

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'templates')

#Open our model 
ml_model = pickle.load(open('ml_model.pkl','rb'))
dl_model = pickle.load(open('dl_model.pkl','rb'))

tfidf_vect = joblib.load('tfidf_vec.pkl')
    
###################################################
#Function
def pre_processing(text):
  #remove username
  text = re.sub('@[^\s]+', '', text)

  #remove hashtags
  text = re.sub('#', '', text)

  #remove English characters
  text = re.sub(r'\b[A-Za-zÀ-ž]\b',' ', text)

  #remove RT (Retweet)
  text =  re.sub('RT','',text)

  #remove emojis
  emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+")   

  text = emoji_pattern.sub(r'', text)

  #remove digits
  text =  text.translate(str.maketrans('', '', digits))

  #remove punctuation
  arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|\!”…“–ـ'''
  english_punctuations = string.punctuation
  punctuations_list = arabic_punctuations + english_punctuations
  text= text.translate(str.maketrans('','',punctuations_list))

  #Normalization
  text= re.sub(r'(.)\1+', r'\1', text) 

  #remove repeating characters
  text= re.sub(r'(.)\1+', r'\1', text) 

  #remove single character
  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
  
  # #remove stopwords
  # import arabicstopwords.arabicstopwords as stp
  # stop_words = list(stp.classed_stopwords_list())
  # text= ' '.join([word for word in text.split() if word not in (stop_words)])

  #stemming
  from nltk.stem.isri import ISRIStemmer
  st = ISRIStemmer()
  text= st.stem(text)

  # #tokenization
  # from nltk.tokenize import RegexpTokenizer
  # tokenizer = RegexpTokenizer(r'\w+')
  # text = tokenizer.tokenize

  return text
###################################################
#create our "home" route using the "index.html" page
@app.route('/')
def index():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/predict', methods = ['POST'])
def predict():
  url = 'https://recruitment.aimtechnologies.co/ai-tasks'
  tweet_id = request.form.to_dict()
  to_predict_list = list(requests.post(url, json=[tweet_id['review_text']]).json().values())[0]
  text = pre_processing(to_predict_list)
  
  if text == 'ID not found':
    pred = text
  else:
    pred = dl_model.predict(tfidf_vect.transform([text]))[0]
      
  return render_template('predict.html', prediction = pred)
  
if __name__ == '__main__':
    app.run(debug=True)
