import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google_play_scraper import Sort, reviews_all
from nlp_id.lemmatizer import Lemmatizer

import re
import matplotlib.pyplot as plt
import csv
import sys
import string, unicodedata

# warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import nltk
from nltk.corpus import words
nltk.download('words')

# stopoword bahasa Indonesia
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
listStopword =  set(stopwords.words('indonesian'))

def scrap(alamat):
  result = pd.DataFrame(reviews_all(alamat,
                                      lang='id',
                                      country='id',
                                      sort=Sort.NEWEST))
  result1 = result[['userName','content','score',	'at','reviewCreatedVersion']]
  return result1

def Case_Folding(text):
    #menghapus link
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

    # Menghapus Tanda Baca
    text = re.sub(r'[^\w]|_',' ', text)

    # Hapus non-ascii
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Menghapus Angka
    text = re.sub(r"\S*\d\S*", "", text).strip()
    text = re.sub(r"\b\d+\b", " ", text)

    # Mengubah text menjadi lowercase
    text = text.lower()

    # Menghapus white space
    text = re.sub(r'[\s]+', ' ', text)

    return text

def split_word(teks):
    list_teks = []
    for txt in teks.split(" "):
        list_teks.append(txt)
    return list_teks

def satu(teks):
  text = (" ").join(teks)
  return text

slang_dictionary = pd.read_csv('https://raw.githubusercontent.com/fawazdhianabil/AKSEL/main/slangformal.csv')
slang_dict = pd.Series(slang_dictionary['slang'].values,index=slang_dictionary['formal']).to_dict()

def normalisasi(text):
    for word in text:
        if word in slang_dict.keys():
            text = [slang_dict[word] if word == s else s for s in text]
    return text

def remove_stopwords(token):
    text = [word for word in token if word not in listStopword]
    return text

def remove_stopwords2(teks):
    unwanted_words = ['https', 'google','kamu','ga', 'gue', 'lo', 'lu', 'dan','saya','dia','c','ente','elu','nya','di','ini','ke']
    text = [word for word in teks if not word in unwanted_words]
    return text

from nlp_id.lemmatizer import Lemmatizer
lemmatizer = Lemmatizer()

# untuk sentimen
def sentimen(df):
  text_clean = []

  for idx, text in enumerate(df['content']):
    clean_text = str(text).replace(str(df['userName'][idx]), '')
    clean_text = re.sub(r'@[\w]+','',clean_text)
    text_clean.append(clean_text)

  df['Text_Clean'] = text_clean

  df['Text_Clean'] = df['Text_Clean'].drop_duplicates()
  df = df.dropna()
  df = df.reset_index(drop=True)

  df['Case_Folding'] = df['Text_Clean'].apply(Case_Folding)
  df['Tokenizing'] = df['Case_Folding'].apply(split_word)
  df['Normalisasi'] = df['Tokenizing'].apply(normalisasi)
  df['Normalisasi'] = df['Normalisasi'].apply(satu).str.replace('enggak', 'tidak').apply(split_word)
  df['Stopword'] = df['Normalisasi'].apply(lambda x: remove_stopwords(x))
  df['Stopword'] = df['Normalisasi'].apply(lambda x: remove_stopwords2(x))
  df['Lemmatisasi'] = df['Stopword'].apply(satu).apply(lemmatizer.lemmatize).apply(split_word)
  df['Text_Clean2'] = df['Lemmatisasi'].apply(satu).str.findall(r'\w{2,}').str.join(' ').apply(split_word)

  df['Final_Cek'] = df['Text_Clean2'].apply(satu)
  df['Final_Cek'] = df['Final_Cek'].str.replace('engenggakk', 'tidak')
  df['Final_Cek'] = df['Final_Cek'].str.replace('enggak', 'tidak')
  df['Final_Cek'] = df['Final_Cek'].str.replace('apk', 'aplikasi')
  df['Final_Cek'] = df['Final_Cek'].str.replace('bsih', 'bsi')
  df['Final_Cek'] = df['Final_Cek'].str.replace('aplikasih', 'aplikasi')
  df['Final_Cek'] = df['Final_Cek'].str.replace('broken', 'patah')
  df['Final_Cek'] = df['Final_Cek'].str.replace('kecewa', 'kekecewaan')
  df['Final_Cek'] = df['Final_Cek'].str.replace('blok', 'blokir')
  df['Final_Cek'] = df['Final_Cek'].str.replace('sulit', 'tidak')
  df['Final_Cek'] = df['Final_Cek'].str.replace('erorr', 'error')
  df['Final_Cek'] = df['Final_Cek'].str.replace('lelet', 'lama')
  df['Final_Cek'] = df['Final_Cek'].str.replace('erur', 'error')
  df['Final_Cek'] = df['Final_Cek'].str.replace('kga', 'tidak')
  df['Final_Cek'] = df['Final_Cek'].str.replace('engengtidak', 'tidak')
  df['Final_Cek'] = df['Final_Cek'].str.replace('tks', 'terima kasih')
  df['Final_Cek'] = df['Final_Cek'].str.replace('ganngguan', 'ganggu')
  df['Final_Cek'] = df['Final_Cek'].str.replace('errornya', 'error')
  df['Final_Cek'] = df['Final_Cek'].str.replace('engtidak', 'tidak')
  df['Final_Cek'] = df['Final_Cek'].str.replace('aplikasix', 'aplikasi')
  df['Final_Cek'] = df['Final_Cek'].str.replace('ss', 'cuplikan')
  df['Final_Cek'] = df['Final_Cek'].str.replace('baharu', 'baru')

  df['Final_Cek'] = df['Final_Cek'].drop_duplicates().apply(split_word)
  df = df.dropna()
  df = df.reset_index(drop=True)

  df['Untokenizing'] = df['Final_Cek'].apply(satu)

  # Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)

  # Loads lexicon positive and negative data
  import requests
  lexicon_positive = dict()
  url = 'https://raw.githubusercontent.com/fawazdhianabil/AKSEL/main/lexicon_positive.csv'
  response =requests.get(url)
  reader = csv.reader(response.text.splitlines())
  for row in reader:
    lexicon_positive[row[0]] = int(row[1])
        

  import requests
  lexicon_negative = dict()
  url = 'https://raw.githubusercontent.com/fawazdhianabil/AKSEL/main/lexicon_negative.csv'
  response =requests.get(url)
  reader = csv.reader(response.text.splitlines())
  for row in reader:
    lexicon_negative[row[0]] = int(row[1])

  # Function to determine sentiment polarity of tweets
  def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score >= 0):
        polarity = 'positive'
    elif (score < 0):
        polarity = 'negative'
    return score, polarity

  hasil = df['Final_Cek'].apply(sentiment_analysis_lexicon_indonesia)
  hasil = list(zip(*hasil))
  df['polarity_score'] = hasil[0]
  df['polarity'] = hasil[1]

  return df

# code untuk streamlit
st.title('Analisis Sentimen')

with st.sidebar :
    selected = option_menu('Menu Analisis',
                           ['Crawling Data Playstore',
                            'Analisis Sentimen by Lexicon'],
                            default_index=0)

if (selected=='Crawling Data Playstore'):
    st.title('Crawling Data Playstore')

    alamat = st.text_input("Masukan Alamat Aplikasi","")
    proses = st.button('Proses Crawling')

    if proses:
        result = reviews_all(alamat,
                             lang='id',
                             country='id',
                             sort=Sort.NEWEST)
        
        st.write('Crawling Data Berhasil!',pd.DataFrame(result))

if (selected=='Analisis Sentimen by Lexicon'):
    st.title('Analisis Sentimen by Lexicon')

    alamat_ = st.text_input("Masukan Alamat Aplikasi","")
    proses_analisis = st.button('Proses Analisis')

    if proses_analisis:
        df = scrap(alamat=alamat_)
        df_n = sentimen(df)
        st.write('Sentimen Analisis Berhasil!',df_n)