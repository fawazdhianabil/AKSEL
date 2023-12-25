import streamlit as st
from streamlit_option_menu import option_menu

from fpdf import FPDF
from PIL import Image
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from io import BytesIO
import requests

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
import unicodedata

# warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import nltk
from nltk.corpus import words
nltk.download('words')

from wordcloud import WordCloud

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
    result1 = result[['userName','content','score','at','reviewCreatedVersion','appVersion']]
    return result, result1

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

def jam(df,jdl):
  df = df
  df['at'] = pd.to_datetime(df['at'], errors='coerce')
  jam1 = df['at'].dt.round('H')
  jam1 = jam1.dt.strftime('%H:%M:%S')
  df['jam'] = jam1
  fig, ax = plt.subplots(figsize = (20, 6))
  x_values = jam1.value_counts().sort_index().index
  y_values = jam1.value_counts().sort_index()
  sns.lineplot(ax = ax, x = x_values, y = y_values)
  ax.set_title(f'Banyak Review {jdl} \n (Berdasarkan Jam)', fontsize = 18)
  ax.set_xlabel('Jam')
  ax.set_xticks(x_values)
  ax.set_xticklabels(x_values, rotation = 45)
  ax.set_ylabel('Frekuensi')
  plt.grid()
  st.pyplot(fig)
  fig.savefig('Jam.jpg')

def bulan(df,jdl):
  df = df
  bulan1 = df['at'].dt.round('30D')
  bulan1 = bulan1.dt.strftime('%Y-%m-%d')
  df['bulan'] = bulan1
  fig, ax = plt.subplots(figsize = (20, 6))
  x_values = bulan1.value_counts().sort_index().index
  y_values = bulan1.value_counts().sort_index()
  sns.lineplot(ax = ax, x = x_values, y = y_values)
  ax.set_title(f'Banyak Review pada {jdl} \n (Berdasarkan Bulan)', fontsize = 18)
  ax.set_xlabel('Bulan')
  ax.set_xticks(x_values)
  ax.set_xticklabels(x_values, rotation = 45)
  ax.set_ylabel('Frekuensi')
  plt.grid()
  st.pyplot(fig)
  fig.savefig('Bulan.jpg')

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
    
def apk(df):
    values = df.appVersion.value_counts(ascending=True).keys().tolist()
    counts = df.appVersion.value_counts(ascending=True).tolist()

    versi_x = values
    versi_y = counts

    fig = plt.figure(figsize = (12, 10))
    plt.bar(versi_x, versi_y,color='lightblue')
    plt.xlabel("Versi", fontweight ='bold')
    plt.ylabel("Frekuensi", fontweight ='bold')
    plt.title("Versi Aplikasi yang Sering Menerima Review", fontweight ='bold', fontsize = 14)

    plt.xticks(versi_x, rotation = 30)
    st.pyplot(fig)
    fig.savefig('apk.jpg')
    
def pos(kata_positif,jdl):
    # SENTIMEN positif
    
    positif_kata = kata_positif.value_counts().nlargest(20)
     
    positif_x = positif_kata.index
    positif_y = positif_kata.values

    fig = plt.figure(figsize = (12, 10))
    plt.bar(positif_x, positif_y,color='lightgreen')
    plt.xlabel("Kata", fontweight ='bold')
    plt.ylabel("Frekuensi", fontweight ='bold')
    plt.title(f"Kata pada Sentimen Positive {jdl}", fontweight ='bold', fontsize = 14)

    plt.xticks(positif_x, rotation = 30)
    fig.savefig('Positive.jpg')
    st.pyplot(fig)
    fig.savefig('Positive.jpg')

def neg(kata_negatif,jdl):
    # SENTIMEN negatif
    
    negatif_kata = kata_negatif.value_counts().nlargest(20)
     
    negatif_x = negatif_kata.index
    negatif_y = negatif_kata.values

    fig = plt.figure(figsize = (12, 10))
    plt.bar(negatif_x, negatif_y,color='lightcoral')
    plt.xlabel("Kata", fontweight ='bold')
    plt.ylabel("Frekuensi", fontweight ='bold')
    plt.title(f"Kata pada Sentimen Negative {jdl}", fontweight ='bold', fontsize = 14)

    plt.xticks(negatif_x, rotation = 30)
    fig.savefig('negative.jpg')
    st.pyplot(fig)
    fig.savefig('negative.jpg')

def wordcloud(df,jdl):
    positive_review = df[df['polarity'] == 'positive']
    positive_words = positive_review['Untokenizing'].apply(split_word)
    
    fig, ax = plt.subplots(1,2, figsize = (20, 16))
    list_words_postive=''
    for row_word in positive_words:
        for word in row_word:
            list_words_postive += ' '+(word)
    wordcloud_positive = WordCloud(width = 1800, height = 1500, background_color = 'black', colormap = 'Greens'
                               , min_font_size = 12).generate(list_words_postive)
    ax[0].set_title(f'Word Cloud dari Kata Positive {jdl}', fontsize = 14)
    ax[0].grid(False)
    ax[0].imshow((wordcloud_positive))
    fig.tight_layout(pad=0)
    ax[0].axis('off')

    negative_review = df[df['polarity'] == 'negative']
    negative_words = negative_review['Untokenizing'].apply(split_word)

    list_words_negative=''
    for row_word in negative_words:
        for word in row_word:
            list_words_negative += ' '+(word)
    wordcloud_negative = WordCloud(width = 1800, height = 1500, background_color = 'black', colormap = 'Reds'
                               , min_font_size = 12).generate(list_words_negative)
    ax[1].set_title(f'Word Cloud dari Kata Negative {jdl}', fontsize = 14)
    ax[1].grid(False)
    ax[1].imshow((wordcloud_negative))
    fig.tight_layout(pad=0)
    ax[1].axis('off')
    st.pyplot(fig)

def wc_positive(df,jdl):
    positive_review = df[df['polarity'] == 'positive']
    positive_words = positive_review['Untokenizing'].apply(split_word)

    fig, ax = plt.subplots(figsize = (15, 10))
    list_words_postive=''
    for row_word in positive_words:
        for word in row_word:
            list_words_postive += ' '+(word)
    wordcloud_positive = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Greens'
                               , min_font_size = 10).generate(list_words_postive)
    ax.set_title(f'Word Cloud dari Kata Positive {jdl}', fontsize = 14)
    ax.grid(False)
    ax.imshow((wordcloud_positive))
    fig.tight_layout(pad=0)
    wordcloud_positive.to_file('wc_positive.jpg')
    ax.axis('off')
    fig.savefig('wc_positive.jpg')

def wc_negative(df,jdl):
    negative_review = df[df['polarity'] == 'negative']
    negative_words = negative_review['Untokenizing'].apply(split_word)

    fig, ax = plt.subplots(figsize = (15, 10))
    list_words_negative=''
    for row_word in negative_words:
        for word in row_word:
            list_words_negative += ' '+(word)
    wordcloud_negative = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Reds'
                               , min_font_size = 10).generate(list_words_negative)
    ax.set_title(f'Word Cloud dari Kata Negative {jdl}', fontsize = 14)
    ax.grid(False)
    ax.imshow((wordcloud_negative))
    fig.tight_layout(pad=0)
    wordcloud_negative.to_file('wc_negative.jpg')
    ax.axis('off')
    fig.savefig('wc_negative.jpg')

def sentimen(df,jdl):
    df_n = sentimen(df)
    sizes = [count for count in df_n['polarity'].value_counts()]
    labels = list(df_n['polarity'].value_counts().index)
    j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
    hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
    wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
    st.success('Sentimen Analisis Berhasil!')
    st.success(f'Data Tanggal {hari} Pukul {wkt} WITA')
    st.write(df_n)
    st.write('='*88)
    st.write('Ringkasan Data :')
    st.write('Data Sebelum Text Preprocessing :',df.shape[0])
    st.write('Data Sesudah Text Preprocessing :',df_n.shape[0])
    st.write('Jumlah Sentiment Negative :',sizes[0])
    st.write('Jumlah Sentiment Positive :',sizes[1])

    fig, ax = plt.subplots(figsize = (6, 6))
    explode = (0.1, 0)
    colors = ['lightcoral', 'lightgreen']
    ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
    ax.set_title(f'Sentiment Polarity Pada Data {jdl}', fontsize = 16, pad = 20)
    st.write('='*88)
    st.pyplot(fig)
    fig.savefig('Polarity.jpg')
    st.write('='*88)
    apk(df)
    st.write('='*88)
    wordcloud(df_n,jdl)
    st.write('='*88)
    kata_positif = pd.Series(" ".join(df_n[df_n["polarity"] == 'positive']["Untokenizing"].astype("str")).split())
    kata_negatif = pd.Series(" ".join(df_n[df_n["polarity"] == 'negative']["Untokenizing"].astype("str")).split())
    pos(kata_positif,jdl)
    st.write('='*88)
    neg(kata_negatif,jdl)
    st.write('='*88)
    jam(df,jdl)
    st.write('='*88)
    bulan(df,jdl)

    j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
    hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
    wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
    WIDTH =210
    HEIGHT = 297
    pdf = FPDF()
    #hal pertama
    wc_positive(df_n,jdl)
    wc_negative(df_n,jdl)
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.ln(5)
    pdf.write(5,f'Analisis Sentimen {jdl}')
    pdf.ln(7)
    pdf.set_font('Arial','B',12)
    pdf.write(4,f'Tanggal : {hari}, Pukul : {wkt} WITA')
    pdf.image("apk.jpg",0,30,WIDTH/2-5)
    pdf.image("Polarity.jpg",WIDTH/2+5,30,WIDTH/2-5)
    pdf.image("wc_positive.jpg",5,120,WIDTH/2-10,HEIGHT/5+5)
    pdf.image("wc_negative.jpg",WIDTH/2+5,120,WIDTH/2-10,HEIGHT/5+5)
    pdf.image("Positive.jpg",0,190,WIDTH/2-5,HEIGHT/5+5)
    pdf.image("negative.jpg",WIDTH/2+5,190,WIDTH/2-5,HEIGHT/5+5)

    #hal kedua
    pdf.add_page()
    pdf.set_font('Arial','B',16)   
    pdf.ln(5)
    pdf.write(5,f'Ringkasan Statistik {jdl}')
    pdf.ln(7)
    pdf.set_font('Arial','B',12)
    pdf.write(4,f'Tanggal : {hari}, Pukul : {wkt} WITA')
    pdf.image("Jam.jpg",0,30,WIDTH/2-5,HEIGHT/5+5)
    pdf.image("Bulan.jpg",WIDTH/2+5,30,WIDTH/2-5,HEIGHT/5+5)

    #ringkasan
    pdf.set_font('Arial','B',12)
    pdf.ln(80)
    pdf.write(5,'='*76)
    pdf.ln(10)
    pdf.write(5,'Ringkasan Data :')
    pdf.ln(10)
    pdf.write(1,f'Data Sebelum Text Preprocessing :{df.shape[0]}')
    pdf.ln(6)
    pdf.write(1,f'Data Sesudah Text Preprocessing :{df_n.shape[0]}')
    pdf.ln(6)
    pdf.write(1,f'Jumlah Sentiment Negative :{sizes[0]}')
    pdf.ln(6)
    pdf.write(1,f'Jumlah Sentiment Positive :{sizes[1]}')

    pdf.output(f'Sentimen Analisis {jdl}.pdf','F')
    with open(f"Sentimen Analisis {jdl}.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Download Report",data=PDFbyte,
                       file_name=f"Sentimen Analisis {jdl}.pdf",
                       mime='application/octet-stream')


# code untuk streamlit
st.markdown("<h1 style='text-align: center; color: black;'>Web App <span style='color: red;'>Sentimen</span>", unsafe_allow_html=True)
    

with st.sidebar :
    #st.image(img,width = 290)
    selected = option_menu('Main Menu',
                           ['Scraping Data Playstore',
                            'Analisis Sentimen (Aplikasi)',
                            'Analisis Sentimen (Produk)'],
                            default_index=0)
#scraping playstore
if (selected=='Scraping Data Playstore'):
    st.title('Scraping Data Playstore')
    al = st.selectbox('Silahkan Pilih Aplikasi',('AKSEL',
                                                 'Merchant Mobile (QRIS)',
                                                 'IBB Mobile',
                                                 'Brimo',
                                                 'Livin By Mandiri',
                                                 'BNI Mobile',
                                                 'BTN Mobile',
                                                 'Perusahaan Lainnya'),
                          index=None,placeholder='Pilih')
    if al == 'AKSEL':
        alamat = 'id.co.bankkalsel.mobile_banking'
        jdl = 'AKSEL'
    elif al == 'Merchant Mobile (QRIS)':
        alamat = 'com.dwidasa.kalsel.mbqris.android'
        jdl = 'Merchant Mobile (QRIS)'
    elif al == 'IBB Mobile':
        alamat = 'id.co.bankkalsel.mobileibb'
        jdl = 'IBB Mobile'
    elif al == 'Brimo':
        alamat = 'id.co.bri.brimo'
        jdl = 'Brimo'
    elif al == 'Livin By Mandiri':
        alamat = 'id.bmri.livin'
        jdl = 'Livin By Mandiri'
    elif al == 'BNI Mobile':
        alamat = 'src.com.bni'
        jdl = 'BNI Mobile'
    elif al == 'BTN Mobile':
        alamat = 'id.co.btn.mobilebanking.android'
        jdl = 'BTN Mobile'
    elif al == 'Perusahaan Lainnya':
        alamat = st.text_input('Masukkan URL Perusahaan',key=0)
        jdl = st.text_input('Masukkan Nama Aplikasi Perusahaan',key=1)
    proses = st.button('Proses Scraping',key='gp')
    if proses:
        result1 = scrap(alamat=alamat)
        result = result1[0]
        if result.shape[0] > 0:
            st.success(f'Scraping {result.shape[0]} Data Berhasil!')
            st.write(pd.DataFrame(result))
            st.download_button(label='Download Data Mentah', data = pd.DataFrame(result).to_csv(index=False), file_name='Data Mentah.csv')
        else:
            st.error('Data Ulasan Tidak Ada',icon='🚨')
            st.error('Hal ini Disebabkan Belum Ada Ulasan')

#================================ANALISIS=========================
if (selected=='Analisis Sentimen (Aplikasi)'):
    st.title('Analisis Sentimen (Aplikasi)')
    sc = st.selectbox('Silahkan Pilih Sumber Data',('Google Playstore',
                                                 'Upload Data'),
                          index=None,placeholder='Pilih')
    if sc == 'Upload Data':
        data_file = st.file_uploader("Upload CSV file",type=["csv"])
        if data_file is not None:
            data_file_raw = pd.read_csv(data_file)
            jdl = st.text_input('Masukkan Nama Aplikasi')
        else :
            st.write('Silahkan Upload Data')
    elif sc == 'Google Playstore':
        al = st.selectbox('Silahkan Pilih Aplikasi',('AKSEL',
                                                     'Merchant Mobile (QRIS)',
                                                     'IBB Mobile',
                                                     'Brimo',
                                                     'Livin By Mandiri',
                                                     'BNI Mobile',
                                                     'BTN Mobile',
                                                     'Perusahaan Lainnya'),
                          index=None,placeholder='Pilih')
    
    pil =  st.selectbox('Simpan Sebagai...',('PDF','Tidak Menyimpan Report'),
                          index=None,placeholder='Pilih')
    
    if pil == 'PDF' and sc == 'Google Playstore':
        if al == 'AKSEL':
            alamat = 'id.co.bankkalsel.mobile_banking'
            jdl = 'AKSEL'
        elif al == 'Merchant Mobile (QRIS)':
            alamat = 'com.dwidasa.kalsel.mbqris.android'
            jdl = 'Merchant Mobile (QRIS)'
        elif al == 'IBB Mobile':
            alamat = 'id.co.bankkalsel.mobileibb'
            jdl = 'IBB Mobile'
        elif al == 'Brimo':
            alamat = 'id.co.bri.brimo'
            jdl = 'Brimo'
        elif al == 'Livin By Mandiri':
            alamat = 'id.bmri.livin'
            jdl = 'Livin By Mandiri'
        elif al == 'BNI Mobile':
            alamat = 'src.com.bni'
            jdl = 'BNI Mobile'
        elif al == 'BTN Mobile':
            alamat = 'id.co.btn.mobilebanking.android'
            jdl = 'BTN Mobile'
        elif al == 'Perusahaan Lainnya':
            alamat = st.text_input('Masukkan URL Perusahaan',key=0)
            jdl = st.text_input('Masukkan Nama Aplikasi Perusahaan',key=1)
        proses_analisis = st.button('Proses Analisis')
        if proses_analisis:
            try:
                df1 = scrap(alamat=alamat)
                df = df1[1]
                jdl = jdl
                sentimen(df,jdl)

            except Exception as e:
                st.write(e)
                st.error('Data Ulasan Tidak Ada',icon='🚨')
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.error(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.error('Hal ini Disebabkan Belum Ada Ulasan')
                
    elif pil == 'PDF' and sc == 'Upload Data':
        proses_analisis = st.button('Proses Analisis')
        if proses_analisis:
            try:
                df=data_file_raw
                df_n = sentimen(data_file_raw[['userName','content','score','at','reviewCreatedVersion','appVersion']])
                sizes = [count for count in df_n['polarity'].value_counts()]
                labels = list(df_n['polarity'].value_counts().index)
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.success('Sentimen Analisis Berhasil!')
                st.success(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.write(df_n)
                st.write('='*88)
                st.write('Ringkasan Data :')
                st.write('Data Sebelum Text Preprocessing :',df.shape[0])
                st.write('Data Sesudah Text Preprocessing :',df_n.shape[0])
                st.write('Jumlah Sentiment Negative :',sizes[0])
                st.write('Jumlah Sentiment Positive :',sizes[1])

                fig, ax = plt.subplots(figsize = (6, 6))
                explode = (0.1, 0)
                colors = ['lightcoral', 'lightgreen']
                ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
                ax.set_title(f'Sentiment Polarity Pada Data {jdl}', fontsize = 16, pad = 20)
                st.write('='*88)
                st.pyplot(fig)
                fig.savefig('Polarity.jpg')
                st.write('='*88)
                apk(df)
                st.write('='*88)
                wordcloud(df_n,jdl)
                st.write('='*88)
                kata_positif = pd.Series(" ".join(df_n[df_n["polarity"] == 'positive']["Untokenizing"].astype("str")).split())
                kata_negatif = pd.Series(" ".join(df_n[df_n["polarity"] == 'negative']["Untokenizing"].astype("str")).split())
                pos(kata_positif,jdl)
                st.write('='*88)
                neg(kata_negatif,jdl)
                st.write('='*88)
                jam(df,jdl)
                st.write('='*88)
                bulan(df,jdl)

                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                WIDTH =210
                HEIGHT = 297
                pdf = FPDF()
                #hal pertama
                wc_positive(df_n,jdl)
                wc_negative(df_n,jdl)
                pdf.add_page()
                pdf.set_font('Arial','B',16)
                pdf.ln(5)
                pdf.write(5,f'Analisis Sentimen {jdl}')
                pdf.ln(7)
                pdf.set_font('Arial','B',12)
                pdf.write(4,f'Tanggal : {hari}, Pukul : {wkt} WITA')
                pdf.image("apk.jpg",0,30,WIDTH/2-5)
                pdf.image("Polarity.jpg",WIDTH/2+5,30,WIDTH/2-5)
                pdf.image("wc_positive.jpg",5,120,WIDTH/2-10,HEIGHT/5+5)
                pdf.image("wc_negative.jpg",WIDTH/2+5,120,WIDTH/2-10,HEIGHT/5+5)
                pdf.image("Positive.jpg",0,190,WIDTH/2-5,HEIGHT/5+5)
                pdf.image("negative.jpg",WIDTH/2+5,190,WIDTH/2-5,HEIGHT/5+5)

                #hal kedua
                pdf.add_page()
                pdf.set_font('Arial','B',16)   
                pdf.ln(5)
                pdf.write(5,f'Ringkasan Statistik {jdl}')
                pdf.ln(7)
                pdf.set_font('Arial','B',12)
                pdf.write(4,f'Tanggal : {hari}, Pukul : {wkt} WITA')
                pdf.image("Jam.jpg",0,30,WIDTH/2-5,HEIGHT/5+5)
                pdf.image("Bulan.jpg",WIDTH/2+5,30,WIDTH/2-5,HEIGHT/5+5)

                #ringkasan
                pdf.set_font('Arial','B',12)
                pdf.ln(80)
                pdf.write(5,'='*76)
                pdf.ln(10)
                pdf.write(5,'Ringkasan Data :')
                pdf.ln(10)
                pdf.write(1,f'Data Sebelum Text Preprocessing :{df.shape[0]}')
                pdf.ln(6)
                pdf.write(1,f'Data Sesudah Text Preprocessing :{df_n.shape[0]}')
                pdf.ln(6)
                pdf.write(1,f'Jumlah Sentiment Negative :{sizes[0]}')
                pdf.ln(6)
                pdf.write(1,f'Jumlah Sentiment Positive :{sizes[1]}')

                pdf.output(f'Sentimen Analisis {jdl}.pdf','F')
                with open(f"Sentimen Analisis {jdl}.pdf", "rb") as pdf_file:
                    PDFbyte = pdf_file.read()

                st.download_button(label="Download Report",
                                   data=PDFbyte,
                                   file_name=f"Sentimen Analisis {jdl}.pdf",
                                   mime='application/octet-stream')
            except Exception as e:
                st.error('Data Ulasan Tidak Ada',icon='🚨')
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.error(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.error('Hal ini Disebabkan Belum Ada Ulasan')

    
    if pil == 'Tidak Menyimpan Report' and sc == 'Google Playstore':
        if al == 'AKSEL':
            alamat = 'id.co.bankkalsel.mobile_banking'
            jdl = 'AKSEL'
        elif al == 'Merchant Mobile (QRIS)':
            alamat = 'com.dwidasa.kalsel.mbqris.android'
            jdl = 'Merchant Mobile (QRIS)'
        elif al == 'IBB Mobile':
            alamat = 'id.co.bankkalsel.mobileibb'
            jdl = 'IBB Mobile'
        elif al == 'Brimo':
            alamat = 'id.co.bri.brimo'
            jdl = 'Brimo'
        elif al == 'Livin By Mandiri':
            alamat = 'id.bmri.livin'
            jdl = 'Livin By Mandiri'
        elif al == 'BNI Mobile':
            alamat = 'src.com.bni'
            jdl = 'BNI Mobile'
        elif al == 'BTN Mobile':
            alamat = 'id.co.btn.mobilebanking.android'
            jdl = 'BTN Mobile'
        elif al == 'Perusahaan Lainnya':
            alamat = st.text_input('Masukkan URL Perusahaan',key=0)
            jdl = st.text_input('Masukkan Nama Aplikasi Perusahaan',key=1)
        proses_analisis = st.button('Proses Analisis')
        if proses_analisis:
            try:
                df1 = scrap(alamat=alamat)
                df = df1[1]
                df_n = sentimen(df)
                sizes = [count for count in df_n['polarity'].value_counts()]
                labels = list(df_n['polarity'].value_counts().index)
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.success('Sentimen Analisis Berhasil!')
                st.success(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.write(df_n)
                st.write('='*88)
                st.write('Ringkasan Data :')
                st.write('Data Sebelum Text Preprocessing :',df.shape[0])
                st.write('Data Sesudah Text Preprocessing :',df_n.shape[0])
                st.write('Jumlah Sentiment Negative :',sizes[0])
                st.write('Jumlah Sentiment Positive :',sizes[1])

                fig, ax = plt.subplots(figsize = (6, 6))
                explode = (0.1, 0)
                colors = ['lightcoral', 'lightgreen']
                ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
                ax.set_title(f'Sentiment Polarity Pada Data {jdl}', fontsize = 16, pad = 20)
                st.write('='*88)
                st.pyplot(fig)
                fig.savefig('Polarity.jpg')
                st.write('='*88)
                apk(df)
                st.write('='*88)
                wordcloud(df_n,jdl)
                st.write('='*88)
                kata_positif = pd.Series(" ".join(df_n[df_n["polarity"] == 'positive']["Untokenizing"].astype("str")).split())
                kata_negatif = pd.Series(" ".join(df_n[df_n["polarity"] == 'negative']["Untokenizing"].astype("str")).split())
                pos(kata_positif,jdl)
                st.write('='*88)
                neg(kata_negatif,jdl)
                st.write('='*88)
                jam(df,jdl)
                st.write('='*88)
                bulan(df,jdl)

            except Exception as e:
                st.error('Data Ulasan Tidak Ada',icon='🚨')
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.error(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.error('Hal ini Disebabkan Belum Ada Ulasan')
    
    elif pil == 'Tidak Menyimpan Report' and sc == 'Upload Data':
        proses_analisis = st.button('Proses Analisis')
        if proses_analisis:
            try:
                df = data_file_raw
                df_n = sentimen(df[['userName','content','score','at','reviewCreatedVersion','appVersion']])
                sizes = [count for count in df_n['polarity'].value_counts()]
                labels = list(df_n['polarity'].value_counts().index)
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.success('Sentimen Analisis Berhasil!')
                st.success(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.write(df_n)
                st.write('='*88)
                st.write('Ringkasan Data :')
                st.write('Data Sebelum Text Preprocessing :',df.shape[0])
                st.write('Data Sesudah Text Preprocessing :',df_n.shape[0])
                st.write('Jumlah Sentiment Negative :',sizes[0])
                st.write('Jumlah Sentiment Positive :',sizes[1])

                fig, ax = plt.subplots(figsize = (6, 6))
                explode = (0.1, 0)
                colors = ['lightcoral', 'lightgreen']
                ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
                ax.set_title(f'Sentiment Polarity Pada Data {jdl}', fontsize = 16, pad = 20)
                st.write('='*88)
                st.pyplot(fig)
                fig.savefig('Polarity.jpg')
                st.write('='*88)
                apk(df)
                st.write('='*88)
                wordcloud(df_n,jdl)
                st.write('='*88)
                kata_positif = pd.Series(" ".join(df_n[df_n["polarity"] == 'positive']["Untokenizing"].astype("str")).split())
                kata_negatif = pd.Series(" ".join(df_n[df_n["polarity"] == 'negative']["Untokenizing"].astype("str")).split())
                pos(kata_positif,jdl)
                st.write('='*88)
                neg(kata_negatif,jdl)
                st.write('='*88)
                jam(df,jdl)
                st.write('='*88)
                bulan(df,jdl)

            except :
                st.error('Data Ulasan Tidak Ada',icon='🚨')
                j = int(datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%H"))
                hari = datetime.now(ZoneInfo('Asia/Jakarta')).strftime("%d/%m/%Y")
                wkt = datetime.now(ZoneInfo('Asia/Jakarta')).strftime(f"{j+1}:%M:%S")
                st.error(f'Data Tanggal {hari} Pukul {wkt} WITA')
                st.error('Hal ini Disebabkan Belum Ada Ulasan')
