import joblib
import streamlit as st
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 
import aspose.words as aw
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
#from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import nltk
nltk.download('stopwords')
nltk.download('punkt')


st.markdown('''
<style>
.stApp {
    
    background-color:#red;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#FFFFFF;\
    border-style: false;\
    border-width: 2px;\
    color:White;\
    font-size:14px;\
    background-color:#red;\
    text-align:center;\
    font-family: Source Sans Pro;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: BLUE;
    
}
.st-b7 {
    color: #C9C9C9;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('!', '',text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('⇨', '',text)
    text = re.sub(':', '',text)
    text = re.sub('•', '',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

svm_pkl=joblib.load(open("./SVM_model.pkl","rb"))
word_vectorizer=joblib.load(open("./word_vectorizer.pkl","rb"))

#st.sidebar.width=1000
st.sidebar.title("Resume Classifier")
uploaded_resume = st.sidebar.file_uploader("Upload your resume Here :", type={"doc", "docx","pdf"})



if st.sidebar.button(label="Predict"):
    if uploaded_resume is not None:
       doc = aw.Document(uploaded_resume)
       doc_txt = doc.get_text().strip()
       #st.write(doc_txt)
       doc_txt = clean_text(doc_txt)
       #text = st.write(doc_txt)
       
       tfidfvector = word_vectorizer.transform([doc_txt])
       category = svm_pkl.predict(tfidfvector)[0]
       if category == 0:
           st.sidebar.write("Predicted Category:","Peoplesoft Resume")
       elif category == 1:
               st.sidebar.write("Predicted Category:","React JS Developer Resume")    
       elif category == 2:
                   st.sidebar.write("Predicted Category:","SQL Developer Resume")   
       elif category == 3:
                       st.sidebar.write("Predicted Category:","Workday Resume")  
       else:
           st.sidebar.write("Provided Resume is not from the mentioned categoty")  
           
                           
       
    # WORDCLOUD 
       #st.title("WordCloud")   
       #wc = WordCloud().generate(doc_txt)
       #fig1 = plt.figure(figsize=(10,10))
       #plt.imshow(wc, interpolation='bilinear')
       #plt.axis("off")
       #st.pyplot(fig1)  
    
      # Generate word cloud
       st.subheader("WordCloud")
       wordcloud = WordCloud(stopwords=STOPWORDS,
                            background_color='black',
                            width=1200,
                            height=900
                           ).generate(doc_txt)

      # Display the word cloud
       st.image(wordcloud.to_array(), use_column_width=True)
    
    if uploaded_resume is not None:
            st.subheader("Bar Graph of Top 10 words")
            # Filtering the nouns and verbs only
    
            nlp = spacy.load("en_core_web_sm")
            one_block=doc_txt
            doc_block=nlp(one_block)
            nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
            print(nouns_verbs[100:200])
    
            #Counting tokens again
    
            cv = CountVectorizer()
    
            X = cv.fit_transform(nouns_verbs)
            sum_words = X.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
            words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
            wf_df = pd.DataFrame(words_freq)
            wf_df.columns = ['word', 'count']
            #wf_df[0:10]
    
            st.bar_chart(data=wf_df[0:10],x='word',y='count',use_container_width=True,height=500)
            
    
            # PIE CHART
            
            st.subheader("Pie Chart of Top 10 words")
            # Create the pie chart using plotly.express
            fig2 = px.pie(wf_df[0:10], values='count', names='word',height=500)
    
            # Display the pie chart using Streamlit
            st.plotly_chart(fig2)