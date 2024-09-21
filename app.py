import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

st.title("Email/SMS Spam Classifier")

sms = st.text_input("Enter the message")

result = None  # Initialize the result variable

if st.button('Predict'):
    transform_sms = transform_text(sms)
    vector = tfidf.transform([transform_sms])
    result = model.predict(vector)[0]

if result is not None:
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
