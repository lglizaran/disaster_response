import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # remove urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # remove punctuation and make lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize words
    tokens = word_tokenize(text)
    # remove stop words
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return tokens

class PunctPercExtractor(BaseEstimator, TransformerMixin):

    def punct(self, text):
        punct = len(re.findall(r"[^a-zA-Z0-9]", text)) / len(text)
        return punct

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_punct = pd.Series(X).apply(self.punct)
        return pd.DataFrame(X_punct)


engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql('DisasterResponse', engine)

model = joblib.load("../models/classifier.pkl")


for message in df.loc[20:60,:].message.tolist():
    print(message)
    lassification_labels = model.predict([message])[0]
    classification_results = dict(zip(df.columns[4:], lassification_labels))
    print(classification_results)
