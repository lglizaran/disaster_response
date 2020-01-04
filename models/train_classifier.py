# import libraries
import pandas as pd
import numpy as np
import re
import sys
import pickle

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    # engine = create_engine('sqlite:///../data/DisasterResponse.db')
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    # print (engine.table_names())

    df = pd.read_sql('DisasterResponse', engine)

    X = df.message.values # list of all messages
    y = df[df.columns[4:]].values # matrix of all 1/0 values of categories
    category_names = df.columns[4:]

    return X, y, category_names


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


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('puct_perc', PunctPercExtractor())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    # pipeline = Pipeline([
    #         ('vect', CountVectorizer(tokenizer=tokenize)),
    #         ('tfidf', TfidfTransformer()),
    #         ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    #         ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.75, 1.0),
        # 'vect__max_features': (None, 5000)
        #'clf__n_estimators': [50, 100, 200],
        #'clf__min_samples_split': [2, 3, 4]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    # predict y values
    y_pred = model.predict(X_test)
    # print the result of the prediction for each category
    for i in range(36):
        print(category_names[i],'______________________')
        print(classification_report(y_test[:,i], y_pred[:,i]))

        labels = np.unique(y_pred[:,i])
        confusion_mat = confusion_matrix(y_test[:,i], y_pred[:,i], labels=labels)
        accuracy = (y_pred[:,i] == y_test[:,i]).mean()

        print("Labels:", labels)
        print("Confusion Matrix:\n(real\predicted)\n", confusion_mat)
        print("Accuracy:", accuracy,'\n\n')
    pass


def save_model(model, model_filepath):
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
