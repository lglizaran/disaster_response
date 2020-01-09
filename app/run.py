import pandas as pd
from flask import Flask
from flask import render_template

import plotly.graph_objs as go
import plotly, json

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# from flask import Flask
# from flask import render_template, request, jsonify
# from plotly.graph_objs import Bar
# import plotly.graph_objs as go
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
genre_counts = df.groupby('genre').count()['message'].values
genre_names = df.groupby('genre').count()['message'].index.values

# load model
model = joblib.load("../models/classifier.pkl")

# create the plotly visualizations
graphs = []
graph_one = go.Figure([go.Bar(
    x=genre_names,
    y=genre_counts,
    marker_color='rgb(55, 83, 109)'
    )
])
graph_one.update_layout(
    title = 'Distribution of Message Genres',
    xaxis=dict(
        title = 'Genre',
        tickfont=dict(family='Helvetica', size=14, color='#7f7f7f')
        ),
    yaxis = dict(title = 'Count'),
    plot_bgcolor = 'rgb(255,255,255)',
    font=dict(family='Helvetica', size=14, color='#7f7f7f')
)
graphs.append(graph_one)

# second graph
x_val = df[df.columns[4:]].sum().sort_values(ascending=False)[:10].index
y_val = df[df.columns[4:]].sum().sort_values(ascending=False)[:10].values

graph_two = go.Figure([go.Bar(
    x=x_val,
    y=y_val,
    marker_color='rgb(55, 83, 109)'
    )
])
graph_two.update_layout(
    title = 'Distribution of Message classification',
    xaxis=dict(
        title = 'Response',
        tickfont=dict(family='Helvetica', size=14, color='#7f7f7f')
        ),
    yaxis = dict(title = 'Count'),
    plot_bgcolor = 'rgb(255,255,255)',
    font=dict(family='Helvetica', size=14, color='#7f7f7f')
)
graphs.append(graph_two)

# third chart
x_val = df.message.str.len().values
# y_val = df[df.columns[4:]].mean().sort_values(ascending=False)[:10].values
graph_three = go.Figure([go.Histogram(
    x=x_val,
    histnorm='probability',
    # nbins=20,
    xbins=dict(start=0,end=800,size=5),
    marker_color='rgb(55, 83, 109)')]
)
# graph_three = go.Figure([go.Bar(
#     x=x_val,
#     y=y_val,
#     marker_color='rgb(55, 83, 109)'
#     )
# ])
graph_three.update_layout(
    title = 'Histogram of text length',
    xaxis=dict(
        # title = 'Response',
        tickfont=dict(family='Helvetica', size=14, color='#7f7f7f')
        ),
    # yaxis = dict(title = 'Percentage of trues'),
    plot_bgcolor = 'rgb(255,255,255)',
    font=dict(family='Helvetica', size=14, color='#7f7f7f')
)
graphs.append(graph_three)

# encode plotly graphs in JSON
ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],
    #
    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    # ]

    # # Create arrays x,y
    # data = {'name':['Tom', 'nick', 'krish', 'jack'], 'age':[20, 21, 19, 18]}
    # df1 = pd.DataFrame(data)
    # x_val = df1.name.values.tolist()
    # y_val = df1.age.values.tolist()
    #
    # # create the plotly visualizations
    # graph_one = [go.Scatter(
    #     x = x_val,
    #     y = y_val,
    #     mode='lines',
    #     name='chart'
    # )]
    # layout_one = dict(title = 'title', xaxis=dict(title = 'x_label'), yaxis = dict(title = 'y_label'))
    # graphs = []
    # graphs.append(dict(data=graph_one,layout=layout_one))
    #
    # # encode plotly graphs in JSON
    # ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    # graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()
