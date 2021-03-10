'''
Date: 10th March 2021
Author: Daniela Yassuda Yamashita
Description: train_classifier.py script. Provides a intuitive user interface
			to process disaster messages easially using the machine learning
			pipeline in backend.
'''

# Import libraries
import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
app = Flask(__name__)
nltk.download(['stopwords','wordnet','punkt', 'wordnet', 'averaged_perceptron_tagger'])

#-------------------------------------------------------------------------
class NumberOfNounsVerbsExtractor(BaseEstimator, TransformerMixin):
    '''
    This class is to create a Customise transform function 
    to count the number o nouns in a disaster message, creating
    additional features to the model training
    '''
        
    def NumberOfNouns(self, text):
        """
        NumberOfNouns function

        This function counters the number of nouns in a 
        disaster message.
        
        INPUT:
        text -> string containing the entire (raw) disaster
                message
        OUTPUT:
        number_nouns -> Number of nouns in a disaster message
        """
        
        # Separate the Message into sentences
        sentence_list = nltk.sent_tokenize(text) 
        
        number_nouns = 0 # Counter of nouns
        for sentence in sentence_list:
            # Get the tags of each word in the sentence
            pos_tags = nltk.pos_tag(tokenize(sentence)) 
            
            # Count the number of noums in a sentence 
            for i in range(len(pos_tags)):
                word, tag = pos_tags[i]
                if tag in ['NN']:
                    number_nouns += 1 
                    
        return number_nouns

    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.NumberOfNouns)
        
        return pd.DataFrame(X_tagged)
#-------------------------------------------------------------------------

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Database_Disaster_Response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()