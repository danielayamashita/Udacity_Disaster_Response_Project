# import libraries
import sys
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix

nltk.download(['stopwords','wordnet','punkt', 'wordnet', 'averaged_perceptron_tagger'])

class NumberOfNounsVerbsExtractor(BaseEstimator, TransformerMixin):
    '''
    This class is to create a Customise transform function 
    to count the number o nouns in a disaster message, creating
    additional features to the model training
    '''
    def __init__(self, tag_choice = 0):
        self.tag = tag_choice
        
    def NumberOfNounsVerbs(self, text):
        sentence_list = nltk.sent_tokenize(text)
        number_nouns = 0
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for i in range(len(pos_tags)):
                word, tag = pos_tags[i]
                if tag in ['NN']:
                    number_nouns += 1 
        return number_nouns

    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.NumberOfNounsVerbs)

    
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Database_Disaster_Response', engine)  

    X = df['message'].values
    y = df.iloc[:, 4:].values 
    
    categories = df.columns #
    return X, y, categories


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('CountNouns', NumberOfNounsVerbsExtractor())
        ])),

    
        ('clf', RandomForestClassifier())
    ])
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2))
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline 


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test) # make the prediction based on the developped model
    
    accuracy = (y_pred == Y_test).mean() #
    #print("Category names: ", category_names)
    print("Accuracy:", accuracy)
    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    
    #print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
     """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(category_names)
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