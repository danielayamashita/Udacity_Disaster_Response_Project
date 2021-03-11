'''
Date: 10th March 2021
Author: Daniela Yassuda Yamashita
Description: train_classifier.py script. Train a machine learning pipeline using
			Random Forest classifier and Natural Languade processing. This script
			receives as input the `DisasterResponse.db` SQL database and gives as
			output the machine learning pipeline `classifier.pkl`.
'''
# import libraries
import sys
import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
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
from sklearn.metrics import classification_report

nltk.download(['stopwords','wordnet','punkt', 'wordnet', 'averaged_perceptron_tagger'])

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


def load_data(database_filepath):
    """
    Load DAta function

    This function load the dataset from a SQLite database.

    INPUT:
        database_filepath -> SQL file datapath
    OUTPUT:
        X -> Numpy array of Features of the test dataset (disaster messages)
        y -> Numpy array of labels of the test dataset (type of message)
        categories -> list of all names indentifying the type of message
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Database_Disaster_Response', engine)  

    X = df['message'].values
    y = df.iloc[:, 4:].values 
    
    categories = df.columns[4:] # Get the name of categories
    
    #NUM_SAMPLES = 1000
    #X  = X[-NUM_SAMPLES:]
    #y = y[-NUM_SAMPLES:]
    
    print("Shape X: ", X.shape)
    print("Shape y: ", y.shape)
    return X, y, categories


def tokenize(text):
    """
    Tokenize function

    This function clean, tokenize and lemmatize the text message. 

    OUTPUT:
    tokens -> the most important words after processing the raw message.
    """
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
    """
    Build Model function

    This function builds a pipeline and a GridSearchCV model
    to classify Disaster messages into 36 categories.

    OUTPUT:
    model -> GridSearchCV or Scikit Pipeline object
    """
    
    # Define the Machine Learning pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('CountNouns', NumberOfNounsVerbsExtractor())
        ])),

    
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = 1)))
    ])
    
    # Define the parameters to be tuned in the Machine learning model
    parameters = {
        'features__transformer_weights': (
            {'text_pipeline': 1, 'CountNouns': 0.5},
            {'text_pipeline': 0.5, 'CountNouns': 1},
            {'text_pipeline': 0.8, 'CountNouns': 1},
        )
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv 


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalulate Model function

    This function prints the accuracy and the f1 score to evaluate
    the model performance.

    INPUT:
    model -> GridSearchCV or Scikit Pipeline object
    X_test -> Numpy array of Features of the test dataset (disaster messages)
    Y_test -> Numpy array of labels of the test dataset (type of message)
    category_name -> list of all names indentifying the type of message
    """
    Y_pred = model.predict(X_test) # make the prediction based on the developped model 
    
    accuracy = (Y_pred == Y_test).mean() # Calculate the average accuracy of the model prediction
    
    # print the metrics to evaluate the model
    print('Average overall accuracy {0:.2f}%'.format(accuracy*100))
    
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test[:,idx], Y_pred[:,idx]))
        
    print("\nBest Parameters:", model.best_params_)



def save_model(model, model_filepath):
    """
    Save Model function

    This function saves trained model as Pickle file, to be loaded later.

    INPUT:
    model -> GridSearchCV or Scikit Pipeline object
    model_filepath -> destination path to save .pkl    file  """
    pickle.dump(model, open(model_filepath, 'wb'))


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