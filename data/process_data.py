# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer')
    
    return df

def clean_data(df):

    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # Rename de DataFrame columns according to the categories name
    # specified in the firs row of 'categories' datagrame
    row = categories.loc[0,:] # Select the first row of the categories dataframe
    category_colnames = row.apply(lambda x: x.split('-')[0])# use this row to extract a list of new column names for categories.
    categories.columns = category_colnames # rename the columns of `categories`
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1]).astype(int)
    
    # Drop the original categories column from `df`
    df = df.drop(['categories'],axis = 1)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis = 1)
    
    # Remove NaN values of all columns except 'original'
    df = df.dropna(how = 'any',subset = df.columns[df.columns != 'original'],axis = 0)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['message'])
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///Database_Disaster_Response.db')
    df.to_sql('Database_Disaster_Response', engine, index=False, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()