'''
Date: 10th March 2021
Author: Daniela Yassuda Yamashita
Description: process_data.py script. Load and Clean data from csv files.
            Save the cleaned data to SQLite database.
'''

# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Load data function

    This function loads the raw data from two csv files and
    creates a DataFrame of the concatenation of both.

    INPUT:
    messages_filepath -> String referring to the filepath of 
                         the csv containing the disaster message
                         files
    categories_filepath -> String referring to the filepath of the 
                           csv containing the categories of messages
    
    OUTPUT:
    df -> DataFrame of the concatenation of the two csv data
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer')
    
    return df

def clean_data(df):
	"""
	Clean data function

	This function cleans the data, by processing text,
	removing duplicate data and nan values.

	INPUT:
	df -> DataFrame of raw messages and categories data

	OUTPUT:
	df -> DataFrame of cleaned data
	"""

	# Create a dataframe of the 36 individual category columns
	categories = df['categories'].str.split(';',expand=True)

	# Rename de DataFrame columns according to the categories name
	# specified in the firs row of 'categories' datagrame
	row = categories.loc[0,:] # Select the first row of the categories dataframe
	category_colnames = row.apply(lambda x: x.split('-')[0])# use this row to extract a list of new column names for categories.
	categories.columns = category_colnames # rename the columns of `categories`

	# Convert category values to just numbers 0 or 1.
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str.split('-').str.get(-1)
		categories[column] = categories[column].astype(int)
		
	# convert column from string to numeric
	categories[categories > 1] = 1

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
    """
    Save Data function
    
    This function save the cleaned data to a SQLite database.

    INPUT:
    df -> DataFrame of cleaned data
            message
    database_filename -> String referring to the filepath in which the
                         database will be saved.
    """
    engine = create_engine('sqlite:///'+ database_filename)
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