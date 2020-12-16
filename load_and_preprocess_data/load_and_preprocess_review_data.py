# imports
import json
import pandas as pd 
import numpy as np 
import re
import yaml
import ast
from datetime import datetime
from nltk.stem import PorterStemmer
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer

# load stop words and tokenizer
stop_words = set(stopwords.words('english'))
stop_words.remove('very')
tokenizer = RegexpTokenizer(r'\w+')

# load review data
f = open('./review_data/reviews.json', encoding='utf-8')

# store the data into a dataframe
master_list = []
for line in f:
    try:
        master_list.append(eval(line))
    except Exception as e:
        pass

master_df = pd.DataFrame(master_list, columns=columns)

print('Number of unique reviews : ', len(master_df.index))
print('Number of unique reviewers : ', master_df['reviewerName'].nunique())

print('The number of Null entries for each feature : ')
print(master_df.isnull().sum())

# drop the observations with null values
master_df = master_df.dropna()

# only get reviews that have ascii in them
clean_list = []
for idx, row in master_df.iterrows():
    review = row['reviewText']
    if review.isascii():
        clean_list.append(list(row))

master_df = pd.DataFrame(clean_list, columns=list(master_df.columns))

# convert reviewer name and review text to lowercase
master_df['reviewerName'] = master_df.apply(lambda x: x['reviewerName'].lower(), axis=1)
master_df['reviewText'] = master_df.apply(lambda x: str(x['reviewText']).lower(), axis=1)

# convert all categories to lower case
for idx, row in master_df.iterrows():
    vals = row['categories']
    try:
        vals = [val.lower() for val in vals]
        master_df.at[idx, 'categories'] = vals
    except Exception as e:
        master_df.at[idx, 'categories'] = 'no category'

# convert all the reviewTimes to datetime objects
for idx, row in master_df.iterrows():
    dt = row['reviewTime']
    master_df.at[idx, 'reviewTime'] = datetime.strptime(dt, '%b %d, %Y')
    udt = row['unixReviewTime']
    master_df.at[idx, 'unixReviewTime'] = datetime.fromtimestamp(udt)

# remove the stop words from review text
for idx, row in master_df.iterrows():
    print(idx)
    review = row['reviewText']
    try:
        words = word_tokenize(review)
        mod_review = []
        for word in words:
            if word not in stop_words:
                mod_review.append(word)
        mod_review = ' '.join(mod_review)
        master_df.at[idx, 'reviewText'] = mod_review
    except Exception as e:
        master_df.at[idx, 'reviewText'] = ''

# stemming all the words in master_df
for idx, row in master_df.iterrows():
    review = row['reviewText']
    try:
        words = word_tokenize(review)
        mod_review = []
        for word in words:
            mod_review.append(porter.stem(word))
        mod_review = ' '.join(mod_review)
        master_df.at[idx, 'reviewText'] = mod_review
    except Exception as e:
        master_df.at[idx, 'reviewText'] = ''

# remove the reviews that donot have any text after lemmatization and removing stop words
master_df = master_df.dropna() 

# filter out only the observations that are related to food and drinks businesses
restaurant_list = []
# convert all the categories to lower case
for idx, row in master_df.iterrows():
    vals = row['categories']
    flag = 0
    try:
        for val in vals:
            if 'restaurant' in val or 'bar' in val or 'cafe' in val or 'ice cream' in val or 'bistro' in val:
                flag = 1
        if flag == 1:
            restaurant_list.append(list(row))
    except Exception as e:
        pass

# convert those observations to a dataframe
restaurant_df = pd.DataFrame(restaurant_list, columns=list(master_df.columns))

# removing the punctuation marks from reviewText
for idx, row in restaurant_df.iterrows():
    review = row['reviewText']
    review = tokenizer.tokenize(review)
    review = ' '.join(review)
    restaurant_df.at[idx, 'reviewText'] = review

# store the preprocessed review data into restaurants.pickle
restaurant_df.to_pickle('restaurants.pickle')