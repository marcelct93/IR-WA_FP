#####Imports
#Imports for general usage
import json
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.option_context('display.max_rows', None)
pd.set_option('display.width', 800) 
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
import math
import time
from config import *
import re
import datetime
import os
import sys
import pickle
import operator
import uuid

#Imports for plots
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import seaborn as sns 


#Imports for scraping
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

#Imports for searching, indexing and ranking
import nltk  
nltk.download('stopwords');
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import collections
from numpy import linalg as la

    


#####Classes for general usage

#Class for create a top inside a loop
class topper:
    """
    Class to get the x vals with more score. The strategy is simple: only append if the new value is 
    greater than the minimum of the values retained by the topper (and only when a new append is done,
    the minimum is updated (value and score)). It incorporates also a method to sort (which should be
    used after all the appends), and to conver to dataframe, given the names of the labels for value and score.
    """
    def __init__(self, size):
        self.size = size
        self.top = [('',0)]*size
        self.minimum = 0
    def append(self, val, score):
        if score > self.minimum: #If the new score is greater than the mininum...
            minimum_index = np.argmin([i[1] for i in self.top]) # compute the position of the minimum
            self.top[minimum_index] = (val,score) # insert new (value, score) replacing the minimum
            new_minimum_index = np.argmin([i[1] for i in self.top]) # compute again minimum index
            self.minimum = self.top[new_minimum_index][1] # get the new minimum score
    def sort(self):
        self.top.sort(key=operator.itemgetter(1), reverse=True) # sort the top descending
    def to_dataframe(self, label1, label2):
        vals, scores = map(list, zip(*self.top)) # map to lists, and then create df
        return pd.DataFrame({label1: vals, label2: scores})


#####Functions for general usage

#Functions to save and load data
def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
    
def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_df(PATH): 
    """
    Load all the data from the tweets dataset.
    """  
    try:
        with open(PATH + '/covid_0.json', 'rb') as f: # try reading the first file
            data = f.readlines()
            data = [json.loads(str_) for str_ in data]
        df_tweets = pd.DataFrame.from_records(data)
    except: # if it cannot be read, raise this message
        print("No data found in {}/ path, please run create_collection.py.".format(PATH))
        exit(0)
    for filename in os.listdir(PATH): # read the rest of the files and concat to the dataframe created above
        if filename.startswith('covid_') and filename != 'covid_0.json':
            with open(PATH + '/' + filename, 'rb') as f:
                data = f.readlines()
                data = [json.loads(str_) for str_ in data]
                df_tweets = pd.concat([pd.DataFrame.from_records(data), df_tweets], ignore_index=True, sort=False)
    return df_tweets

def getTerms(tweet, stops):
    """
    Given the text of a tweet, return a list of terms
    """
    stemming = PorterStemmer()
    tweet = tweet.replace("RT ", "").strip() # drop RT tag
    tweet = tweet.lower() # lower the characters
    tweet = re.sub(r'[^\w\s]','',tweet).strip() #remove puntiation
    tweet = tweet.split()  # split by blank spaces
    tweet = [x for x in tweet if x not in stops] # ignore stops 
    tweet = [stemming.stem(word) for word in tweet] # stem the terms
    return tweet


def preprocess(PATH):
    """
    Preprocesses the dataset given the path (it calls to load_df), and returns a "simplified" dataframe: df_tweets_simp
    """
    
    print("\n======================\nLoading dataset...")
    df_tweets = load_df(PATH) # load dataset
    print("\n======================\nPreprocessing dataset...")
    
    df_tweets = df_tweets[['id', 'text', 'user', 'entities', 'quoted_status', 'created_at']] # define the columns in order to get rid of...
    # ...data that we don't need
    
    df_tweets_simp = pd.DataFrame() # initialize simplified dataframe

    # the following lists retrieve some information in the same order as the input columns
    likes = []
    retweets = []
    hashtags = [] # list of lists

    for quoted_tweet in df_tweets['quoted_status']: # use the column quoted status to look for retweets
        if not pd.isnull(quoted_tweet): # if it's a retweet, append likes and retweets counts
            likes.append(quoted_tweet['favorite_count'])
            retweets.append(quoted_tweet['retweet_count'])
        else:
            likes.append(0) # else, append 0s
            retweets.append(0)
    
    for entity in df_tweets['entities']: # use the column entities to look for hashtags
        if not entity['hashtags']: # if there is no hashtags, append and empty list
            hashtags.append([])
        else:
            tweet_hashtag = [] # else, get the hashtags
            for hashtag in entity['hashtags']:
                tweet_hashtag.append(hashtag['text'])
            hashtags.append(tweet_hashtag)
    
    # assign the lists to the columns
    df_tweets_simp['likes'] = likes
    df_tweets_simp['retweets'] = retweets
    df_tweets_simp['hashtags'] = hashtags
    
    # get the date
    df_tweets_simp['date'] = df_tweets['created_at']
    
    # anonymize the user using a hash function with low collision rate
    df_tweets_simp['user'] = [str(uuid.uuid5(uuid.NAMESPACE_OID,user['screen_name'])) for user in df_tweets['user']]
    
    # retain the tweet id
    df_tweets_simp['id'] = df_tweets['id']
    
    # find all mentions in "text" column and replace them by its hash value, consistent with the user column
    #(we want to obfuscate the username, but keep the relations)
    regex = re.compile(r"@[A-z0-9]*", re.IGNORECASE) # regex to find mentions (@[A-z0-9]*)
    anon_texts = []
    for text in df_tweets['text']:
        mentions= set(regex.findall(text)) # find all mentions
        for mention in mentions: # replace them by "mentions" to its hashvalue
            text = text.replace(mention, '@' + str(uuid.uuid5(uuid.NAMESPACE_OID,mention[1:])))
        anon_texts.append(text)
    # save the anonymized texts
    df_tweets_simp['text'] = anon_texts
        
    del df_tweets # delete df_texts to free space
    
    # get the url from the tweet id
    df_tweets_simp['url'] = ["https://twitter.com/i/web/status/" + str(tweet_id) for tweet_id in df_tweets_simp['id']]
    
    # get the stops
    stops = set(stopwords.words('english'))
    
    # getTerms (see function above). The lists of terms are column this way we save...
    #...a lot of time in further processing operations
    df_tweets_simp['terms'] = [getTerms(text, stops) for text in df_tweets_simp['text']]
    
    # return the simplified tweets
    return df_tweets_simp


def create_index_tfidf(tweets, numTweets):
    """
    Given the collection of tweets (already preprocessed) and the numTweets, create the index, idf, df
    """
    start_time = time.time()
    index = defaultdict(list)
    idf = defaultdict()
    df = defaultdict()
    i = 0
    for i, row in tweets.iterrows(): # for each tweet: for each distinct term in the list of terms...  
            i += 1                   # create a new entry in the index with index[term] = (tweet_id, frequency)
            [index[term].append((row['id'],row['terms'].count(term))) for term in set(row['terms'])]
            if(i % 10000 == 0):
                print("{} tweets have been indexed in  {} seconds." .format((i+1),np.round(time.time() - start_time,2)))
    
    # generate the df and idf dictionaries
    print("Creating df and idf...")
    for key in index: # for each term
        count = len(index[key])  # compute the number of tuples in index[term]
        df[key] = count # define the entry with the number of tuples
        idf[key] = np.round(np.log(float(numTweets/count)),4) # define the entry with the:
        # log_4(number of tweets/number of tuples)
        
    return index, df, idf




######Functions for search

def search(query, index, df_tweets):
    """
    As we are only dealing with AND queries, the search simply uses the intersection of the tweet_ids
    of the different queries.
    """
    stops = set(stopwords.words('english'))
    query = getTerms(query, stops)
    for i, term in enumerate(query):
        try:
            if (i == 0):
                tweets = set([term[0] for term in index[term]])
            else:
                termTweets = set([term[0] for term in index[term]])
                tweets = tweets.intersection(termTweets) #Compute the intersection of the last term with the new one
        except:
            pass
    
    return df_tweets[df_tweets['id'].isin(tweets)]


def submit_search(top, index, df_tweets):
    """
    Function to submit search and display a sample of size top
    """
    start_time = 0
    # gets the result of the query
    result_tweets = pd.DataFrame()
    # define the columns to be displayed
    columns = ['text', 'user', 'date', 'hashtags', 'likes', 'retweets', 'url']
    while(len(result_tweets)==0): # check if empty (useful if a search does not have results) to offer the user a reformulation
        # asks for the query
        print("\nInsert your query:\n")
        myQuery = input()
        
        start_time = time.time()
        
        result_tweets = search(myQuery, index, df_tweets) # uses the searc function defined above
        
        if len(result_tweets) == 0:
            print("\n======================\nNo matches found.")
        else:

            print("\n======================\nSample of {} results out of {} for the searched query:\n".format(top, len(result_tweets)))
            print(result_tweets.head(top)[columns].reset_index(drop=True)) # display the result
            print("Total time to find the tweets: {} seconds." .format(np.round(time.time() - start_time,2)))
    return result_tweets, myQuery



######Functions for ranking

def max_min_norm(value, maximum, minimum):
    """
    Simple max-min normalization, with a little trick to avoid 0s
    """
    return ((value + 1) - minimum)/((maximum + 1) - minimum)

def tfidf_rank(query, tweets, index, idf, top):
    """
    It ranks by tf-idf: given query, tweets (simplified dataframe), index, idf and size of top
    """
    stops = set(stopwords.words('english'))
    terms = getTerms(query, stops) # get list of terms of the query
    tweetVectors=defaultdict(lambda: [0]*len(terms)) # generate a dict to tweet vector...
    #...initialized to 0 with the size of the query
    queryVector=[0]*len(terms) # same with the query 

    query_terms_count = collections.Counter(terms) # get the terms of the query with its frequency
    query_norm = la.norm(list(query_terms_count.values())) # get the norm of the query
    
    tweets_id = tweets['id'].tolist() # convert the tweets['id'] to list of ids

    for termIndex, term in enumerate(terms): # for each term of the query
        if term not in index: # if not in index, ignore it
            continue
                       
        queryVector[termIndex]=query_terms_count[term]/query_norm * idf[term] # df*idf of the query term

        for tweetIndex, (tweet_id, tf) in enumerate(index[term]): # for every tweet indexed
            if tweet_id in tweets_id: # if the tweet is in tweets_id (result of the search)
                # get the terms of the tweet with its frequency and the norm
                tweet_terms_count= list(collections.Counter(tweets[tweets['id']==tweet_id]['terms'].to_list().pop()).values())
                tweet_norm = la.norm(tweet_terms_count) 
                tweetVectors[tweet_id][termIndex] = tf/tweet_norm * idf[term] # set into the position of the term vecot df*idf

    top_20 = topper(top) # create a topper (see above) with the size of the top
    # append to the top the dot product of the normalized vectors...
    #...(remember that topper only retains the top k tuples (value, score))
    [top_20.append(tweet_id, np.dot(curTweetVec, queryVector)) for tweet_id, curTweetVec in tweetVectors.items()]
    top_20.sort() # sort the top
    rankedTweets = top_20.to_dataframe('id', 'score') # convert to dataframe
    rankedTweets = pd.merge(rankedTweets,tweets, on='id',how='left') # left join with tweets to recover the information...
    # ... related to the tweet ids in the top
    
    return rankedTweets

def popularity_rank(query, tweets, index, idf, top):
    """
    Our ranking method, that uses the normalized counts of retweets and likes as weights for the conventional tf-idf.
    """
    # same first steps as tf-idf
    stops = set(stopwords.words('english'))
    terms = getTerms(query, stops)
    tweetVectors = defaultdict(lambda: [0]*len(terms)) 
    queryVector = [1]*len(terms)
    
    # get the maximums and minimums for further normalization
    max_likes = tweets['likes'].max()
    min_likes = tweets['likes'].min()
    max_retweets = tweets['likes'].max()
    min_retweets = tweets['likes'].min()
    
    # same as tf-idf
    tweets_id = tweets['id'].tolist()
    query_terms_count = collections.Counter(terms)  
    query_norm = la.norm(list(query_terms_count.values())) # get the norm of the query
    
    # convert columns tweets.id, tweets.likes and tweets.retweets to dict [id] -> (likes, retweets)
    tweets_id_populuarity_dict = dict(zip(tweets.id, zip(tweets.likes, tweets.retweets)))
    
    for termIndex, term in enumerate(terms): # for each term of the query
        if term not in index: # if not in index, ignore it
            continue
                       
        queryVector[termIndex]=query_terms_count[term] * idf[term] # df*idf of the query term

        for tweetIndex, (tweet_id, tf) in enumerate(index[term]): # for every tweet indexed
            if tweet_id in tweets_id: # if the tweet is in tweets_id (result of the search)
                # get the terms of the tweet with its frequency and the norm
                tweet_terms_count= list(collections.Counter(tweets[tweets['id']==tweet_id]['terms'].to_list().pop()).values())
                tweet_norm = la.norm(tweet_terms_count) 
                tweetVectors[tweet_id][termIndex] = tf/tweet_norm * idf[term] # set into the position of the term vecot df*idf

    top_20 = topper(top) # create topper size top
    # retain the top 20 WEIGHTENING the tf-idf  score by the norm of retweets and likes count
    [top_20.append(tweet_id, 
                   max_min_norm(tweets_id_populuarity_dict[tweet_id][0], max_likes, min_likes)
                   *max_min_norm(tweets_id_populuarity_dict[tweet_id][1], max_retweets, min_retweets)
                   *np.dot(curTweetVec, queryVector)) 
                   for tweet_id, curTweetVec in tweetVectors.items()]
    top_20.sort() # sort top
    
    rankedTweets = top_20.to_dataframe('id', 'score') # to df
    rankedTweets = pd.merge(rankedTweets,tweets, on='id',how='left') # left join with tweets df

    
    return rankedTweets


def rank(top, result_tweets, index, idf, myQuery):
    """
    Function to call (interctively) one of the ranking functions defined above and display the final top 20
    """
    columns = ['text', 'user', 'date', 'hashtags', 'likes', 'retweets', 'url', 'score'] # columns to be displayed
    methods_dict = {"1":tfidf_rank,"2":popularity_rank} # dict of methods (functions defined above as object)
    selection = 0
    ranked_tweets = pd.DataFrame()
    start_time = 0
    while(selection not in methods_dict): # while selection does not correspond to one of the methods of methods_dict...
        # ask the user for the preferred ranking method
        selection = input("\n======================\nAvailable ranking methods:\n\n  1. Ranking based on TF-IDF.\n  2. Ranking based on popularity.\n\nInsert the number of the preferred ranking method:\n\n")
        
        if selection in methods_dict: # if selection is valid...
            start_time = time.time()
            ranked_tweets = methods_dict[selection](myQuery, result_tweets, index, idf, top) # get top 20 tweets...
            # ... using the method selected
        else:
            print("No method corresponding to your selection, try again.")

    # display results
    if len(ranked_tweets) == 0:
        print("\n======================\nNo matches were found.")
    else:
        print("\n======================\nTop {} of {} matches using {}:\n".format(top, len(result_tweets), methods_dict[selection].__name__))
        print(ranked_tweets[columns].reset_index(drop=True))
        print("Total time to rank the tweets: {} seconds." .format(np.round(time.time() - start_time,2)))
