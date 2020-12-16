######Imports required
from utils import *

# input path
PATH = 'data'

start_time = time.time()

# preprocess the files in PATH (see utils.py)
df_tweets_simp = preprocess(PATH)


print("Total time to preprocess dataset of {} tweets: {} seconds." .format(len(df_tweets_simp),np.round(time.time() - start_time,2)))

start_time = time.time()
print("\n======================\nCreating index...")
numTweets = len(df_tweets_simp)
myIndex, df, idf = create_index_tfidf(df_tweets_simp,numTweets) # create index, df and idf (see utils.py)
print("Total time to create the index: {} seconds." .format(np.round(time.time() - start_time,2)))
       
del df # delete df to free space

top = 20

submit_new_query = 'y'
while (submit_new_query == 'y'): # while 'y' allow the user to submit new queries
    result_tweets, myQuery = submit_search(top, myIndex, df_tweets_simp) # submit search and get results (utils.py)
    rank(top, result_tweets, myIndex, idf, myQuery) # rank results (see utils.py, it allows the user to choose the method)
    print("\n======================\nDo you want to submit a new query? (Y/N)")
    submit_new_query = input().lower() # allow the user to submit a new query
