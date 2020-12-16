######Imports required
from utils import *

######Listener definition
class MyStreamListener(StreamListener):
    def __init__(self, api, OUTPUT_FILENAME, stop_condition=10):
        # initialize
        super(MyStreamListener, self).__init__()       
        self.num_tweets = 0
        self.filename = OUTPUT_FILENAME
        self.stop_condition = stop_condition 

    def on_status(self, status):
        # write tweets to filename define
        with open(self.filename, "a+") as f:
            tweet = status._json 
            f.write(json.dumps(tweet) + '\n')
            self.num_tweets += 1
            
            # stop if reached the stop limit
            if self.num_tweets <= self.stop_condition:
                return True
            else:
                return False
        
    def on_error(self, status):
        print(status) 
        return False


######Define API keys
access_token1 = '1236732195630448640-j8aA8GAK48lgdSfwH9AMRfIZJRY8jn'
access_token_secret1 = 'pV1oOmIAARLzB3jWMjNMTpaP5eL0uXXJOUBtYN1HJqThL'
consumer_key1 = 'qONF6qWqgroXJph813Klgp8bw'
consumer_secret1 = 'RuIqud8pUCIdvfn2IhpB1R9opav5u6Cjn5WNbKqMDX82gOR1Gh'

#######Authenticate
auth = OAuthHandler(consumer_key1, consumer_secret1)
auth.set_access_token(access_token1, access_token_secret1)
api = API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


######Retreive tweets
start_time = time.time()

#Get size of collection from argv
print('\n\n')
try:
    collection_size = int(sys.argv[1])
except:
    print("Please, specify the number of tweets to be retrieved (the number must be a multiple of 1000).")
    exit()
#We are going to create a new file for every 1000 tweets
stop_condition = 1000
iterations = int(collection_size/stop_condition)

#Keywords
TRACKING_KEYWORDS = ['covid-19', 'covid', 'coronavirus', 'corona', 'sars-cov', 'sars-covid',
                     'sars-cov-19' , 'sars-cov', 'sars-cov-2', 'SARS-CoV', 'SARS-CoV-2', 
                     'SARS-COVID', 'epidemic', 'pandemic', 'quarantine', 'lockdown', '#COVID', 
                     '#coronavirus', '#COVID19', '#StayHomeStaySafe', '#StayHome', '#LockDownNow']

#Check if path exists
PATH = 'data' 
if not os.path.exists(PATH):
    os.makedirs(PATH)

print("\n======================\nStarting retrieveing:")

#Listen and save blocks of 1000 tweets
for i in range(iterations):
    iter_start = np.round(time.time(), 2)
    print("  Iteration {} started at {}.".format(i, time.strftime("%H:%M:%S", time.localtime())))
    OUTPUT_FILENAME = PATH + '/covid_{}.json'.format(i) #output filename
    l = MyStreamListener(api, OUTPUT_FILENAME, stop_condition) # initialize the streamer
    stream = Stream(auth=api.auth, listener=l) # stream
    stream.filter( # define filter
    track=TRACKING_KEYWORDS, 
    is_async=False, 
    languages = ['en']
    )
    print("  Iteration {} ({} collected) completed in {} seconds.".format(i, (i+1)*stop_condition , np.round(time.time() - iter_start, 2)))
    time.sleep(10) # sleep every 1000 tweets for "politeness" 

print("Total time to create the collection: {} seconds" .format(np.round(time.time() - start_time, 2)))



######Load all files from data/
with open(PATH + '/covid_0.json', 'rb') as f:
    data = f.readlines()
    data = [json.loads(str_) for str_ in data]
df_tweets = pd.DataFrame.from_records(data)

for filename in os.listdir(PATH):
    if filename.startswith('covid_') and filename != 'covid_0.json':
        with open(PATH + '/' + filename, 'rb') as f:
            data = f.readlines()
            data = [json.loads(str_) for str_ in data]
            df_tweets = pd.concat([pd.DataFrame.from_records(data), df_tweets], ignore_index=True)

print("\n======================\nLoading dataset...")
print("{} tweets have been loaded to the dataset.".format(len(df_tweets)))


######Compute statistics
tweets_count = len(df_tweets)
unique_tweets_count = len(df_tweets['id'].unique())
unique_users_count = len(np.unique([user['id'] for user in df_tweets['user']]))
retweets_count = sum([retweet.startswith('RT ') for retweet in df_tweets['text']])
total_words_count = sum([len(text.split()) for text in df_tweets['text']])

print("\n======================\nStatistics: ")
print(" Total number of tweets: {}.".format(tweets_count))
print(" Total number of unique tweets: {}.".format(unique_tweets_count))
print(" Total number of unique users: {}.".format(unique_users_count))
print(" Total number of original tweets (not retweet): {}.".format(tweets_count - retweets_count))
print(" Total number of words (not unique): {}.".format(total_words_count))

