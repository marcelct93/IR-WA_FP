						REQUIRED LIBRARIES
-----------------------------------------------------------------------------------------------------------------------------------------------
json
pandas
numpy
matplotlib
csv
collections
math
time
config
re
datetime
os
sys
pickle
operator
uuid
wordcloud
PIL
seaborn
tweepy
nltk  
array
sklearn
gensim
networkx
random
scipy
implicit
igraph




					INSTRUCTIONS TO RUN THE SEARCH ENGINE
-----------------------------------------------------------------------------------------------------------------------------------------------
(In the path search-egine/)
1. Collect the tweets:
	Command:
		python create_collection.py 1000

	Where 1000 is the number of tweets to be retrieved (it can be any number, but it has to be multiple of 1000). You can modify the twitter tokens in create_collection.py.

	The collection will be stored in search-engine/data (the folder will be created) in files of the size of 1000 tweets (e.g. data/covid_0.json, data/covid_1.json, etc.). 

-----------------------------------------------------------------------------------------------------------------------------------------------
(In the path search-egine/)
2. Run the search engine:
	Requirements:
	You have to run before the create_collection script (or to load manually into search-engine/data a json file containing the tweets with the following filename: covid_0.json).	

	Command:
		python search.py

	It is an interactive script. It will load the data collection, reprocess it and finally it will ask for a query with:
		Insert your query:
	Type your query and press enter.

	Then it will display a sample of 20 results, and ask for a ranking method with:
		Available ranking methods:

		  1. Ranking based on TF-IDF.
		  2. Ranking based on popularity.

		Insert the number of the preferred ranking method:
	Type 1 or 2 to select the ranking method, and press enter.

	Finally, it will ask if you want to do a new query:
		Do you want to submit a new query? (Y/N)
	Type y or n and press enter.

