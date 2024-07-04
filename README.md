# Sentiment-analysis

Text representations and the use of text classification for sentiment analysis. Sentiment analysis is extensively used to study customer behaviors using reviews and survey responses, online and social media, and healthcare materials for marketing and costumer service applications. Using the Amazon reviews dataset which contains real reviews for jewelry products sold on Amazon.

The dataset is downloadable at: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz

The steps involve :

Dataset Preparation - Keeping only the Reviews and Ratings fields in the input data frame to generate data. Creating a three-class classification problem according to the ratings. Selecting 20,000 random reviews from each rating class and create a balanced dataset to perform the required tasks on the downsized dataset. Split your dataset into 80% training dataset and 20% testing dataset.

Dataset Cleaning - Convert all reviews into lowercase, remove the HTML and URLs from the reviews, remove non-alphabetical characters, remove extra spaces, perform contractions on the reviews.

Preprocessing - Using NLTK package, and remove the stop words and perform lemmatization.

Feature Extraction - Use TF-IDF

Model Training - Using Perceptron, SVM, Logistic Regression, Multinomial Naive Bayes and calculate their precision, recall, and f1-score.
