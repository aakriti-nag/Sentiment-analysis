#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re as re
import contractions
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score,mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score


# In[2]:


#! pip install bs4 # in case you don't have it installed
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


df = pd.read_table('amazon_reviews_us_Beauty_v1_00.tsv', sep='\t', low_memory=False, on_bad_lines='skip')
df


# In[4]:


df.reset_index(drop=True, inplace=True)


# ## Keep Reviews and Ratings

# In[5]:


enter = df [['star_rating','review_body']]
enter.head()


# In[6]:


enter['star_rating'].unique()


# In[7]:


enter['star_rating'] = enter['star_rating'].str.replace(r'\D', '')


# In[8]:


# Drop rows that has NaN values 
enter = enter.dropna()


# In[9]:


enter['star_rating'].unique()


# In[10]:


#keep only integers
enter['star_rating'] = enter['star_rating'].astype(int)


# In[11]:


enter.head()


# In[12]:


enter['star_rating'].unique()


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[13]:


def filter(x):
    if (x <= 2):
        return '1'
    if (x == 3):
        return '2'
    if (x > 3) :
        return '3'
enter['category'] = enter['star_rating'].apply(filter)


# In[14]:


enter.head()


# In[15]:


enter = enter.drop('star_rating', axis=1)


# In[16]:


enter = enter.groupby('category').apply(lambda s: s.sample(20000))
enter


# In[ ]:





# # Data Cleaning
# 
# 

# In[17]:


#average length of reviews before cleaning
before_clean = enter['review_body'].str.len().mean()


# In[18]:


# convert all reviews into lowercase
enter['review_body'] = enter['review_body'].str.lower()


# In[19]:


enter


# In[20]:


#remove the HTML and URLs from the review
def remove(s):
    s=s.replace(r'<[^<>]*>', '')
    s=s.replace(r'http\S+', '').replace(r'www\S+', '')
    return s
enter['review_body']=enter['review_body'].apply(remove)
enter


# In[21]:


#Perform contractions on the reviews
enter['review_body'] = enter['review_body'].apply(lambda x: [contractions.fix(word) for word in x.split()])
enter['review_body'] = [' '.join(map(str, l)) for l in enter['review_body']]


# In[22]:


#remove non alphabetical characyers and extra spaces
def remove(sentence):
    sentence=re.sub(r'[^a-zA-Z]',' ',sentence)
    return re.sub(' +', ' ',sentence)
enter['review_body'] = enter['review_body'].apply(remove)


# In[23]:


enter


# In[24]:


# average length of reviews after cleaning 
after_clean=enter['review_body'].str.len().mean()

print('Before and after cleaning : '+str(before_clean), ', '+str(after_clean))
# # Pre-processing

# In[26]:


# average length of reviews before preprocessing 
before_preprocessing = enter['review_body'].str.len().mean()


# ## remove the stop words 

# In[27]:


def remove_non_ascii(text): 
    return ''.join(i for i in text if ord(i)<128) 
 
enter['review_body'] = enter['review_body'].apply(remove_non_ascii) 


# In[28]:


# stop = stopwords.words('english')
# enter['review_body'] = enter['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# ## perform lemmatization  

# In[29]:


lem = WordNetLemmatizer()

enter['review_body']=enter['review_body'].apply(lambda x: " ".join([lem.lemmatize(w) for w in nltk.word_tokenize(x)]))

enter


# In[30]:


# average length of reviews after preprocessing
after_preprocessing = enter['review_body'].str.len().mean()


# In[31]:


print('Before and after preprocessing : '+str(before_preprocessing), ', '+str(after_preprocessing))


# # TF-IDF Feature Extraction

# In[32]:


# from sklearn.feature_extraction.text import TfidfVectorizer
imp = TfidfVectorizer(ngram_range=(1,3))
x = imp.fit_transform(enter['review_body'])


# In[33]:


# from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, enter['category'], test_size=0.2, random_state=42)


# # Perceptron

# In[34]:


from sklearn.linear_model import Perceptron
clf_per = Perceptron(random_state=42, tol=1e-3)
clf_per.fit(X_train,Y_train)


# In[35]:


pred_per= clf_per.predict(X_test)


# In[36]:


print(confusion_matrix(Y_test,pred_per))


# In[37]:


target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, pred_per, target_names=target_names))


# In[39]:


precision, recall, fscore, support = score(Y_test, pred_per)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[40]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test,pred_per))


# In[43]:


precision, recall, fscore, support = score(Y_test, pred_per)
sum_precision=0
sum_recall=0
sum_f1score=0
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],', Recall:', recall[i], ', F1 score:',fscore[i])
    sum_precision+= precision[i]
    sum_recall+= recall[i]
    sum_f1score+= fscore[i]
print('Averages : ' + str(sum_precision/3)+' , '+str(sum_recall/3)+' , '+str(sum_f1score/3))


# In[ ]:





# # SVM

# In[44]:


from sklearn.svm import LinearSVC
clf_svm= LinearSVC(random_state=0)
clf_svm.fit(X_train,Y_train)


# In[45]:


pred_svm=clf_svm.predict(X_test)


# In[46]:


target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, pred_svm, target_names=target_names))


# In[49]:


precision, recall, fscore, support = score(Y_test, pred_svm)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[50]:


print("Accuracy:",metrics.accuracy_score(Y_test,pred_svm))


# In[51]:


precision, recall, fscore, support = score(Y_test, pred_svm)
sum_precision=0
sum_recall=0
sum_f1score=0
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],', Recall:', recall[i], ', F1 score:',fscore[i])
    sum_precision+= precision[i]
    sum_recall+= recall[i]
    sum_f1score+= fscore[i]
print('Averages : ' + str(sum_precision/3)+' , '+str(sum_recall/3)+' , '+str(sum_f1score/3) )


# In[ ]:





# # Logistic Regression

# In[52]:


from sklearn.linear_model import LogisticRegression

clf_log= LogisticRegression(random_state=42, max_iter=10000)
clf_log.fit(X_train,Y_train)


# In[53]:


pred_log=clf_log.predict(X_test)


# In[54]:


target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, pred_log, target_names=target_names))


# In[56]:


precision, recall, fscore, support = score(Y_test, pred_log)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[57]:


print("Accuracy:",metrics.accuracy_score(Y_test,pred_log))


# In[58]:


precision, recall, fscore, support = score(Y_test, pred_log)
sum_precision=0
sum_recall=0
sum_f1score=0
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],', Recall:', recall[i], ', F1 score:',fscore[i])
    sum_precision+= precision[i]
    sum_recall+= recall[i]
    sum_f1score+= fscore[i]
print('Averages : ' + str(sum_precision/3)+' , '+str(sum_recall/3)+' , '+str(sum_f1score/3) )


# # Naive Bayes

# In[59]:


from sklearn.naive_bayes import MultinomialNB

clf_naive = MultinomialNB()
clf_naive.fit(X_train, Y_train)


# In[60]:


pred_naive = clf_naive.predict(X_test)


# In[61]:


target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, pred_naive, target_names=target_names))


# In[62]:


print("Accuracy:",metrics.accuracy_score(Y_test,pred_naive))


# In[63]:


precision, recall, fscore, support = score(Y_test, pred_naive)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[64]:


precision, recall, fscore, support = score(Y_test, pred_naive)
sum_precision=0
sum_recall=0
sum_f1score=0
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],', Recall:', recall[i], ', F1 score:',fscore[i])
    sum_precision+= precision[i]
    sum_recall+= recall[i]
    sum_f1score+= fscore[i]
print('Averages : ' + str(sum_precision/3)+' , '+str(sum_recall/3)+' , '+str(sum_f1score/3))


# In[ ]:




