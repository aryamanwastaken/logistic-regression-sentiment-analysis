import os  # for os.path.basename
import pandas as pd  # for reading csv file
import numpy as np  # for array operations
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for visualization
import re  # for regular expressions
import nltk  # for text manipulation
from nltk.corpus import stopwords  # for stop words
from nltk.stem.porter import PorterStemmer  # for stemming
from nltk.stem import WordNetLemmatizer  # for lemmatization 
from nltk.tokenize import word_tokenize,sent_tokenize  # for tokenization
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words
from sklearn.model_selection import train_test_split, cross_val_score  # for splitting into train and test
from sklearn.linear_model import LogisticRegression  # for logistic regression
from sklearn.metrics import classification_report  # for model evaluation
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning

train = '../data/Corona_tweets_train.csv'  # path to train data
test = '../data/Corona_tweets_test.csv'  # path to test data

trainOriginal = pd.read_csv(train, encoding='latin-1')  # reading train data
testOriginal = pd.read_csv(test, encoding='latin-1')  # reading test data

train = trainOriginal.copy()  # making a copy of train data
test = testOriginal.copy()  # making a copy of test data

train.head()  # printing first 5 rows of train data
test.head()  # printing first 5 rows of test data

train.info()  # printing info of train data

train.isnull().sum()  # checking for null values in train data

train['Location'].value_counts()[:60]  # checking for top 60 locations in train data

#splitting location into word pairs
train['Location'] = train['Location'].str.split(",").str[0]
test['Location'] = test['Location'].str.split(",").str[0]  

train['Location'].value_counts()[:60]  # checking for top 60 locations in train data

train['TweetAt'].value_counts()  # checking for tweets in train data

train['Sentiment'].value_counts()  # checking for sentiments in train data

plt.figure(figsize=(10,10))  # setting figure size
sns.countplot(y='Location',data=train,order=train.Location.value_counts().iloc[  
    0:19].index).set_title("Tweeted locations")  # plotting locations

sns.set_style("whitegrid")  # setting style
sns.set(rc={'figure.figsize':(11,8)})  # setting figure size
sns.countplot(train['Sentiment'])  # plotting sentiments

labels = ['Positve', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative']  # setting labels
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#ff5645']  # setting colors
explode = (0.05,0.05,0.05,0.05,0.05)   # setting explode
plt.pie(train.Sentiment.value_counts(), colors = colors, labels=labels,
        autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)  # plotting pie chart
centreCircle = plt.Circle((0,0),0.70,fc='white')  # setting circle
fig = plt.gcf()  # getting current figure
fig.gca().add_artist(centreCircle)  # adding circle
plt.tight_layout()  # setting layout
plt.show()  # showing plot

plotDf = train.iloc[:,[2,5]] #[:,[2,5]] is the location and sentiment columns
plotDf  # printing plotDf

sns.set(rc={'figure.figsize':(15,9)})  # setting figure size
gg = train.Location.value_counts()[:5].index  # getting top 5 locations
plt.title('Sentiment Categories of the First 5 Top Locations', fontsize=16, fontweight='bold')  # setting title
sns.countplot(x = 'Location', hue = 'Sentiment', data = plotDf, order = gg)  # plotting countplot

train['Identity'] = 0  # adding identity column in train data
test['Identity'] = 1   # adding identity column in test data
covid = pd.concat([train, test])  # concatenating train and test data
covid.reset_index(drop=True, inplace=True)  # resetting index

covid.head()  # printing first 5 rows of covid data

covid['Sentiment'] = covid['Sentiment'].str.replace('Extremely Positive', 'Positive')  # replacing Extremely Positive with Positive
covid['Sentiment'] = covid['Sentiment'].str.replace('Extremely Negative', 'Negative')  # replacing Extremely Negative with Negative
 
covid = covid.drop('ScreenName', axis=1)  # dropping ScreenName column
covid = covid.drop('UserName', axis=1)  # dropping UserName column
covid  # printing covid data

sns.set_style("whitegrid")  # setting style
sns.set(rc={'figure.figsize':(11,8)})  # setting figure size
sns.countplot(covid['Sentiment'])  # plotting sentiments

labels = ['Positve', 'Negative', 'Neutral']  # setting labels
colors = ['lightblue','lightsteelblue','silver']  # setting colors
explode = (0.1, 0.1, 0.1)  # setting explode
plt.pie(covid.Sentiment.value_counts(), colors = colors, labels=labels,
        shadow=300, autopct='%1.1f%%', startangle=90, explode = explode)  # plotting pie chart
plt.show()   # showing plot

plt.figure(figsize=(10,10))  # setting figure size
sns.countplot(y='Location',data=train,order=train.Location.value_counts().iloc[
    0:19].index).set_title("Tweeted locations")  # plotting locations

covid['Sentiment'] = covid['Sentiment'].map({'Neutral':0, 'Positive':1, 'Negative':2})  # mapping sentiments to numbers

hashTags=covid['OriginalTweet'].str.extractall(r"(#\S+)")  # extracting hashtags
hashTags = hashTags[0].value_counts()  # counting hashtags
hashTags[:50]  # printing top 50 hashtags

mentions = train['OriginalTweet'].str.extractall(r"(@\S+)")  # extracting mentions
mentions = mentions[0].value_counts()  # counting mentions
mentions[:50]  # printing top 50 mentions

#  Function to clean tweets
def clean(text):
    text = re.sub(r'http\S+', " ", text)  # removing urls
    text = re.sub(r'@\w+',' ',text)  # removing mentions
    text = re.sub(r'#\w+', ' ', text)  # removing hashtags
    text = re.sub(r'\d+', ' ', text)    # removing digits
    text = re.sub('r<.*?>',' ', text)  # removing html tags
    text = text.split()  # splitting text
    text = " ".join([word for word in text if not word in stopWord])  # removing stop words
    
    return text  # returning text

stopWord = stopwords.words('english')  # setting stop words
covid['OriginalTweet'] = covid['OriginalTweet'].apply(lambda x: clean(x))  # applying clean function
covid.head()    # printing first 5 rows of covid data

covid = covid[['OriginalTweet','Sentiment','Identity']]  # selecting columns
covid.head()    # printing first 5 rows of covid data
 
covid['Corpus'] = [nltk.word_tokenize(text) for text in covid.OriginalTweet]  # tokenizing text
lemma = nltk.WordNetLemmatizer()  # setting lemmatizer
covid.Corpus = covid.apply(lambda x: [lemma.lemmatize(word) for word in x.Corpus], axis=1)  # lemmatizing text
covid.Corpus = covid.apply(lambda x: " ".join(x.Corpus),axis=1)  # joining text

covid.head()    # printing first 5 rows of covid data

train = covid[covid.Identity==0]  # selecting train data
test = covid[covid.Identity==1]  # selecting test data
train.drop('Identity',axis=1, inplace=True)  # dropping Identity column
test.drop('Identity',axis=1, inplace=True)  # dropping Identity column
test.reset_index(drop=True,inplace=True)  # resetting index

train.head()  # printing first 5 rows of train data

test.head()  # printing first 5 rows of test data

XTrain = train.Corpus  # setting XTrain
yTrain = train.Sentiment  # setting yTrain

XTest = test.Corpus     # setting XTest
yTest = test.Sentiment  # setting yTest

XTrain, XVal, yTrain, yVal = train_test_split(XTrain, yTrain, test_size=0.2,random_state=42)  # splitting train data into train and validation data

XTrain.shape, XVal.shape, yTrain.shape, yVal.shape, XTest.shape, yTest.shape  # printing shapes of train, validation and test data        

vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=5).fit(covid.Corpus)  # setting vectorizer

# The CountVectorizer is utilized here to convert the text corpus into a sparse matrix of token counts.
# This transformation is essential for preparing the text data for machine learning algorithms which
# require numerical input data.

# Parameters used in CountVectorizer:

# - stop_words: 'english' is specified to remove common English words ('the', 'a', etc.) that do not
#   carry much meaning and are not useful for distinguishing between texts.
# - ngram_range: (1, 2) indicates that both unigrams (single words) and bigrams (pairs of consecutive words)
#   will be considered. This can capture phrases and multi-word expressions which might be important for
#   understanding the sentiment.
# - min_df: 5 specifies that the vocabulary will only include terms that appear in at least five documents.
#   This helps to focus on words that are more likely to be relevant and avoids overfitting to very rare terms.

# The fit method of CountVectorizer learns the vocabulary dictionary and the transform method converts the text
# documents into a document-term matrix. This matrix has rows corresponding to the documents and columns
# corresponding to the terms, with cell values being the term frequencies in the documents.

# The output of the CountVectorizer is a sparse matrix XTrainVec, XValVec, and XTestVec, which are then used
# to train the logistic regression model and evaluate its performance on validation and test datasets.

XTrainVec = vectorizer.transform(XTrain)  # transforming XTrain
XValVec = vectorizer.transform(XVal)  # transforming XVal
XTestVec = vectorizer.transform(XTest)  # transforming XTest

logReg = LogisticRegression(random_state=42)  # setting logistic regression model

cross_val_score(LogisticRegression(random_state=42),  
                XTrainVec, yTrain, cv=10, verbose=1, n_jobs=-1).mean()  # getting cross validation score

model = logReg.fit(XTrainVec, yTrain)  # fitting model

print(classification_report(yVal, model.predict(XValVec)))  # printing classification report
 
penalty = ['l2']  # setting penalty
C = np.logspace(0, 4, 10)  # setting C
hyperparameters = dict(C=C, penalty=penalty)  # setting hyperparameters
 
logRegGrid = GridSearchCV(logReg, hyperparameters, cv=5, verbose=0)  # setting grid search


bestModel = logRegGrid.fit(XTrainVec, yTrain)  # fitting best model

# Best hyperparameters combination

print('Best Penalty:', bestModel.best_estimator_.get_params()['penalty'])  # printing best penalty
print('Best C:', bestModel.best_estimator_.get_params()['C'])  # printing best C

# Final Logistic Regression model performance

yPred = bestModel.predict(XTestVec)  # predicting yPred
 

print(classification_report(yTest, bestModel.predict(XTestVec)))    # printing classification report


######################################################################################################
# EXTRA
######################################################################################################

#  Classification metrics:
# Precision deals with the accuracy of the positive predictions.
# precision = TP / TP + FP
# TP is the number of true positives, and FP is the number of false positives.

# Recall, also called sensitivity or true positive rate (TPR) is the ratio of positive instances that
# are correctly detected by the classifier.
# recall = TP / TP + FN
# TP is the number of true positives FP is the number of false positives and FN is the number of
# false negatives.

# But the metric of choice to measure the performance of the logistic regression model in this
# project is the F1-score.The F1 score is the harmonic mean of precision and recall.
# Whereas the regular mean treats all values equally, the harmonic mean gives much more weight to low
# values. As a result, the classifier will only get a high F1 score if both recall and precision are
# high.
# A less concise metric also available is the confusion matrix. The general idea involves counting
# the number of times instances of class A are classified as class B.

#  Implementation:

# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_val_predict

# yPred = bestModel.predict(XTestVec)
# print(confusion_matrix(yTest, yPred))

# NB: it's possible that classification metrics wont't be able to handle a mix of multilabel-indicator
# and multiclass targets.
