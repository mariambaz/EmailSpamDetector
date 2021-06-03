import numpy as np
import re
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class multinomialNaiveBayes:
  def init(self): 
    pass

  def getClasses(self, y_train):
    return np.unique(y_train)

  def fit(self, X_train, y_train, smoothingConstant = 1):
    numRows, numColumns = X_train.shape
    self.classes = self.getClasses(y_train)
    n_classes = len(self.classes)

    self.priors = np.zeros(n_classes)
    self.likelihoods = np.zeros((n_classes, numColumns))

    for classIndex, cl in enumerate(self.classes):
        X_class = X_train[cl == y_train]
        self.priors[classIndex] = X_class.shape[0] / numRows
        sum_in_category = X_class.sum(axis=0)
        smoothed = sum_in_category + smoothingConstant
        self.likelihoods[classIndex, :] = smoothed/np.sum(smoothed)

  def predict(self, X_test):
    predictions = []
    for x in X_test:
      predictions.append(self.predictSingleRow(x))
    return predictions

  def predictSingleRow(self, x_test):
    posteriors = []
    for classIndex, cl in enumerate(self.classes):
        prior_c = np.log(self.priors[classIndex])
        likelihoods_c = self.findLikelihoods(self.likelihoods[classIndex,:], x_test)
        posteriors_c = np.sum(likelihoods_c) + prior_c
        posteriors.append(posteriors_c)
    return self.classes[np.argmax(posteriors)]

  def findLikelihoods(self, cls_likeli, x_test):
    return np.log(cls_likeli) * x_test

  def score(self, X_test, y_test):
    y_pred = self.predict(X_test)
    count = 0
    for i in range(0,len(y_pred)):
      if y_pred[i] == y_test.iloc[i]:
        count += 1
    return count/len(y_test)


##MAIN##
if __name__ == "__main__":
 
  #Reading data set from csv file
  dataFrame = pd.read_csv(r'spam_ham_dataset.csv')

  #All punctuation characters to be removed
  punctuation = '''!()-[]\{\};:'"\,<>./?=$%^&*_~1234567890'''

  #Reading all stop words from a file
  file = open('english-stopwords.txt','r')
  stopWords = [word.strip() for word in file.readlines()]
  file.close()

  #used for stemming
  porter = PorterStemmer()

  preprocessed = []

  wordAndValues = {}

  for count, b in enumerate(dataFrame['Body']):
    b = str(b)
    b = re.sub("https*\S+", " ", b) #Remove urls 
    b = re.sub('\s{2,}', " ", b) #Replace any concatenated sequence of whitespaces with a single 1
    b = re.sub("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", ' ', b)	#Remove Emails
    b = re.sub("@\S+", " ", b) #Remove mentions
    b = re.sub("#\S+", " ", b) #Remove hashtags
    b = re.sub("\'\w+", ' ', b) #Remove apostrophe and what follows

    for char in b: #Remove punctuation
      if (char in punctuation):
        b = b.replace(char, " ")

    for word in nltk.tokenize.word_tokenize(b): #Tokenization process
      word = word.lower()                       #Convert to lowercase
      if (word in stopWords):                   #Neglect stop words
        continue
      stemmed = porter.stem(word)               #Stemming
      if (stemmed in wordAndValues.keys()):     #Getting count of stemmed words
        wordAndValues[stemmed][count] += 1
      else: 
        wordAndValues[stemmed] = [0 for i in range(len(dataFrame))] #initialize if not yet
        wordAndValues[stemmed][count] += 1
  emailBOW = pd.DataFrame(wordAndValues).to_numpy()

  X_train, X_test, y_train, y_test = train_test_split(emailBOW, dataFrame['Label'], test_size =0.2, random_state = 0)

  #MY IMPLEMENTATION
  classifier  = multinomialNaiveBayes()
  classifier.fit(X_train, y_train, 1)
  print("Accuracy:", classifier.score(X_test, y_test) * 100)