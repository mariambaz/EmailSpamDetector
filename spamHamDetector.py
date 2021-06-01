import numpy as np
import re
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer

##MAIN##
if __name__ == "__main__":

  dataFrame = pd.read_csv (r'./spam_ham_dataset.csv')

  punctuation = '''!()-[]\{\};:'"\,<>./?@=#$%^&*_~1234567890'''

  file = open('english-stopwords.txt','r')
  stopWords = [word.strip() for word in file.readlines()]
  file.close()

  porter = PorterStemmer()

  preprocessed = []
  uniqueWordsSet = set()

  for b in dataFrame['Body']:
    b = str(b)
    b = re.sub("https*\S+", " ", b)
    b = re.sub('\s{2,}', " ", b)
    for char in b:
      if (char in punctuation):
        b = b.replace(char, "")
    puncRemoved = ""
    for word in nltk.word_tokenize(b):
      word = word.lower()
      if (word in stopWords):
        continue
      stemmed = porter.stem(word)
      puncRemoved += stemmed + " "
      uniqueWordsSet.add(stemmed)
    preprocessed.append(puncRemoved.split())

  dataFrame['Body'] = preprocessed

  newDF = pd.DataFrame(np.zeros((len(preprocessed),len(uniqueWordsSet))),columns=list(uniqueWordsSet))

  for it, b in enumerate(dataFrame['Body']):
    for word in b:
      newDF[word].iloc[it] += 1
  print(newDF)