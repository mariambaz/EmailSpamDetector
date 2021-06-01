import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

def preProcessing(listOfContent):
  punctuation = '''!()-[]\{\};:'"\,<>./?@=#$%^&*_~1234567890'''
  file = open('english-stopwords.txt','r')
  stopWords = [word.strip() for word in file.readlines()]
  file.close()
  porter = PorterStemmer()
  preprocessed = []
  for b in dataFrame['Body']:
    for char in b:
      if (char in punctuation):
        b = b.replace(char, "")
    puncRemoved = ""
    for word in b.split():
      if (word in stopWords):
        continue
      puncRemoved += porter.stem(word) + " "
    puncRemoved = puncRemoved.lower()
    preprocessed.append(puncRemoved)
  return preprocessed
##MAIN##
if __name__ == "__main__":
  #Reading data from
  dataFrame = pd.read_csv (r'./spam_ham_dataset.csv')
  #Preprocessing
  dataFrame['Body'] = preProcessing(dataFrame['Body'])
  print(dataFrame['Body'])