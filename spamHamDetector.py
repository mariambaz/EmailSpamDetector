import numpy as np
import pandas as pd

def preProcessing(listOfContent):
  file = open('english-stopwords.txt','r')
  stopWords = [word.strip() for word in file.readlines()]
  file.close()
  punctuation = '''!()-[]\{\};:'"\,<>./?@=#$%^&*_~1234567890'''
  preprocessed = []
  for b in dataFrame['Body']:
    for char in b:
      if (char in punctuation):
        b = b.replace(char, "")
    puncRemoved = ""
    for word in b.split():
      if (word in stopWords):
        continue
      puncRemoved += word + " "
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