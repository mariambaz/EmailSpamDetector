import numpy as np
import pandas as pd

##MAIN##
if __name__ == "__main__":
  #Reading data from
  dataFrame = pd.read_csv (r'./spam_ham_dataset.csv')

  #Preprocessing
  f = open('english-stopwords.txt','r')
  stopWords = [word.strip() for word in f.readlines()]
  f.close()
  punc = '''!()-[]\{\};:'"\,<>./?@=#$%^&*_~'''
  preprocessed = []
  for b in dataFrame['Body']:
    for char in b:
      if (char in punc):
        b = b.replace(char, "")
    puncRemoved = ""
    for word in b.split():
      if (word in stopWords):
        continue
      puncRemoved += word + " "
    preprocessed.append(puncRemoved)
  dataFrame['Body'] = preprocessed
  print(dataFrame['Body'])