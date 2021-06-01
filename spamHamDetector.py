import numpy as np
import pandas as pd

##MAIN##
if __name__ == "__main__":
  #Reading data from
  dataFrame = pd.read_csv (r'./spam_ham_dataset.csv')
  #Preprocessing
  punc = '''!()-[]\{\};:'"\,<>./?@=#$%^&*_~'''
  afterPunc = []
  for b in dataFrame['Body']:
    for char in b:
      if (char in punc):
        b = b.replace(char, "")
    
    afterPunc.append(b)
  dataFrame['Body'] = afterPunc
  print(dataFrame['Body'])