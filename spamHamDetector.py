import numpy as np
import pandas as pd

##MAIN##
if __name__ == "__main__":
  print("HELLO")
  dataFrame = pd.read_csv (r'./spam_ham_dataset.csv')
  print (dataFrame)