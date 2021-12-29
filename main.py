import time

from  preprocessing import *


"[comment], [toxicity]"
dataset = pd.read_csv('toxic_comment_classification_dataset.csv',nrows=100000)





dataset = createDataset(dataset)

#percDistibution(dataset)


"""pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)"""


"""
doAP = doApriori(dataset)
print(doAP)
"""





dataset = preProcess(dataset)

tfIdf = TfIdf(dataset["cleanComment"])


featured = selectFeature(dataset)

percDistibution(dataset)





final = postProcessing(dataset)













