from  preprocessing import *

"[comment], [toxicity]"
dataset = pd.read_csv('toxic_comment_classification_dataset.csv',nrows=100000)

dataset = createDataset(dataset)

dataset = preProcess(dataset)

tfIdf = TfIdf(dataset["cleanComment"])

featured = selectFeature(dataset)

final = postProcessing(dataset)













