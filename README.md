# Sentiment Analysis

NLP Sentiment analysis implementation for toxicity detection in tweets. You can use your dataset but the column names must be "[comment] and [toxicity]".

## Preprocessing

createDataset function samples 1000 toxic and positive comments by random. After that, it does  stopword removal -> stemming -> lemmatization.

percDistribution() function can be used to log details about the dataset split.

## Feature Representation

By TfIdf() function, you can call and see the TfIdf values directly from main and with the doApriori() function, you can make apriori. The default in this project is first two value of ['tokenized'] column. Here's an example of TfIdf :

![image](https://user-images.githubusercontent.com/18538179/147675939-9bc243f8-68d2-4c1a-9dce-75becc0d567c.png)

## Post Processing

The postprocessing function splits the dataset into train and test sets. We then experiment with the following algorithms:

- Logistic regression
- Knn
- Xgb
- SVC
- RandomForest
- Naive Bayes
- DecisionTree
- Kmean

## Test Results

After postprocessing, we use k-cross-validation to test the performance.


![image](https://user-images.githubusercontent.com/18538179/147676640-a5b89639-5f77-4e79-a858-d60a89babff9.png)
