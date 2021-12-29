# SentimentAnalysis
Basic NLP Sentiment analysis implementation for toxicity in tweets. You can use your dataset but the column names must be "[comment] and [toxicity]".

## Preprocessing

First, we are taking 1000 toxic 1000 pure comments by random inside the createDataset() Function. After that, we are doing some word removal, stopword cleaning, stemming, and lemmatization.

You can write the toxic and pure tweets number by percDistribution() function.

## Future Representation

By TfIdf() function, you can call and see the TfIdf values directly from main and with the doApriori() function, you can make apriori. The default in this project is first two value of ['tokenized'] column. Here's an example of TfIdf :

![image](https://user-images.githubusercontent.com/18538179/147675939-9bc243f8-68d2-4c1a-9dce-75becc0d567c.png)

## Post Processing

Inside the postProcessing() function we are separating the dataset to test and train and we are using :

- Logistic regression
- Knn
- Xgb
- SVC
- RandomForest
- Naive Bayes
- DecisionTree
- Kmean

## performance review

At the end of the postProcessing() function the program does k cross-validation to test and prints the mean values :

![image](https://user-images.githubusercontent.com/18538179/147676640-a5b89639-5f77-4e79-a858-d60a89babff9.png)
