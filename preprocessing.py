import random

import contractions
import matplotlib.pyplot
import numpy as np
import pandas
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import time

from sklearn.metrics import f1_score

nltk.download("stopwords")
from textblob import TextBlob
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import apriori
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score



def removeUrl(text):
    """
        Remove URLs from a sample string
    """
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def removeHtml(text):
    """
        Remove the html in sample text
    """
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


def removeNonAscii(text):
    """
        Remove non-ASCII characters
    """
    return re.sub(r'[^\x00-\x7f]', r'', text)


def removeSpecialCharacters(text):
    """
        Remove special special characters, including symbols, emojis, and other graphic characters
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def removePunct(text):
    """
        Remove the punctuation
    """
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)


def otherClean(text):
    """
            Other manual text cleaning techniques
        """
    # Typos, slang and other
    sample_typos_slang = {
        "w/e": "whatever",
        "usagov": "usa government",
        "recentlu": "recently",
        "ph0tos": "photos",
        "amirite": "am i right",
        "exp0sed": "exposed",
        "<3": "love",
        "luv": "love",
        "amageddon": "armageddon",
        "trfc": "traffic",
        "16yr": "16 year"
    }

    # Acronyms
    sample_acronyms = {
        "mh370": "malaysia airlines flight 370",
        "okwx": "oklahoma city weather",
        "arwx": "arkansas weather",
        "gawx": "georgia weather",
        "scwx": "south carolina weather",
        "cawx": "california weather",
        "tnwx": "tennessee weather",
        "azwx": "arizona weather",
        "alwx": "alabama weather",
        "usnwsgov": "united states national weather service",
        "2mw": "tomorrow"
    }

    # Some common abbreviations
    sample_abbr = {
        "$": " dollar ",
        "€": " euro ",
        "4ao": "for adults only",
        "a.m": "before midday",
        "a3": "anytime anywhere anyplace",
        "aamof": "as a matter of fact",
        "acct": "account",
        "adih": "another day in hell",
        "afaic": "as far as i am concerned",
        "afaict": "as far as i can tell",
        "afaik": "as far as i know",
        "afair": "as far as i remember",
        "afk": "away from keyboard",
        "app": "application",
        "approx": "approximately",
        "apps": "applications",
        "asap": "as soon as possible",
        "asl": "age, sex, location",
        "atk": "at the keyboard",
        "ave.": "avenue",
        "aymm": "are you my mother",
        "ayor": "at your own risk",
        "b&b": "bed and breakfast",
        "b+b": "bed and breakfast",
        "b.c": "before christ",
        "b2b": "business to business",
        "b2c": "business to customer",
        "b4": "before",
        "b4n": "bye for now",
        "b@u": "back at you",
        "bae": "before anyone else",
        "bak": "back at keyboard",
        "bbbg": "bye bye be good",
        "bbc": "british broadcasting corporation",
        "bbias": "be back in a second",
        "bbl": "be back later",
        "bbs": "be back soon",
        "be4": "before",
        "bfn": "bye for now",
        "blvd": "boulevard",
        "bout": "about",
        "brb": "be right back",
        "bros": "brothers",
        "brt": "be right there",
        "bsaaw": "big smile and a wink",
        "btw": "by the way",
        "bwl": "bursting with laughter",
        "c/o": "care of",
        "cet": "central european time",
        "cf": "compare",
        "cia": "central intelligence agency",
        "csl": "can not stop laughing",
        "cu": "see you",
        "cul8r": "see you later",
        "cv": "curriculum vitae",
        "cwot": "complete waste of time",
        "cya": "see you",
        "cyt": "see you tomorrow",
        "dae": "does anyone else",
        "dbmib": "do not bother me i am busy",
        "diy": "do it yourself",
        "dm": "direct message",
        "dwh": "during work hours",
        "e123": "easy as one two three",
        "eet": "eastern european time",
        "eg": "example",
        "embm": "early morning business meeting",
        "encl": "enclosed",
        "encl.": "enclosed",
        "etc": "and so on",
        "faq": "frequently asked questions",
        "fawc": "for anyone who cares",
        "fb": "facebook",
        "fc": "fingers crossed",
        "fig": "figure",
        "fimh": "forever in my heart",
        "ft.": "feet",
        "ft": "featuring",
        "ftl": "for the loss",
        "ftw": "for the win",
        "fwiw": "for what it is worth",
        "fyi": "for your information",
        "g9": "genius",
        "gahoy": "get a hold of yourself",
        "gal": "get a life",
        "gcse": "general certificate of secondary education",
        "gfn": "gone for now",
        "gg": "good game",
        "gl": "good luck",
        "glhf": "good luck have fun",
        "gmt": "greenwich mean time",
        "gmta": "great minds think alike",
        "gn": "good night",
        "g.o.a.t": "greatest of all time",
        "goat": "greatest of all time",
        "goi": "get over it",
        "gps": "global positioning system",
        "gr8": "great",
        "gratz": "congratulations",
        "gyal": "girl",
        "h&c": "hot and cold",
        "hp": "horsepower",
        "hr": "hour",
        "hrh": "his royal highness",
        "ht": "height",
        "ibrb": "i will be right back",
        "ic": "i see",
        "icq": "i seek you",
        "icymi": "in case you missed it",
        "idc": "i do not care",
        "idgadf": "i do not give a damn fuck",
        "idgaf": "i do not give a fuck",
        "idk": "i do not know",
        "ie": "that is",
        "i.e": "that is",
        "ifyp": "i feel your pain",
        "IG": "instagram",
        "iirc": "if i remember correctly",
        "ilu": "i love you",
        "ily": "i love you",
        "imho": "in my humble opinion",
        "imo": "in my opinion",
        "imu": "i miss you",
        "iow": "in other words",
        "irl": "in real life",
        "j4f": "just for fun",
        "jic": "just in case",
        "jk": "just kidding",
        "jsyk": "just so you know",
        "l8r": "later",
        "lb": "pound",
        "lbs": "pounds",
        "ldr": "long distance relationship",
        "lmao": "laugh my ass off",
        "lmfao": "laugh my fucking ass off",
        "lol": "laughing out loud",
        "ltd": "limited",
        "ltns": "long time no see",
        "m8": "mate",
        "mf": "motherfucker",
        "mfs": "motherfuckers",
        "mfw": "my face when",
        "mofo": "motherfucker",
        "mph": "miles per hour",
        "mr": "mister",
        "mrw": "my reaction when",
        "ms": "miss",
        "mte": "my thoughts exactly",
        "nagi": "not a good idea",
        "nbc": "national broadcasting company",
        "nbd": "not big deal",
        "nfs": "not for sale",
        "ngl": "not going to lie",
        "nhs": "national health service",
        "nrn": "no reply necessary",
        "nsfl": "not safe for life",
        "nsfw": "not safe for work",
        "nth": "nice to have",
        "nvr": "never",
        "nyc": "new york city",
        "oc": "original content",
        "og": "original",
        "ohp": "overhead projector",
        "oic": "oh i see",
        "omdb": "over my dead body",
        "omg": "oh my god",
        "omw": "on my way",
        "p.a": "per annum",
        "p.m": "after midday",
        "pm": "prime minister",
        "poc": "people of color",
        "pov": "point of view",
        "pp": "pages",
        "ppl": "people",
        "prw": "parents are watching",
        "ps": "postscript",
        "pt": "point",
        "ptb": "please text back",
        "pto": "please turn over",
        "qpsa": "what happens",  # "que pasa",
        "ratchet": "rude",
        "rbtl": "read between the lines",
        "rlrt": "real life retweet",
        "rofl": "rolling on the floor laughing",
        "roflol": "rolling on the floor laughing out loud",
        "rotflmao": "rolling on the floor laughing my ass off",
        "rt": "retweet",
        "ruok": "are you ok",
        "sfw": "safe for work",
        "sk8": "skate",
        "smh": "shake my head",
        "sq": "square",
        "srsly": "seriously",
        "ssdd": "same stuff different day",
        "tbh": "to be honest",
        "tbs": "tablespooful",
        "tbsp": "tablespooful",
        "tfw": "that feeling when",
        "thks": "thank you",
        "tho": "though",
        "thx": "thank you",
        "tia": "thanks in advance",
        "til": "today i learned",
        "tl;dr": "too long i did not read",
        "tldr": "too long i did not read",
        "tmb": "tweet me back",
        "tntl": "trying not to laugh",
        "ttyl": "talk to you later",
        "u": "you",
        "u2": "you too",
        "u4e": "yours for ever",
        "utc": "coordinated universal time",
        "w/": "with",
        "w/o": "without",
        "w8": "wait",
        "wassup": "what is up",
        "wb": "welcome back",
        "wtf": "what the fuck",
        "wtg": "way to go",
        "wtpa": "where the party at",
        "wuf": "where are you from",
        "wuzup": "what is up",
        "wywh": "wish you were here",
        "yd": "yard",
        "ygtr": "you got that right",
        "ynk": "you never know",
        "zzz": "sleeping bored and tired"
    }

    sample_typos_slang_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_typos_slang.keys()) + r')(?!\w)')
    sample_acronyms_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_acronyms.keys()) + r')(?!\w)')
    sample_abbr_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)')

    text = sample_typos_slang_pattern.sub(lambda x: sample_typos_slang[x.group()], text)
    text = sample_acronyms_pattern.sub(lambda x: sample_acronyms[x.group()], text)
    text = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], text)

    return text


def snowballStemmer(text):
    """
        Stem words in list of tokenized words with SnowballStemmer
    """

    stemmer = nltk.SnowballStemmer("english")
    stems = [stemmer.stem(i) for i in text]
    return stems


def lemmatizeWord(text):
    """
        Lemmatize the tokenized words
    """
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(i) for i in text]
    return lemma


def removeTypo(text):
    return TextBlob(text).correct().string


def normaliseToxicityValue(dataset):
    dataset["toxicity"] = preprocessing.binarize(X=dataset[["toxicity"]], threshold=0.0000000000001)
    return dataset


def preProcess(dataset):
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()

    dataset['cleanComment'] = dataset["comment"].apply(lambda x: x.lower())
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: contractions.fix(x))
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: removeUrl(x))
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: removeHtml(x))
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: removeNonAscii(x))
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: removeSpecialCharacters(x))
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: removePunct(x))
    dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: otherClean(x))

    # dataset["cleanComment"] = dataset["cleanComment"].apply(lambda x: removeTypo(x))

    dataset['tokenized'] = dataset['cleanComment'].apply(nltk.word_tokenize)

    stop = set(stopwords.words('english'))
    dataset['tokenized'] = dataset['tokenized'].apply(lambda x: [word for word in x if word not in stop])
    dataset['tokenized'] = dataset['tokenized'].apply(lambda x: snowballStemmer(x))
    dataset['tokenized'] = dataset['tokenized'].apply(lambda x: lemmatizeWord(x))
    dataset['cleanComment'] = dataset['tokenized'].apply(lambda x: ' '.join(x))
    normaliseToxicityValue(dataset)

    return dataset


def TfIdf(dataset):
    cv1 = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
    x = cv1.fit_transform(dataset)

    return x



def doApriori(dataset):
    te = TransactionEncoder();
    te_ary = te.fit_transform(dataset['tokenized'].head(2))

    df = pd.DataFrame(te_ary, columns=te.columns_)
    aprioriDf = apriori(df, use_colnames=True)

    return aprioriDf

from sklearn.feature_selection import SelectFromModel

def selectFeature(dataset):
    del dataset['comment']





def postProcessing(dataset):
    x = dataset["cleanComment"]
    y = dataset["toxicity"]

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=42)

    tempTfIdf = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")

    x_train_tfIdf = tempTfIdf.fit_transform(x_train)
    x_test_tfIdf = tempTfIdf.transform(x_test)


    lr = doLogisticRegression(x_train_tfIdf,y_train)
    knn = doKNN(x_train_tfIdf,y_train)
    xgb = doXGB(x_train_tfIdf,y_train)
    svc = doSVC(x_train_tfIdf,y_train)
    randomForest = doRandomForest(x_train_tfIdf,y_train)
    nB = doNaiveBayes(x_train_tfIdf,y_train)
    dT = doDecisionTree(x_train_tfIdf,y_train)
    kM = doKMeans(x_train_tfIdf,y_train)






    lrCross = doSumCrossValScore(lr,x_train_tfIdf,y_train)
    knnCross = doSumCrossValScore(knn,x_train_tfIdf,y_train)
    xgbCross = doSumCrossValScore(xgb,x_train_tfIdf,y_train)
    svcCross = doSumCrossValScore(svc, x_train_tfIdf, y_train)
    randomForestCross = doSumCrossValScore(randomForest, x_train_tfIdf, y_train)
    nBCross = doSumCrossValScore(nB, x_train_tfIdf, y_train)
    dTCross = doSumCrossValScore(dT, x_train_tfIdf, y_train)
    kMCross = doSumCrossValScore(kM, x_train_tfIdf, y_train)


    printDataSumData = {'Accuracy': [lrCross[0],knnCross[0],xgbCross[0],svcCross[0],randomForestCross[0],nBCross[0],dTCross[0],kMCross[0]],
                    'F1 value': [lrCross[1],knnCross[1],xgbCross[1],svcCross[1],randomForestCross[1],nBCross[1],dTCross[1],kMCross[1]],
                    'Presicion': [lrCross[2],knnCross[2],xgbCross[2],svcCross[2],randomForestCross[2],nBCross[2],dTCross[2],kMCross[2]],
                    'Recall' :    [lrCross[3],knnCross[3],xgbCross[3],svcCross[3],randomForestCross[3],nBCross[3],dTCross[3],kMCross[3]]}

    printDataSum = pd.DataFrame(printDataSumData,index=['Log Regression', 'KNN', 'XGB', 'SVC', 'Random Forest','Naive Bayes','Decision Tree','K Mean'])


    print(printDataSum)



def doSumCrossValScore(model,xTest,yTest):
    kCross = 2
    acc = cross_val_score(model,xTest,yTest,cv=kCross,scoring='accuracy')
    f1Macro = cross_val_score(model, xTest, yTest, cv=kCross, scoring='f1')
    precision = cross_val_score(model, xTest, yTest, cv=kCross, scoring='precision')
    reCall = cross_val_score(model, xTest, yTest, cv=kCross, scoring='recall')

    return acc.max(),f1Macro.max(),precision.max(),reCall.max()



def calculateSqrt(xtrain,xtest,ytrain,ytest,model):

    trainSqrt = sqrt(mean_squared_error(ytrain,model.predict(xtrain)))
    testSqrt = sqrt(mean_squared_error(ytest,model.predict(xtest)))
    return testSqrt-trainSqrt




def percDistibution(dataset):
  
    tempDataset =dataset[(dataset['toxicity'] ==0)]
    print(" Normal yorum sayısı ",len(tempDataset))
    print(" Toksik yorum sayısı ",len(dataset)-len(tempDataset))





def createDataset(x):
    nRandom = random.randint(0,10000)
    tRandom = random.randint(0,10000)
    neutral_train = x[x['toxicity'] == 0].iloc[nRandom:nRandom+1000, :]
    toxic_train = x[x['toxicity'] !=  0].iloc[tRandom:tRandom+1000, :]
    balanced_train = pd.concat([toxic_train, neutral_train], axis=0)

    return balanced_train

def doLogisticRegression(xValue,yValue):
    lr = LogisticRegression()
    lr.fit(xValue,yValue)
    return lr

def doKNN(xValue, yValue):
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(xValue,yValue)
    return knn

def doXGB(xValue, yValue):
    xgb=XGBClassifier(use_label_encoder =False,eval_metric='rmse')
    xgb.fit(xValue,yValue)

    return xgb

def doSVC(xValue, yValue):
    svc = LinearSVC()
    svc.fit(xValue,yValue)
    return svc

def doRandomForest(xValue, yValue):
    randomForest = RandomForestClassifier(n_estimators=100,random_state=42)
    randomForest.fit(xValue,yValue)
    return randomForest



def doNaiveBayes(xValue,yValue):
    nB = MultinomialNB()
    nB.fit(xValue,yValue)
    return nB

def doDecisionTree(xValue,yValue):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(xValue,yValue)
    return clf

def doKMeans(xValue,yValue):
    kmeans = KMeans(n_clusters=2,random_state=42).fit(xValue,yValue)

    return kmeans

"""def doHierarchicalClustering(xValue,yValue):
    hc = FeatureAgglomeration().fit(xValue,yValue)
    return hc"""