try:

    import pandas as pd

    import string

    from tqdm import tqdm
    from textblob import TextBlob

    from nltk.corpus import stopwords
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize
    import re

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    import swifter

    tqdm.pandas()
except Exception as e:
    print("Error : {} ".format(e))

import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

df = pd.read_json("./News_Category_Dataset_v3.json", lines=True)


df["category"].value_counts().plot(kind="bar", figsize=(15, 10))

df.columns

df.describe()

df.isna().sum()

df.head(2)

df["category"].unique()

stop_words_ = set(stopwords.words("english"))
wn = WordNetLemmatizer()
my_sw = ["make", "amp", "news", "new", "time", "u", "s", "photos", "get", "say"]


def black_txt(token):
    return (
        token not in stop_words_
        and token not in list(string.punctuation)
        and len(token) > 2
        and token not in my_sw
    )


def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    clean_text = [
        wn.lemmatize(word, pos="v")
        for word in word_tokenize(text.lower())
        if black_txt(word)
    ]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)


def subj_txt(text):
    return TextBlob(text).sentiment[1]


def polarity_txt(text):
    return TextBlob(text).sentiment[0]


def len_text(text):
    if len(text.split()) > 0:
        return len(set(clean_txt(text).split())) / len(text.split())
    else:
        return 0


df["text"] = df["headline"] + " " + df["short_description"]

df["text"] = df["text"].swifter.apply(clean_txt)
df["polarity"] = df["text"].swifter.apply(polarity_txt)
df["subjectivity"] = df["text"].swifter.apply(subj_txt)
df["len"] = df["text"].swifter.apply(lambda x: len(x))

X = df[["text", "polarity", "subjectivity", "len"]]
y = df["category"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
v = dict(zip(list(y), df["category"].to_list()))

text_clf = Pipeline(
    [
        ("vect", CountVectorizer(analyzer="word", stop_words="english")),
        ("tfidf", TfidfTransformer(use_idf=True)),
        ("clf", MultinomialNB(alpha=0.01)),
    ]
)

text_clf.fit(x_train["text"].to_list(), list(y_train))

import numpy as np

X_TEST = x_test.iloc[:, x_test.columns.get_loc("text")].tolist()
Y_TEST = list(y_test)

predicted = text_clf.predict(X_TEST)

c = 0

for doc, category in zip(X_TEST, predicted):

    if c == 2:
        break

    print("-" * 55)
    print(doc)
    print(v[category])
    print("-" * 55)

    c = c + 1

np.mean(predicted == Y_TEST)

docs_new = [
    "Ten Months After George Floyd’s Death, Minneapolis Residents Are at War Over Policing"
]

predicted = text_clf.predict(docs_new)

v[predicted[0]]

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(text_clf, f)

# load
with open("model.pkl", "rb") as f:
    clf2 = pickle.load(f)

docs_new = [
    "Ten Months After George Floyd’s Death, Minneapolis Residents Are at War Over Policing"
]
predicted = clf2.predict(docs_new)

v[predicted[0]]
