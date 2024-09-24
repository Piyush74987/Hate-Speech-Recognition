import pandas as pd
import os
import numpy as np
import nltk
import re
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

stemmer = nltk.SnowballStemmer("english")

from nltk.corpus import stopwords
import string

stopword = set(stopwords.words("english"))

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(current_dir + "/twitter_data.csv")

df["labels"] = df["class"].replace(
    {
        0: "Hate Speech Detected",
        1: "Offensive language detected",
        2: "No hate and offensive speech",
    }
)
df = df[["tweet", "labels"]]


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    nostop: list[str]
    nostop = [word for word in text.split(" ") if word not in stopword]
    text = " ".join(nostop)
    nostop = [stemmer.stem(word) for word in text.split(" ")]
    return " ".join(nostop)


df["tweet"] = df.apply(lambda x: clean(x["tweet"]), axis=1)
x = np.array(df["tweet"])
y = np.array(df["labels"])


cv = CountVectorizer(stop_words="english")
x = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(score)
res = confusion_matrix(y_test, y_pred)
print(res)

# Save the trained model
joblib.dump(clf, "decision_tree_model.pkl")
joblib.dump(cv, "count_vectorizer.pkl")
