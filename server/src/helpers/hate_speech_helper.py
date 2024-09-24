import joblib
import re
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

from helpers.helper import get_cur_dir


clf: DecisionTreeClassifier = joblib.load(
    get_cur_dir() + "/../../models/hatespeech/decision_tree_model.pkl"
)
cv: CountVectorizer = joblib.load(
    get_cur_dir() + "/../../models/hatespeech/count_vectorizer.pkl"
)

stopword = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")


def process_text(input_text: str) -> bool:
    text = clean(input_text)
    x_new = cv.transform([text]).toarray()
    predicted_label = clf.predict(x_new)
    if (
        predicted_label[0] == "Hate Speech Detected"
        or predicted_label[0] == "Offensive language detected"
    ):
        return True
    else:
        return False


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
