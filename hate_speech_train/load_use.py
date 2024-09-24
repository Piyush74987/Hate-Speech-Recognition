import pandas as pd
import os
import argparse
import nltk
import re
import joblib

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


def preprocess_text_file(file_path):
    with open(file_path, "r") as file:
        hate_speech_lines = []  # List to store lines with hate speech
        for line in file:
            text = clean(line)
            x_new = cv.transform([text]).toarray()
            predicted_label = clf.predict(x_new)
            if (
                predicted_label[0] == "Hate Speech Detected"
                or predicted_label[0] == "Offensive language detected"
            ):
                hate_speech_lines.append(line)
        return hate_speech_lines  # Return the list of lines with hate speech


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg1", type=str, help="File Path")
    args = parser.parse_args()
    url = args.arg1

    # Load the saved model and vectorizer
    clf = joblib.load("decision_tree_model.pkl")
    cv = joblib.load("count_vectorizer.pkl")

    lines_with_hate_speech = preprocess_text_file(url)

    print(lines_with_hate_speech)
