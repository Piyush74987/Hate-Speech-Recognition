import joblib
from helpers.helper import get_cur_dir

clf2 = joblib.load(get_cur_dir() + "/../../models/contextdetection/model.pkl")
v_loaded = joblib.load(get_cur_dir() + "/../../models/contextdetection/v_dict.pkl")


def get_context(input_text: list[str]) -> str:
    text = " ".join(input_text[:1000])
    predicted = clf2.predict([text])
    predicted_label = v_loaded[predicted[0]]
    return predicted_label
