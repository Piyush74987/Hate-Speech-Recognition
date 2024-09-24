from helpers.context_detection_helper import get_context
from helpers.hate_speech_helper import clean, process_text
from helpers.helper import get_visible_text


def context_detection_handler(url: str):
    text = get_visible_text(url)
    if text is None:
        return {"error": "Failed to fetch the webpage."}
    context = get_context(text.split("\n"))
    return {"context": context}


def hate_speech_url_handler(url: str):
    detected = False
    text = get_visible_text(url)
    if text is None:
        return {"error": "Failed to fetch the webpage.", "hate_speech": detected}
    text = clean(text)
    detected = process_text(text)
    return {"error": "", "hate_speech": detected}


def hate_speech_handler(text: str):
    cleaned_text = clean(text)
    detected = process_text(cleaned_text)
    return {"hate_speech": detected}
