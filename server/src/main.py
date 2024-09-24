from fastapi import FastAPI
from handlers import (
    hate_speech_handler,
    context_detection_handler,
    hate_speech_url_handler,
)
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def home():
    return {"message": "You have reached the DSBDA project made by TheAimHero"}


@app.get("/hate_speech")
def hate_speech_url(url: str | None = None):
    if url is None:
        return {"error": "Please provide a valid URL."}
    return hate_speech_url_handler(url)


class HateSpeechRequest(BaseModel):
    text: str | None = None


@app.post("/hate_speech")
def hate_speech(req_body: HateSpeechRequest | None = None):
    if req_body is None:
        return {"error": "Please provide a valid text."}
    if req_body.text is None or req_body.text == "":
        return {"error": "Please provide a valid text."}
    return hate_speech_handler(req_body.text)


@app.get("/context_detection")
def context_detection(url: str | None = None):
    if url is None:
        return {"error": "Please provide a valid URL."}
    return context_detection_handler(url)
