FROM python

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
