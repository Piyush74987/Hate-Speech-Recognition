# Hate Speech Recognition
This project is aimed at identifying and classifying hate speech in textual data using machine learning and natural language processing (NLP). By automating the detection of harmful content, this tool helps in mitigating online harassment and abusive language, making digital platforms safer.

## Key Features:
### Data Preprocessing:
Cleaned and preprocessed raw text by removing noise such as stop words, punctuation, special characters, and URLs.
Employed tokenization, lemmatization, and case normalization to enhance text quality for analysis.
### Feature Engineering:
Implemented various text vectorization techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and CountVectorizer to convert the processed text into machine-understandable formats.
Explored word embeddings for semantic analysis of text.
### Model Training:
Built multiple classification models including Logistic Regression, Support Vector Machines (SVM), Random Forest, and Naive Bayes to predict hate speech.
Applied hyperparameter tuning to optimize the models for better accuracy and efficiency.
### Evaluation:
Evaluated the models using performance metrics like accuracy, precision, recall, and F1-score to ensure robust detection.
Visualized results using confusion matrix and ROC curves to interpret classification performance.
### Web Application:
Deployed the trained model using a Flask-based web application, allowing users to input text and receive real-time hate speech classification.
User-friendly interface for quick and easy identification of hateful content.
Technologies and Tools:
## Languages:
Python (for scripting, model building, and deployment),
## Libraries:
### NLP:
NLTK, TextBlob,
### Machine Learning: 
Scikit-learn
### Data Handling:
Pandas, NumPy
### Model Evaluation:
Matplotlib, Seaborn
### Web Framework:
Flask (for building and deploying the web app)
### Model Deployment: 
Flask API for real-time prediction of hate speec
