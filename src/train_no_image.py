"""
This script trains the Vanilla Catboost model.
"""
import mlflow
import os
from dotenv import load_dotenv
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from get_data import get_data_without_images

load_dotenv()
# get the data
X_train, X_test, y_train, y_test = get_data_without_images()

# some preprocesssing in the X_train['name'] column
# preprocessing function
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# function to pre process the text
def preprocess_nltk(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Apply the preprocessing function to the 'name' and brand column

X_train['name'] = X_train['name'].apply(preprocess_nltk)
X_train['brandName'] = X_train['brandName'].apply(preprocess_nltk)

# Train the Catboost model with mlflow experiment tracking
# parameters_dict

params = {
        "iterations": 100,
        "task_type": 'GPU',
        "devices": '0',
        "random_state": 42,
        "eval_metric": 'Accuracy',
        "loss_function": 'MultiClass'
    }

fit_params = {
        "early_stopping_rounds": 20
    }

mlflow.set_experiment("Catboost Vanilla")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
with mlflow.start_run() as run:
    
    train_pool = Pool(
    data=X_train, 
    label=y_train, 
    cat_features=['brandName'],
    text_features=['name'])

# test pool for Catboost
    test_pool = Pool(
        data=X_test, 
        label=y_test, 
        cat_features=['brandName'],
        text_features=['name'])

  
    
    # generate the model object
    model = CatBoostClassifier(**params)

  # Train the model
    model.fit(
        train_pool,
        verbose=10,
        early_stopping_rounds=20
    )

    y_preds = model.predict(X_test)
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_preds)


    # Log the model
    mlflow.catboost.log_model(model, "model")

    # Log model parameters
    mlflow.log_params(params)
    mlflow.log_params(fit_params)
    mlflow.log_metric("accuracy", accuracy)
    # Log the classification report
    report = classification_report(y_test, y_preds, output_dict=True,zero_division=0)
    mlflow.log_dict(report,artifact_file="classification_report.json")
    

