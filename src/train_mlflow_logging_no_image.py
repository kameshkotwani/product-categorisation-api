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
from sklearn.model_selection import ParameterGrid

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

# # function to pre process the text
# def preprocess_nltk(text):
#     text = text.lower()
#     tokens = word_tokenize(text)
#     tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
#     tokens = [t for t in tokens if t not in stop_words]
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     return " ".join(tokens)

# # Apply the preprocessing function to the 'name' and brand column

# X_train['name'] = X_train['name'].apply(preprocess_nltk)
# X_train['brandName'] = X_train['brandName'].apply(preprocess_nltk)

# # Apply the preprocessing function to the 'name' and brand column of test data
# X_test['name'] = X_train['name'].apply(preprocess_nltk)
# X_test['brandName'] = X_train['brandName'].apply(preprocess_nltk)

# Train the Catboost model with mlflow experiment tracking


mlflow.set_experiment("Catboost Vanilla")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

#
base_params = {
    "task_type": 'GPU',
    "devices": '0',
    "eval_metric": 'Accuracy',
    "loss_function": 'MultiClass',
    "verbose": 0  # Set to 0 to reduce clutter during grid search
}

param_grid = {
    "iterations": [ 300,1000, 500],
    "depth": [6, 7]
}

# Convert parameter grid to a list of dictionaries.
grid = list(ParameterGrid(param_grid))

# ---------------------------
# Step 2: Set Up MLflow Experiment and Parent Run
# ---------------------------


# Start a parent run to group all tuning runs
with mlflow.start_run(run_name="processed_text_run_1") as parent_run:
    # ---------------------------
    # Step 3: Iterate Over Parameter Grid and Train Models
    # ---------------------------
    for grid_params in grid:
        # Merge the base parameters with the current grid combination
        current_params = {**base_params, **grid_params}
        
        # Start a nested MLflow run for this specific combination.
        with mlflow.start_run(run_name=f"{current_params}", nested=True) as child_run:
            # Create and train the model with current parameters
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
            
            model = CatBoostClassifier(**current_params)

            model.fit(
                train_pool,
                early_stopping_rounds=20,
                verbose=10
            )
            
            # Get predictions and evaluate performance on the test set.
            y_preds = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_preds)
            report = classification_report(y_test, y_preds, output_dict=True, zero_division=0)
            
            # Log parameters and performance metrics to MLflow.
            mlflow.log_params(current_params)
            mlflow.log_metric("accuracy", test_accuracy)
            mlflow.log_dict(report, artifact_file="classification_report.json")
            
            # Log the trained model.
            mlflow.catboost.log_model(model, "model")
            
            print(f"Completed run with params: {current_params}")
            print(f"Test Accuracy: {test_accuracy}")
    

