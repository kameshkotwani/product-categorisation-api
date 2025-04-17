import pandas as pd
import numpy as np
from gensim.models import FastText
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import spacy
from get_data import get_data_without_images
# -------------------------------
# Step 1: Load and prepare data
# -------------------------------
X_train,X_test,y_train,y_test = get_data_without_images()

# -------------------------------
# Step 2: Tokenisation using spaCy
# -------------------------------
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    return [token.text.lower() for token in nlp(str(text)) if not token.is_punct and not token.is_space]

def tokenize_column(column):
    return [tokenize_text(text) or ['<unk>'] for text in column]

# Apply tokenisation
product_tokens = tokenize_column(X_train['name'])
brand_tokens = tokenize_column(X_train['brandName'])

# -------------------------------
# Step 3: Train FastText model
# -------------------------------
combined_corpus = product_tokens + brand_tokens
ft_model = FastText(sentences=combined_corpus, vector_size=100, window=3, min_count=1, sg=1, epochs=50)

# -------------------------------
# Step 4: Get sentence vectors
# -------------------------------
def sentence_vector(text):
    tokens = tokenize_text(text)
    vectors = [ft_model.wv[t] if t in ft_model.wv else ft_model.wv['<unk>'] for t in tokens]
    return np.mean(vectors, axis=0)

# Generate sentence vectors for product and brand
X_product = np.vstack([sentence_vector(text) for text in X_train['name']])
X_brand = np.vstack([sentence_vector(text) for text in X_train['brandName']])
X = np.concatenate([X_product, X_brand], axis=1)

# -------------------------------
# Step 5: Encode target labels
# -------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])

# -------------------------------
# Step 6: Train CatBoostClassifier
# -------------------------------
model = CatBoostClassifier(
    iterations=1000,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    class_weights="Balanced",
    verbose=100
)

model.fit(X, y)

# -------------------------------
# Step 7: Prediction function
# -------------------------------
def predict_category(product_text, brand_text):
    prod_vec = sentence_vector(product_text)
    brand_vec = sentence_vector(brand_text)
    full_vec = np.concatenate([prod_vec, brand_vec]).reshape(1, -1)
    pred_class = model.predict(full_vec)
    return label_encoder.inverse_transform(pred_class.astype(int))[0]

# -------------------------------
# Step 8: Example prediction
# -------------------------------
if __name__ == "__main__":
    example_product = "Revlon hair dye 10ml"
    example_brand = "Revlon"
    prediction = predict_category(example_product, example_brand)
    print(f"Predicted category: {prediction}")
