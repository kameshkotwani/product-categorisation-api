from catboost import CatBoostClassifier,Pool
from image_embeddings_generator import EmbeddingsGenrator
from get_data import get_data_with_images
# code to generate class_weights
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score,accuracy_score
import mlflow
from sklearn.model_selection import ParameterGrid
import os 
from dotenv import load_dotenv
load_dotenv()

# embedding generator object custom
emb_generator = EmbeddingsGenrator()

# load the data
X_train,X_test,y_train,y_test = get_data_with_images()

# compute class weights
weights = compute_class_weight(
    class_weight='balanced',
    classes=y_train.unique(),
    y=y_train.tolist()
)

class_weights:dict = dict(zip(y_train.unique(), weights))


def generate_embeddings_and_merge(X_train:pd.DataFrame,X_test:pd.DataFrame, pca:PCA):
    train_embeddings_reduced = pca.fit_transform(emb_generator.generate_embeddings(X_train))
    
    print(f"Generating embeddings with PCA _components:", pca.components_[0])
    test_embeddings_reduced = pca.transform(emb_generator.generate_embeddings(X_test))

    # embeddings columns for both train and test will be same
    emb_columns = [f'img_feat_{i}' for i in range(train_embeddings_reduced.shape[1])]
    
    train_embedding_df = pd.DataFrame(train_embeddings_reduced, columns=emb_columns)
    test_embedding_df = pd.DataFrame(test_embeddings_reduced, columns=emb_columns)
    
    # Printing the shape of the generated embeddings
    print(f"Train Embeddings Shape: {train_embedding_df.shape}")
    print(f"Test Embeddings Shape: {test_embedding_df.shape}")
        
    # return the final transformed dfs X_train and X_test
    return (
        pd.concat([X_train.reset_index(drop=True), train_embedding_df], axis=1),
        pd.concat([X_test.reset_index(drop=True), test_embedding_df], axis=1)
    )



# =============================================================================
# 3. Train CatBoost with Image Features (and other features)
# =============================================================================



# Define parameter grids
pca_param_grid = {'n_components': [128, 256, 512,1024]}
catboost_param_grid = {
    'iterations': [300,500, 1000],
    'depth': [6, 8, 10]
}

# Create parameter combinations
param_grid = list(ParameterGrid({
    'pca__n_components': pca_param_grid['n_components'],
    'catboost__iterations': catboost_param_grid['iterations'],
    'catboost__depth': catboost_param_grid['depth']
}))

# Start MLflow experiment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("catboost_experiment_with_images")

for params in param_grid:
    with mlflow.start_run() as run:
        # Log PCA parameters
        n_components = params['pca__n_components']
        pca = PCA(n_components=n_components)
        mlflow.log_param("pca_n_components", n_components)

        # Log CatBoost parameters
        iterations = params['catboost__iterations']
        depth = params['catboost__depth']
        mlflow.log_param("catboost_iterations", iterations)
        mlflow.log_param("catboost_depth", depth)

        # generate train and test dataframes based on pca values
        final_train_df,test_df = generate_embeddings_and_merge(X_train,X_test,pca)

        # Create a CatBoost Pool.
        # Note: We’re specifying 'brandName' as a categorical feature and 'name' as text,
        # while the image features are numeric.
        train_pool = Pool(
            data=final_train_df.drop(columns=['imageId']),
            label=y_train,
            cat_features=['brandName'],
            text_features=['name']
        )
        test_pool = Pool(
            data=test_df.drop(columns=['imageId']),
            cat_features=['brandName'],
            text_features=['name']
        )
        # Initialize CatBoostClassifier
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            task_type='GPU',
            devices='0',
            verbose=10,
            early_stopping_rounds=20,
            class_weights=class_weights
        )

        # Train the model
        model.fit(train_pool)

        # Predict and evaluate
        y_preds = model.predict(test_pool)
        b_accuracy_score = balanced_accuracy_score(y_test, y_preds)
        accuracy = accuracy_score(y_test, y_preds)
        
        # Log metrics
        mlflow.log_metric("balanced_accuracy", b_accuracy_score)
        mlflow.log_metric("accuracy", accuracy)

