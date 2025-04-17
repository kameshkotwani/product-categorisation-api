import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from typing import Optional
from catboost import CatBoostClassifier, Pool
from consts import IMAGES_DIR

from exceptions_ import EmbeddingGenerationError
from sklearn.decomposition import PCA

class EmbeddingsGenrator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define image transforms – these must match what the pre-trained network expects.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),   # ResNet50 expects 224x224 inputs
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Load the pre-trained ResNet50 model and remove the final classification layer.
        self.model_resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # this is removing the classification layer to further use transfer learning
        self.model_resnet.fc = torch.nn.Identity()  

        self.model_resnet = self.model_resnet.to(self.device)

        self.model_resnet.eval()  # Set model to evaluation mode
        self.pca= None


    def __extract_embedding(self,image_path, model, transform, device):
        """
        Given the path to an image, returns the embedding vector using the provided model.
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(e)

        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            embedding = model(image_tensor)
        # Return a flattened numpy array (e.g., dimension 2048 for ResNet50)
        return embedding.cpu().numpy().flatten()




    def __get_image_path(self,image_index):
        # This helper constructs the file path for the image corresponding to a row.
        return os.path.join(IMAGES_DIR, f'{image_index}.jpg')

    def generate_embeddings(self,df:pd.DataFrame)-> np.ndarray:
        """
        Given a DataFrame with image IDs, generates and returns the embeddings for all images.
        The DataFrame should have a column named 'imageId'.
        
        args:
            df (pd.DataFrame): DataFrame containing image IDs.
        returns:
            np.ndarray: Array of image embeddings.
        """
        # Looping  X_train to extract the embedding for each image.
        embeddings = []
        for _, row in df.iterrows():
            img_path = self.__get_image_path(row['imageId'])
            emb = self.__extract_embedding(img_path, self.model_resnet, self.transform, self.device)
            if emb is None:
                raise EmbeddingGenerationError(f"Failed to generate embedding for image {img_path}")
            embeddings.append(emb)

        # Convert the list of embeddings to a numpy array.
        return np.array(embeddings)


    def train_pca(self,embeddings: np.ndarray, n_components: Optional[int] = 128):
        """
        Trains PCA on the given embeddings and returns the fitted PCA object.
        """
        self.pca = PCA(n_components=n_components)
        self.pca = self.pca.fit(embeddings)

    def generate_reduced_embeddings(self,embeddings: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensionality of the embeddings using PCA.
        
        args:
            embeddings (np.ndarray): Original image embeddings.
            n_components (int): Number of dimensions to reduce to.
            
        returns:
            np.ndarray: Reduced dimensionality embeddings.
        """
        # check if the pca is fitted or not
        if not hasattr(self.pca,"components_"):
            raise ValueError("PCA is not fitted, please run train_pca to fit first")
        
        
        return self.pca.transform(embeddings)
        
        