from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Dict
from contextlib import asynccontextmanager
import catboost
import pandas as pd
import uvicorn
# load the model
model = catboost.CatBoostClassifier()
model.load_model(fname="model/catboost_model_without_image.cbm")


@asynccontextmanager
async def lifespan(app: FastAPI):

    if model is None:
        raise RuntimeError("Failed to load model or vectorizer")

    print("Model loaded")

    # Yield control back to FastAPI
    yield
    # Cleanup logic (if needed) when the app shuts down
    print("Shutting down application...")

# Creating a FastAPI instance
app = FastAPI(lifespan=lifespan)


class ProductInput(BaseModel):
    name: str = Field(...,description="The name of the product to classify")
    brandName: str = Field(...,description="The brand name of the product to classify")

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the Product Classification API"}


@app.post("/classify")
async def classify_product(product: ProductInput):
    # Create a DataFrame that aligns with how your model expects the input
    input_df = pd.DataFrame({
        "name": [product.name],
        "brandName": [product.brandName]
    },index=[0])
    
    # The model is expected to have been trained with 'name' as text and 
    # 'brandName' as a categorical feature.
    prediction = model.predict(input_df)
    category = prediction[0][0]  # prediction is typically returned as an array
    
    # Return the predicted category as JSON
    return {"product":product.name,"prediction": category}


# async def classify(request: ClassificationRequest) -> Dict[str, str]:
#     request
#     product_category = "unknown"  # Replace with actual classification logic
#     return {"product_category": product_category}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)