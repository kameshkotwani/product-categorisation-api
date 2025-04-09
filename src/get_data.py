"""
Script to get the data to train models
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from consts import PROCESSED_DATA_DIR
# there are total of 1451 rows in dataset, I have removed 49 rows for which image was not downloadable
def get_data_without_images()->list:
    df = pd.read_csv(PROCESSED_DATA_DIR / 'cleaned_results.csv')
    X = df[['name','brandName']]
    Y = df['categoryName']

    # Cannot use Statrify here since there are categories with only 1 value, so it's not possible to split them
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def get_data_with_images()->list:
    df = pd.read_csv(PROCESSED_DATA_DIR / 'cleaned_results_1.csv')
    X = df[['name','brandName','imageId']]
    Y = df['categoryName']

    # Cannot use Statrify here since there are categories with only 1 value, so it's not possible to split them
    return train_test_split(X, Y, test_size=0.2, random_state=42)