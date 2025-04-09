"""
Script to clean the dataset by removing rows with missing images. and save in processed dir
"""
import os
import pandas as pd
from consts import INTERIM_DATA_DIR,PROCESSED_DATA_DIR
# Define the paths
json_file = INTERIM_DATA_DIR / "final_1500_rows.json"  
images_folder = INTERIM_DATA_DIR / "images_1500"   

# Read the JSON file into a DataFrame
df = pd.read_json(json_file)

columns = ['name', 'brandName', 'imageUrl', 'categoryName']
df = df[columns]

missing_indices = []

# Loop through each row in the DataFrame
for idx in df.index:
    image_filename = f'{idx}.jpg'
    image_path = os.path.join(images_folder, image_filename)
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        missing_indices.append(idx)

# Print the list of indices whose images are missing
print("Indices with missing images:", missing_indices,len(missing_indices))

# Drop the rows corresponding to the missing image files from the DataFrame
df_cleaned = df.drop(index=missing_indices)

print(df_cleaned.index.to_list())
# Save the cleaned DataFrame to a new CSV file

cleaned_csv_file = 'cleaned_results_1.csv'
df_cleaned.to_csv(PROCESSED_DATA_DIR / cleaned_csv_file,index_label='imageId')
print(f"Cleaned data saved to {cleaned_csv_file}")

# Optionally, show basic information about the cleaned DataFrame
print("Cleaned DataFrame shape:", df_cleaned.shape)
