import os
import pandas as pd
import requests
import shutil
from consts import INTERIM_DATA_DIR
def check_images(df, url_column, download_folder=INTERIM_DATA_DIR / "images_1500"):
    """
    For each image URL in df[url_column], try to download it.
    If download succeeds, save it in download_folder.
    If it fails, record that URL in the failed list.
    Returns the list of failed URLs.
    """
    
# Create the directory if it doesn't exist
    if  os.path.exists(INTERIM_DATA_DIR / "images_1500"):
        shutil.rmtree(INTERIM_DATA_DIR / "images_1500")

    os.makedirs(INTERIM_DATA_DIR / "images_1500")
        
    failed_urls = []

    # Iterate over the URLs
    for idx, url in enumerate(df[url_column]):
        try:
            # Attempt to download the image
            response = requests.get(url, timeout=10)
            # Raise an error if the request failed (e.g., 404 or 500)
            response.raise_for_status()
            
            # Create a filename (e.g., using the index)
            # Alternatively, you can parse the actual filename from the URL
            filename = f'{idx}.jpg'
            filepath = os.path.join(download_folder, filename)
            
            # Write the file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
        except Exception as e:
            failed_urls.append(url)

    return failed_urls

if __name__ == '__main__':
    # Example usage:
    # 1. Load your DataFrame 
    df = pd.read_json(INTERIM_DATA_DIR / "final_1500_rows.json")
    
    # 2. Run the check_images function
    failed_urls = check_images(df, url_column='imageUrl')
    
    # 3. Print results
    print(f"Total URLs: {len(df)}")
    print(f"Failed downloads: {len(failed_urls)}")
    print("These URLs could not be downloaded:")
    for u in failed_urls:
        print(u)


