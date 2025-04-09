"""
Script to fetch product data from Qogita API and save it to a JSON file.
"""
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QOGITA_API_URL = "https://api.qogita.com"

SIZE = 500
pages = [10, 11, 12]
all_results = [] 

# Loop through the specified pages and fetch data from each one
for PAGE in pages:
    print(f"Fetching data from page {PAGE}...")
    response = requests.get(
        url=f"{QOGITA_API_URL}/variants/search/?"
            f"&stock_availability=in_stock"
            f"&page={PAGE}"
            f"&size={SIZE}"
    ).json()

    # Get the data from the response (assuming the key is 'results')
    page_results = response.get('results', [])
    all_results.extend(page_results)

# Save the combined results to a JSON file
with open("data/raw/products.json", "w") as f:
    json.dump(all_results, f, indent=4)

