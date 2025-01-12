import requests
import json
from PIL import Image

API_KEY = "sk-55c7a86c9b0e4ef783f5c1b91cef4ccb"  # Your API key from WorqHat
BASE_URL = "https://api.worqhat.com"

# Function to make the API request
def make_request(endpoint, files, data):
    url = f"{BASE_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # Authentication header
        "Content-Type": "multipart/form-data",  # Adjusted for file upload
    }
    try:
        response = requests.post(url, headers=headers, files=files, data=data)  # Use multipart/form-data
        response.raise_for_status()  # Check for HTTP errors (4xx, 5xx)
        return response.json()  # Return the JSON response
    except requests.RequestException as e:
        print(f"Error making request: {e}")
        if response.content:
            print(f"Response Content: {response.content.decode('utf-8')}")
        return None

# Quickstart example: Classify trash in the image
if __name__ == "__main__":
    # Open and validate the image
    try:
        with Image.open("image.png") as img:
            print("Image loaded successfully.")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

    endpoint = "api/ai/images/v2/image-analysis"
    
    data = {
        "output_type": "classify",
        "training_data": "Classify trash into types: plastic, metal, organic, paper, glass.",
        "question": "What type of trash is this image showing?",
        "randomness": "0.0",  
        "stream_data": "true"  
    }
    
    with open("image.png", "rb") as image_file:
        files = {
            "images": image_file  
        }

        response_data = make_request(endpoint, files, data)

    if response_data:
        print("Response from WorqHat API:")
        print(json.dumps(response_data, indent=2))
    else:
        print("Failed to retrieve data.")
