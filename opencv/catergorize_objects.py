import csv

# Load categories from the CSV file
def load_categories(csv_file="object_categories.csv"):
    categories = {"biodegradable": [], "recyclable": [], "hazardous": []}
    
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            object_name = row["Object Name"]
            if row["Biodegradable"] == "1":
                categories["biodegradable"].append(object_name)
            if row["Recyclable"] == "1":
                categories["recyclable"].append(object_name)
            if row["Hazardous"] == "1":
                categories["hazardous"].append(object_name)
    
    return categories

# Determine the category of an object
def get_category(label, categories):
    if label in categories["biodegradable"]:
        return "Biodegradable", (0, 255, 0)  # Green
    elif label in categories["recyclable"]:
        return "Recyclable", (255, 0, 0)  # Blue
    elif label in categories["hazardous"]:
        return "Hazardous", (0, 0, 255)  # Red
    else:
        return None, None
