import csv

# Load coco.names
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Categorization logic
biodegradable = ["apple", "banana", "carrot"]  # Extend this as needed
recyclable = ["bottle", "can", "glass"]
hazardous = ["battery", "lighter", "chemical"]
none = []  # To auto-classify leftover items

# Create CSV structure
rows = []
for cls in classes:
    if cls in biodegradable:
        rows.append([cls, "biodegradable", "", "", ""])
    elif cls in recyclable:
        rows.append([cls, "", "recyclable", "", ""])
    elif cls in hazardous:
        rows.append([cls, "", "", "hazardous", ""])
    else:
        rows.append([cls, "", "", "", "none"])

# Write to a CSV file
with open("object_categories.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Object Name", "Biodegradable", "Recyclable", "Hazardous", "None"])
    writer.writerows(rows)
