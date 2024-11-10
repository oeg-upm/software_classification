import os
import json
from pymongo import MongoClient

# MongoDB connection details
mongo_url = 'mongodb://localhost:27017/'  # Update if necessary
db_name = 'msr2025'
collection_name = 'datacite_software'

# Specify the directory path
directory_path = 'datacite_software_results/'

# List all files in the directory
files = os.listdir(directory_path)

# Connect to MongoDB
client = MongoClient(mongo_url)

# Select the database and collection
db = client[db_name]
collection = db[collection_name]

# Iterate over the files and print their names
for file_name in files:
    print(file_name)
    # Open and read the JSON file
    with open(directory_path+file_name, 'r') as file:
        data = json.load(file)  # Parse the JSON file into a Python dictionary
        result = collection.insert_many(data["data"])
        print(f'Inserted documents with ids: {result.inserted_ids}')
        break
    
# Close the MongoDB connection
client.close()