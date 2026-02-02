import sys, os
from pymongo import MongoClient
import queue
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# print(sys.path)
# print("a: ", os.path.dirname(__file__))
# print("b: ", os.path.join(os.path.dirname(__file__), '../src'))
# print("c: ", os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Connect to MongoDB and save training parameters
# mongodb_url = os.environ.get("MONGODB_URL", "mongodb://mongodb:27017")
# client = MongoClient(mongodb_url)
# database = client["Training_Configuration"]
# collection = database["training_conf"]

# response = collection.find_one({"app_name": "test_app"})
# print("Query Result:")
# print(response)