import os
#Loads the .env file which contains the local environment
from dotenv import load_dotenv
#Used to connect Python to MongoDB
from pymongo import MongoClient

load_dotenv()
#Retrieves the MONGODB_URI variable from the local environment
MONGODB_URI=os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in the .venv file")
#This is now the interface between Python and MongoDB
client=MongoClient(MONGODB_URI)
#Selects the database called Food11
db=client["Food11"]
#Selects the foods collection
foods_collection=db["foods"]
'''This is for testing the connection between python and mongodb using client. A ping is sent to check whether or not MongoDB responds'''
def test_connection():
    try:
        client.admin.command("ping")
        print("MongoDB connection successful!!!")
        return True
    except Exception as e:
        print("MongoDB connection failed!!!!")
        return False
if __name__=="__main__":
    test_connection()