from pymongo import MongoClient, errors
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MongoDBHandler:
    def __init__(self, uri):
        try:
            self.client = MongoClient(uri)
            logging.info("MongoDB connection established.")
        except errors.ConnectionFailure:
            logging.error("Failed to connect to MongoDB.")
            raise

    def insert_records(self, db_name, collection_name, records, avoid_duplicates=False):
        if not records:  # Validate records is not empty
            logging.warning("No records provided to insert.")
            return
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            if avoid_duplicates:
                inserted_count = 0
                for record in records:
                    if collection.count_documents({"unique_identifier": record.get("unique_identifier")}, limit=1) == 0:
                        collection.insert_one(record)
                        inserted_count += 1
                logging.info(f"{inserted_count} records inserted successfully into MongoDB.")
            else:
                result = collection.insert_many(records)
                logging.info(f"{len(result.inserted_ids)} records inserted successfully into MongoDB.")
        except Exception as e:
            logging.error(f"Failed to insert records: {e}")

    def count_records(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            count = collection.count_documents({})
            return count
        except Exception as e:
            logging.error(f"Error counting records: {e}")
            return None

    def get_last_upload_timestamp(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            last_record = collection.find_one(sort=[('_id', -1)])
            if last_record:
                return last_record.get("Accepted Date")
            return None
        except Exception as e:
            logging.error(f"Error retrieving last upload timestamp: {e}")
            return None

    def update_record(self, db_name, collection_name, filter_query, update_query):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            result = collection.update_one(filter_query, {"$set": update_query})
            if result.modified_count > 0:
                logging.info("Record updated successfully.")
            else:
                logging.warning("No records updated.")
        except Exception as e:
            logging.error(f"Failed to update record: {e}")

    def truncate_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            collection.delete_many({})
            logging.info("Collection truncated successfully.")
        except Exception as e:
            logging.error(f"Failed to truncate collection: {e}")

    def read_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            return list(collection.find())
        except Exception as e:
            logging.error(f"Failed to read collection: {e}")
            return []

    def generate_descriptive_statistics(self, db_name, collection_name):
        # Implementation depends on the data and required statistics
        pass

    def drop_duplicates_and_rewrite(self, db_name, collection_name, unique_key):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            data = list(collection.find())
            unique_records = {}
            for record in data:
                key = record.get(unique_key)
                if key and key not in unique_records:
                    unique_records[key] = record
            
            if unique_records:
                collection.delete_many({})
                collection.insert_many(list(unique_records.values()))
                logging.info("Duplicates dropped and table rewritten successfully.")
            else:
                logging.warning("No unique records found to write.")
        except Exception as e:
            logging.error(f"Failed to drop duplicates and rewrite: {e}")


uri = "mongodb+srv://shaun:Chacko1234@finance.dklpskg.mongodb.net/?retryWrites=true&w=majority&appName=finance"
handler = MongoDBHandler(uri)

with open("sec_filings.json","r") as filing:
    records = json.load(filing)

handler.insert_records("finance_database", "sec_filings", records, avoid_duplicates=True)

print("Number of records:", handler.count_records("finance_database", "sec_filings"))

print("Last upload timestamp:", handler.get_last_upload_timestamp("finance_database", "sec_filings"))

filter_query = {"Form Type": "4"}
update_query = {"Filing Content": "Updated content"}
handler.update_record("finance_database", "sec_filings", filter_query, update_query)
handler.truncate_collection("finance_database", "sec_filings")

collection_data = handler.read_collection("finance_database", "sec_filings")
print(collection_data)

handler.generate_descriptive_statistics("finance_database", "sec_filings")
