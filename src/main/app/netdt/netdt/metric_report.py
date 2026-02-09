import sys, os
from pymongo import MongoClient
import datetime

def update_training_results_SL(total_epochs_stage1: int,
                               epoch_stage1: int,
                               training_loss_stage1: list,
                               total_epochs_stage2: int,
                               epoch_stage2: int,
                               training_loss_stage2: list
                               ):
    """
    Update training results to MongoDB for supervised learning tasks.
    """
    # Connect to MongoDB and query the current running task
    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
    if mongodb_url is None:
        print("[ERROR CODE 300: FLOWER_INTERNAL_ERROR] AITRCOMMONDB_URI environment variable not set. Cannot connect to MongoDB.")
        return
    client = MongoClient(mongodb_url)
    database = client["TrainingConfig"]
    collection = database["current_task"]
    item = collection.find_one({
        "status": "running"
    })
    project_id = item["project_id"]
    app_name = item["app_name"]
    model_name = item["model_name"]
    model_version = item["model_version"]
    mode = item["mode"]
    dataset_name = item["dataset_name"]

    # Connect to MongoDB and save training results
    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
    if mongodb_url is None:
        print("[ERROR CODE 300: FLOWER_INTERNAL_ERROR] AITRCOMMONDB_URI environment variable not set. Cannot connect to MongoDB.")
        return
    client = MongoClient(mongodb_url)
    database = client["TrainingResults"]
    collection_name = f"{project_id}_{app_name}_{model_name}_{model_version}_{mode}_{dataset_name}"
    collection = database[collection_name]

    filter_data = {
        "project_id": project_id,
        "app_name": app_name,
        "model_name": model_name,
        "model_version": model_version,
        "mode": mode,
        "dataset_name": dataset_name,
    }

    record = {
        "total_epochs_stage1": total_epochs_stage1,
        "current_epoch_stage1": epoch_stage1,
        "training_loss_stage1": training_loss_stage1,
        "total_epochs_stage2": total_epochs_stage2,
        "current_epoch_stage2": epoch_stage2,
        "training_loss_stage2": training_loss_stage2,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    collection.update_one(filter_data, {"$set": record}, upsert=True)