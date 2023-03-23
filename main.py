import os
from detect_people.detect_people import HumanDetector
from dotenv import load_dotenv
from time import sleep
from firebase_admin import firestore, initialize_app

# CONSTANTS
NODES_COLLECTION = "nodes"
LOGS_COLLECTION = "logs"
NODE_ID = "0" # os.getenv("NODE_ID")
WAIT_TIME = 20

load_dotenv()
app = initialize_app()
db = firestore.client()

node_doc_ref = db.collection(NODES_COLLECTION).document(NODE_ID)
node_doc = node_doc_ref.get()
node_logs_ref = node_doc_ref.collection(LOGS_COLLECTION)

node_info = node_doc.to_dict()

def get_node_info():
    node_doc = node_doc_ref.get()
    node_data = node_doc.to_dict()
    return node_data


# endless script to detect people and upload results
# as logs to firebase
# keep doing this every WAIT_TIME seconds
# stop if get_node_info().get("awake") is False
while True:
    node_data = get_node_info()
    if not node_data.get("awake"):
        print("node is asleep")
        break

    detector = HumanDetector(write_video=True, display_box=True)
    detected = detector.detect_people_timedelta()
    node_logs_ref.add({"people":detected
                       , "created_at":firestore.SERVER_TIMESTAMP}) # type: ignore
    print("done")
    sleep(WAIT_TIME)

print("exiting")

# TODO: Use on_snapshot for awake value
