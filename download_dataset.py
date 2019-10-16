import os
import json
import urllib.request

# path to images of bills
BILL_IMG_DIR = "bills"
FULL_PATH = os.path.join(os.getcwd(), BILL_IMG_DIR)

with open("labelbox_export.json", "r") as export_file:
    data = json.load(export_file)
    data = [label for label in data if label["Label"] != "Skip"]
    for label in data:
        urllib.request.urlretrieve(label["Labeled Data"], os.path.join(FULL_PATH, label["External ID"]))
