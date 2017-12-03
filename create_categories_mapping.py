from tools import reader
from tqdm import tqdm
import json

if __name__ == "__main__":
    category_name_to_id = {}
    for item in tqdm(reader.read_annotations()):
        for obj in item.objects:
            if obj.cls not in category_name_to_id:
                category_name_to_id[obj.cls] = len(category_name_to_id)

    with open("cat2id.json", "w") as f:
        json.dump(category_name_to_id, f)