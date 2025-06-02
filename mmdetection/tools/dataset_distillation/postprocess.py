import json, os
import argparse
import pycocotools.mask as maskUtils
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Postprocess the annotation json file')
    parser.add_argument('json_path', type=str, help='The path of the json file to preprocess')
    parser.add_argument('soft_label', type=str, help='The path of the soft label file')
    parser.add_argument('original_path', type=str, help='The path to the json file from the original dataset')
    parser.add_argument('out', type=str, help='The output annotation json file name')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    json_path = args.json_path
    soft_label = args.soft_label
    original_path = args.original_path
    output = args.out

    with open(json_path, 'r') as json_file2:
        total_contents = json.load(json_file2)

        with open(soft_label, 'r') as json_file:
            contents = json.load(json_file)
            obj_count = 0
            for i in tqdm(range(len(contents))):
                contents[i]["id"] = obj_count
                contents[i]["area"] = contents[i]["bbox"][2] * contents[i]["bbox"][3]
                obj_count += 1
            total_contents["annotations"] = contents

        with open(original_path, 'r') as original:
            contents = json.load(original)
            total_contents["categories"] = contents["categories"]

        s = json.dumps(total_contents)
        ff = open(output, "w")
        ff.write(s)
        ff.close()