import json, os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the annotation json file')
    parser.add_argument('json_path', type=str, help='The path of the json file to preprocess')
    parser.add_argument('original_path', type=str, help='The path to the json file from the original dataset')
    parser.add_argument('out', type=str, help='The output annotation json file name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    original = args.original_path
    content = args.json_path
    output = args.out

    with open(original,'r') as json_file1:
        contents = json.load(json_file1)
        categories = contents["categories"]

    with open(content,'r') as json_file2:
        contents = json.load(json_file2)
        contents["categories"] = categories

    s = json.dumps(contents)
    ff = open(output,"w")
    ff.write(s)
    ff.close()
    print("[INFO] Preprocessing done!")