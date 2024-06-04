import json


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r") as file:
        return json.load(file)


def get_file_names(json_data):
    """Extract file names from JSON data."""
    file_names = set()

    def extract_file_names(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "file_name":
                    file_names.add(value)
                extract_file_names(value)
        elif isinstance(data, list):
            for item in data:
                extract_file_names(item)

    extract_file_names(json_data)
    return file_names


def find_common_file_names(json1, json2):
    """Find common file names between two JSON data."""
    file_names1 = get_file_names(json1)
    file_names2 = get_file_names(json2)
    return file_names1.intersection(file_names2)


if __name__ == "__main__":
    # Replace 'file1.json' and 'file2.json' with the paths to your JSON files
    json1 = load_json("data/VOCdevkit/annotations/voc0712_trainval.json")
    json2 = load_json("data/VOCdevkit/annotations/voc07_test.json")

    common_file_names = find_common_file_names(json1, json2)

    if common_file_names:
        print("Common file names found:")
        for name in common_file_names:
            print(name)
    else:
        print("No common file names found.")
