import os


original_config_files = os.listdir("./configs_to_train")
original_config_files
for i in range(len(original_config_files)):
    if "retinanet" in original_config_files[i]:
        print(f"original_config_files[{i}], # {original_config_files[i]}")
    if "ddod" in original_config_files[i]:
        print(f"original_config_files[{i}], # {original_config_files[i]}")
