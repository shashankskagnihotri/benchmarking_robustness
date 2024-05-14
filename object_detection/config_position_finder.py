import os


print(f"Number of configs to train: {len(os.listdir('./configs_to_train'))}")
print(f"Number of verified, correct configs: {len(os.listdir('./configs_verified'))}")
print(
    f"Number of erroneous configs: {len(os.listdir('./configs_erroneous/verification'))}"
)


original_config_files = os.listdir("./configs_to_train")
original_config_files
for i in range(len(original_config_files)):
    if "retinanet" in original_config_files[i]:
        print(f"original_config_files[{i}], # {original_config_files[i]}")
    if "ddod" in original_config_files[i]:
        print(f"original_config_files[{i}], # {original_config_files[i]}")
