import os


configs_to_train_folder = os.listdir("./configs_to_train")

print(f"Number of configs to train: {len(configs_to_train_folder)}")

num_classes_counter = 0
num_classes_voc_counter = 0

num_classes_voc_changed = 0


list_of_voc_files_with_num_classes = []
list_of_voc_files_with_param_scheduler = []

num_classes_values_and_num_occurence = dict()


for file in configs_to_train_folder:
    with open(f"./configs_to_train/{file}", "r") as f:
        content = f.read()
        if "param_scheduler" in content and "voc" in file:
            list_of_voc_files_with_param_scheduler.append(file)

        if "num_classes" in content:
            num_classes_counter += 1
            if "voc" in file:
                num_classes_voc_counter += 1
                list_of_voc_files_with_num_classes.append(file)
                if (
                    "num_classes = 20" in content
                    or "num_classes=20" in content
                    or "num_classes =20" in content
                    or "num_classes= 20" in content
                ):
                    num_classes_voc_changed += 1
                else:
                    print(f"num_classes not set to 20 in file: {file}")


print(f"Number of configs with num_classes: {num_classes_counter}")
print(
    f"Number of configs with num_classes which are voc configs: {num_classes_voc_counter}"
)
print(
    f"Number of voc configs with num_classes which were now correctly set to 20: {num_classes_voc_changed}"
)


list_of_voc_files_with_num_classes.sort()

print(
    "List of voc files with num_classes:"
    + "\n"
    + "\n".join(list_of_voc_files_with_num_classes)
    + "\n"
    + "\n"
)

list_of_voc_files_with_param_scheduler.sort()

print(
    "List of voc files with param_scheduler:"
    + "\n"
    + "\n".join(list_of_voc_files_with_param_scheduler)
    + "\n"
    + "\n"
)


# Detic has old value of: 22047
