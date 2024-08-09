import os
import importlib.util
import sys
import json

paths = {
    "ade": {
        "SegFormer": {
            "MIT-B0": {
                "path": "../aa_workdir/ade/MIT-B0",
                "skip": ["20240713_125207", "20240713_130933"]
            },
            "MIT-B1": {
                "path": "../aa_workdir/ade/MIT-B1",
                "skip": []
            },
            "MIT-B2": {
                "path": "../aa_workdir/ade/MIT-B2",
                "skip": []
            },
            "MIT-B3": {
                "path": "../aa_workdir/ade/MIT-B3",
                "skip": []
            },
            "MIT-B4": {
                "path": "../aa_workdir/ade/MIT-B4",
                "skip": []
            },
            "MIT-B5": {
                "path": "../aa_workdir/ade/MIT-B5",
                "skip": []
            },
        }
    }
}

def import_config_variables(file_path):

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # dynamically load module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # all variables not starting with '__' are set in the config file
    variables = {name: value for name, value in vars(module).items() if not name.startswith('__')}

    return variables

def json_to_dict(file_path):
    with open(file_path) as fp:
        return_dict = json.load(fp)
    return return_dict

def recursive_get_attacks(paths, add_whole_config = False):
    new_dict = dict()
    
    for k, v in paths.items():
        print(k)
        if not "path" in v:
            new_dict[k] = recursive_get_attacks(v)
        else:
            run_dirs = [f for f in os.listdir(v["path"]) if os.path.isdir(os.path.join(v["path"], f))]
            return_list = list()
            for run_dir in run_dirs:
                if run_dir in v["skip"]:
                    continue
                run_dir_path = os.path.join(v["path"], run_dir)
                config = import_config_variables(os.path.join(run_dir_path, "vis_data/config.py"))
                try:
                    test_result = json_to_dict(os.path.join(run_dir_path, os.path.basename(run_dir_path)+".json"))
                except FileNotFoundError:
                    print(f"No results for {run_dir_path}")
                if add_whole_config:
                    combined_dict = config | test_result
                else:
                    attack_cfg = config["model"]["attack_cfg"]
                    combined_dict = attack_cfg | test_result
                return_list.append(combined_dict)
            new_dict[k] = return_list
        
    return new_dict

attack_results = recursive_get_attacks(paths)

with open("../attack_results.json", "w") as fp:
    json.dump(attack_results, fp)



