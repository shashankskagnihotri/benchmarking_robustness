# import os
# from mmengine.config import Config

# configs_to_train_folder = os.listdir("./configs_to_train")


# for file in configs_to_train_folder:
#     cfg = Config.fromfile(f"./configs_to_train/{file}")
#     print(f"Checking file: {file}")
#     if "param_scheduler" in cfg.train_cfg:
#         print(f"Found param_scheduler in file: {file}")
#         for param_scheduler in cfg.train_cfg.param_scheduler:
#             if param_scheduler.by_epoch:
#                 if "begin" in param_scheduler:
#                     param_scheduler.begin = param_scheduler.begin // 3

#                 if "end" in param_scheduler:
#                     print(f"old_end: {param_scheduler.end}")
#                     param_scheduler.end = param_scheduler.end // 3
#                     print(f"new_end: {param_scheduler.end}")

#                 if "milestones" in param_scheduler:
#                     print(f"old_milestones: {param_scheduler.milestones}")
#                     param_scheduler.milestones = [
#                         milestone // 3 for milestone in param_scheduler.milestones
#                     ]
#                     print(f"new_milestones: {param_scheduler.end}")
import os
from mmengine.config import Config


def adjust_param_scheduler(cfg, factor=3):
    if (
        hasattr(cfg, "model")
        and hasattr(cfg.model, "optim_wrapper")
        and hasattr(cfg.model.optim_wrapper, "param_scheduler")
    ):
        print("Found param_scheduler in configuration.")

        for param_scheduler in cfg.model.optim_wrapper.param_scheduler:
            if hasattr(param_scheduler, "by_epoch") and param_scheduler.by_epoch:
                if hasattr(param_scheduler, "begin"):
                    print(f"Old begin: {param_scheduler.begin}")
                    param_scheduler.begin = param_scheduler.begin // factor
                    print(f"New begin: {param_scheduler.begin}")

                if hasattr(param_scheduler, "end"):
                    print(f"Old end: {param_scheduler.end}")
                    param_scheduler.end = param_scheduler.end // factor
                    print(f"New end: {param_scheduler.end}")

                if hasattr(param_scheduler, "milestones"):
                    print(f"Old milestones: {param_scheduler.milestones}")
                    param_scheduler.milestones = [
                        milestone // factor for milestone in param_scheduler.milestones
                    ]
                    print(f"New milestones: {param_scheduler.milestones}")
    elif hasattr(cfg, "param_scheduler"):
        print("Found param_scheduler in configuration.")

        for param_scheduler in cfg.param_scheduler:
            if hasattr(param_scheduler, "by_epoch") and param_scheduler.by_epoch:
                if hasattr(param_scheduler, "begin"):
                    print(f"Old begin: {param_scheduler.begin}")
                    param_scheduler.begin = param_scheduler.begin // factor
                    print(f"New begin: {param_scheduler.begin}")

                if hasattr(param_scheduler, "end"):
                    print(f"Old end: {param_scheduler.end}")
                    param_scheduler.end = param_scheduler.end // factor
                    print(f"New end: {param_scheduler.end}")

                if hasattr(param_scheduler, "milestones"):
                    print(f"Old milestones: {param_scheduler.milestones}")
                    param_scheduler.milestones = [
                        milestone // factor for milestone in param_scheduler.milestones
                    ]
                    print(f"New milestones: {param_scheduler.milestones}")
    else:
        print("param_scheduler not found in configuration.")


# Iterate through the configuration files
configs_to_train_folder = os.listdir("./configs_to_train")

for file in configs_to_train_folder:
    if "Detic" in file:
        cfg = Config.fromfile(f"./configs_to_train/{file}")
        print(f"Checking file: {file}")

        # Debug: Print the structure of cfg to understand its hierarchy
        print(f"Config structure: {cfg.pretty_text}")

        adjust_param_scheduler(cfg)
