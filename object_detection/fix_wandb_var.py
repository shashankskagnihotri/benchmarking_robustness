import os

from dotenv import load_dotenv

import wandb

load_dotenv()
WAND_PROJECT = os.getenv("WANDB_PROJECT")
WAND_ENTITY = os.getenv("WANDB_ENTITY")
assert WAND_PROJECT, "Please set the WANDB_PROJECT environment variable"
assert WAND_ENTITY, "Please set the WANDB_ENTITY environment variable"

# Initialize wandb API
api = wandb.Api()

# Get all runs in the project
runs = api.runs(f"{WAND_ENTITY}/{WAND_PROJECT}")

# Iterate through all runs and duplicate the "Group" column
for run in runs:
    # Get the current config
    config = run.config

    config["model_name"] = run.group

    # Update the run with the new config
    run.config = config
    run.update()  # Save changes to wandb

    print(f"Updated run {run.name} with duplicated 'Group' column.")

print("Completed duplicating 'Group' column for all runs.")
