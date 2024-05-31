import wandb
import pandas as pd
from dotenv import load_dotenv
import os


def extract_wandb_runs_to_csv(
    project_name: str, entity_name: str, output_file: str
) -> None:
    """
    Extracts run data from a Weights & Biases project and saves it to a CSV file.

    Parameters:
    - project_name (str): The name of the Weights & Biases project.
    - entity_name (str): The Weights & Biases username or team name.
    - output_file (str): The file path where the CSV will be saved.
    """
    # Initialize a wandb API instance
    api = wandb.Api()

    # Get all runs in the specified project
    runs = api.runs(f"{entity_name}/{project_name}")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        # We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    # Create dataframes from the lists
    summary_df = pd.json_normalize(summary_list)
    config_df = pd.json_normalize(config_list)

    # Combine dataframes along with the run names
    runs_df = pd.concat([summary_df, config_df], axis=1)
    runs_df["name"] = name_list

    # Save to CSV
    runs_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Example usage
    load_dotenv()  # Load environment variables from .env file
    project_name = os.getenv("WANDB_PROJECT")
    entity_name = os.getenv("WANDB_ENTITY")
    assert project_name, "Please set the WANDB_PROJECT environment variable"
    assert entity_name, "Please set the WANDB_ENTITY environment variable"
    output_file = "project.csv"
    extract_wandb_runs_to_csv(project_name, entity_name, output_file)
