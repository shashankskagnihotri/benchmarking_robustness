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

    results = []
    for run in runs:
        df_history = run.history()
        if len(df_history) == 0:
            continue
        config = run.config
        config.update({"name": run.name, "state": run.state, "id": run.id})
        df_config_repeated = pd.json_normalize([config] * len(df_history))
        df_combined = pd.concat([df_history, df_config_repeated], axis=1)

        results.append(df_combined)
        # print(df_combined)

    runs_df = pd.concat(results, ignore_index=True)
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
