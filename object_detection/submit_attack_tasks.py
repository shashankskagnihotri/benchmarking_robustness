from pathlib import Path
import submitit
from adv_attack import run_attack_val, pgd_attack, fgsm_attack, bim_attack
from collect_attack_results import collect_results
from tqdm import tqdm
import logging
from rich.logging import RichHandler
import os
import itertools
from dotenv import load_dotenv

from misc import find_latest_epoch_file, find_python_files, format_value

load_dotenv()  # Load environment variables from .env file
my_email = os.getenv("MY_EMAIL")

# Set up the logging configuration to use RichHandler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",  # Custom date format
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Create a logger
logger = logging.getLogger("rich")
WORK_DIR = "slurm/logs/attacks/%j"
TARGETED = False
RANDOM_START = False
RESULT_DIR = "slurm/results"
MODEL_DIR = "models"
ATTACKS = {"PGD": pgd_attack, "FGSM": fgsm_attack, "BIM": bim_attack, "none": "none"}
STEPS_ATTACK = {
    "PGD": [1, 20],  # [1, 5, 10, 20]
    "FGSM": [None],
    "BIM": [1, 20],
    "none": [1],
}
EPSILONS = {
    "PGD": [8],  # [1, 2, 4, 8]
    "FGSM": [8],
    "BIM": [8],
    "none": [1],
}
ALPHAS = {
    "PGD": [0.01],
    "FGSM": [0.01],
    "BIM": [0.01],
    "none": [1],
}
NORMS = {
    "PGD": [None],
    "FGSM": ["inf"],
    "BIM": ["inf"],
    "none": [1],
}

logger.debug("Starting attack tasks")
logger.debug(f"WORK_DIR: {WORK_DIR}")
logger.debug(f"RESULT_DIR: {RESULT_DIR}")
logger.debug(f"TARGETED: {TARGETED}")
logger.debug(f"RANDOM_START: {RANDOM_START}")
logger.debug(f"MODEL_DIR: {MODEL_DIR}")
logger.debug(f"ATTACKS: {ATTACKS}")
logger.debug(f"STEPS_ATTACK: {STEPS_ATTACK}")
logger.debug(f"EPSILONS: {EPSILONS}")
logger.debug(f"ALPHAS: {ALPHAS}")


def submit_attack(config_file, checkpoint_file, attack, attack_kwargs, result_dir):
    job = executor.submit(
        run_attack_val,
        attack,
        config_file,
        checkpoint_file,
        attack_kwargs,
        result_dir,
    )
    return job


checkpoint_files = []
config_files = []

for subdir in Path(MODEL_DIR).iterdir():
    if subdir.is_dir():
        checkpoint_file = str(find_latest_epoch_file(subdir))
        config_file = str(find_python_files(subdir))

        if checkpoint_file != "None" and config_file != "None":
            logger.info(f"checkpoint file: {checkpoint_file}")
            logger.info(f"config file: {config_file}")
            checkpoint_files.append(checkpoint_file)
            config_files.append(config_file)
        else:
            logger.warning(f"No checkpoint or config file found in {subdir}")

logger.debug("setup submitit executor")
logger.info(f"found {len(config_files)} config and checkpoint files")
# num_tasks = min(40, len(config_files))
num_tasks = 1
slurm_mem = 10_000  # 10GB per task
# logger.debug(f"submitting {len(config_files)} tasks")
logger.debug(f"slurm_mem: {slurm_mem}")

executor = submitit.AutoExecutor(folder=WORK_DIR)
executor.update_parameters(
    slurm_partition="gpu_4",
    slurm_gres="gpu:1",
    nodes=1,
    cpus_per_task=1,
    tasks_per_node=num_tasks,
    slurm_mem=slurm_mem,
    slurm_mail_type="all",
)
jobs = []

for attack_name, attack in ATTACKS.items():
    num_steps = STEPS_ATTACK[attack_name]
    epsilons = EPSILONS[attack_name]
    alphas = ALPHAS[attack_name]
    norms = NORMS[attack_name]

    for steps, epsilon, alpha, norm in itertools.product(
        num_steps, epsilons, alphas, norms
    ):
        slurm_time = f"{90 if steps is None else 90*steps}:00"
        executor.update_parameters(slurm_time=slurm_time)

        with executor.batch():
            for config_file, checkpoint_file in zip(config_files, checkpoint_files):
                attack_kwargs = {
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "targeted": TARGETED,
                    "steps": steps,
                    "random_start": RANDOM_START,
                    "norm": norm,
                }

                if attack == fgsm_attack:
                    del attack_kwargs["steps"]
                    del attack_kwargs["random_start"]
                elif attack == pgd_attack:
                    del attack_kwargs["norm"]
                elif attack == bim_attack:
                    del attack_kwargs["random_start"]
                elif attack == "none":
                    attack_kwargs = {}

                logger.debug(str(config_file))
                logger.debug(str(checkpoint_file))
                model_name = str(config_file).split("/")[-1][0:-3]
                result_dir = os.path.join(
                    f"{RESULT_DIR}/"
                    + f"{model_name}/"
                    + f"{attack.__name__ if attack != 'none' else 'none'}_"
                    + "_".join([k + format_value(v) for k, v in attack_kwargs.items()])
                )

                if os.path.exists(result_dir):
                    logger.info(f"skipping {result_dir} as it already exists")
                else:
                    logger.info(
                        f"running attack {attack.__name__ if attack != 'none' else 'none'} with {attack_kwargs}"
                    )
                    logger.info(f"saving results to {result_dir}")
                    job = submit_attack(
                        config_file,
                        checkpoint_file,
                        attack,
                        attack_kwargs,
                        result_dir,
                    )
                    jobs.append(job)

# wait until all jobs are completed:
outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]

# change notification and no need for a gpu
executor.update_parameters(
    slurm_partition="single",
    slurm_time="01:00:00",
    slurm_gres="",
    slurm_mail_type="END, FAIL",
    slurm_mail_user=my_email,
)

executor.submit(collect_results, RESULT_DIR)
