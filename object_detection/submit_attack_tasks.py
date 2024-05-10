from pathlib import Path
import submitit
from adv_attack import run_attack_val, pgd_attack, fgsm_attack, bim_attack
from collect_attack_results import collect_results
from tqdm import tqdm
import re
import logging
from rich.logging import RichHandler
import os
from submitit.core.utils import FailedJobError
from fractions import Fraction

# Set up the logging configuration to use RichHandler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",  # Custom date format
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Create a logger
logger = logging.getLogger("rich")
WORK_DIR = "slurm/work_dir/attacks/%j"
TARGETED = False
RANDOM_START = False
RESULT_DIR = "slurm/results"
MODEL_DIR = "models"
ATTACKS = {
    "PGD": pgd_attack,
    "FGSM": fgsm_attack,
    "BIM": bim_attack,
}
STEPS_ATTACK = {
    "PGD": [5, 10, 20],
    "FGSM": None,
    "BIM": [5, 10, 20],
}
EPSILONS = {
    "PGD": [1 / 255, 2 / 255, 4 / 255, 8 / 255],
    "FGSM": [1 / 255, 2 / 255, 4 / 255, 8 / 255],
    "BIM": [1 / 255, 2 / 255, 4 / 255, 8 / 255],
}
ALPHAS = {
    "PGD": [0.01],
    "FGSM": [0.01],
    "BIM": [0.01],
}
NORMS = {
    "PGD": [None],
    "FGSM": ["inf"],
    "BIM": ["inf"],
}


logger.info("Starting attack tasks")
logger.info(f"WORK_DIR: {WORK_DIR}")
logger.info(f"RESULT_DIR: {RESULT_DIR}")
logger.info(f"TARGETED: {TARGETED}")
logger.info(f"RANDOM_START: {RANDOM_START}")
logger.info(f"MODEL_DIR: {MODEL_DIR}")
logger.info(f"ATTACKS: {ATTACKS}")
logger.info(f"STEPS_ATTACK: {STEPS_ATTACK}")
logger.info(f"EPSILONS: {EPSILONS}")
logger.info(f"ALPHAS: {ALPHAS}")


def find_latest_epoch_file(directory):
    root_dir = Path(directory)
    max_num = -1
    latest_file = None

    # Regex to match files of the form 'epoch_{n}.pth' where {n} is an integer
    pattern = re.compile(r"^epoch_(\d+)\.pth$")

    for file in root_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            current_num = int(match.group(1))
            if current_num > max_num:
                max_num = current_num
                latest_file = file

    return latest_file


def find_python_files(directory):
    python_files = list(directory.glob("*.py"))  # Glob for Python files only

    if python_files:
        assert len(python_files) == 1, f"Multiple Python files found in {directory}"
        return python_files[0]
    else:
        return None


def submit_attack(config_file, checkpoint_file, attack, attack_kwargs, result_dir):
    if os.path.exists(result_dir):
        logger.info(f"Skipping {result_dir} as it already exists")
        return None
    else:
        logger.info(f"Running attack {attack.__name__} with {attack_kwargs}")
        logger.info(f"saving results to {result_dir}")

    job = executor.submit(
        run_attack_val,
        attack,
        config_file,
        checkpoint_file,
        attack_kwargs,
        result_dir,
    )
    return job


def format_value(v):
    if isinstance(v, float):
        fraction = Fraction(v).limit_denominator(1000)
        if fraction.denominator <= 255:
            return f"{fraction.numerator}/{fraction.denominator}"
        else:
            return f"{v:.2f}"  # Limit to 2 decimal places
    return str(v)


checkpoint_files = []
config_files = []

for subdir in Path(MODEL_DIR).iterdir():
    if subdir.is_dir():
        checkpoint_file = find_latest_epoch_file(subdir)
        config_file = find_python_files(subdir)

        if checkpoint_file and config_file:
            logger.info(f"Checkpoint file: {checkpoint_file}")
            logger.info(f"Config file: {config_file}")
            checkpoint_files.append(checkpoint_file)
            config_files.append(config_file)
        else:
            logger.warning(f"No checkpoint or config file found in {subdir}")

logger.info("Setup submitit executor")
executor = submitit.AutoExecutor(folder=WORK_DIR)
executor.update_parameters(
    slurm_partition="gpu_4",
    slurm_gres="gpu:4",
    slurm_time="01:00:00",
    nodes=4,
    cpus_per_task=1,
    tasks_per_node=40,
    slurm_mem="300G",
    slurm_mail_type="NONE",
)
# submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []

logger.info(f"found {len(config_files)} config and checkpoint files")

try:
    for config_file, checkpoint_file in zip(config_files, checkpoint_files):
        for attack_name, attack in ATTACKS.items():
            steps = STEPS_ATTACK[attack_name]
            epsilons = EPSILONS[attack_name]
            alphas = ALPHAS[attack_name]
            norms = NORMS[attack_name]

            for epsilon in epsilons:
                for alpha in alphas:
                    for norm in norms:
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

                        result_dir = os.path.join(
                            f"{RESULT_DIR}_"
                            + "_".join(
                                [k + format_value(v) for k, v in attack_kwargs.items()]
                            )
                        )

                        job = submit_attack(
                            config_file,
                            checkpoint_file,
                            attack,
                            attack_kwargs,
                            result_dir,
                        )
                        if job:
                            jobs.append(job)

except FailedJobError:
    logger.exception("Failed to submit job")
finally:
    # wait until all jobs are completed:
    outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]

    # change notification and no need for a gpu
    executor.update_parameters(
        slurm_partition="single",
        slurm_time="01:00:00",
        slurm_gres="",
        slurm_mail_type="END, FAIL",
        slurm_mail_user="jonas.jakubassa@students.uni-mannheim.de",
    )

    executor.submit(collect_results, RESULT_DIR)
