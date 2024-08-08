from pathlib import Path
import submitit
from corruption_val import run_corruption_val
from tqdm import tqdm
import logging
from rich.logging import RichHandler
import os
import itertools
from dotenv import load_dotenv
from misc import find_latest_epoch_file, find_python_files

load_dotenv()  # Load environment variables from .env file
my_email = os.getenv("MY_EMAIL")
assert my_email is not None, "Please set the MY_EMAIL environment variable"

# Set up the logging configuration to use RichHandler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",  # Custom date format
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Create a logger
logger = logging.getLogger("rich")
WORK_DIR = "slurm/logs/corruptions/%j"
RESULT_DIR = "slurm/results"
MODEL_DIR = "models"
CORRUPTIONS = [
    "3dcc/bit_error",
    "3dcc/color_quant",
    "3dcc/far_focus",
    "3dcc/fog_3d",
    "3dcc/h265_abr",
    "3dcc/h265_crf",
    "3dcc/iso_noise",
    "3dcc/low_light",
    "3dcc/motion_blur_vid",
    "3dcc/near_focus",
    "3dcc/xy_motion_blur",
    "3dcc/z_motion_blur",
    "3dcc/zoom_blur_vid",
    "cc/brightness",
    "cc/contrast",
    "cc/defocus_blur",
    "cc/elastic_transform",
    "cc/fog",
    "cc/frost",
    "cc/gaussian_blur",
    "cc/impulse_noise",
    "cc/jpeg_compression",
    "cc/motion_blur",
    "cc/pixelate",
    "cc/shot_noise",
    "cc/snow",
    "cc/zoom_blur",
]
SEVERITIES = [3]
# SEVERITIES = [1, 2, 3, 4, 5]

logger.debug("Starting attack tasks")
logger.debug(f"WORK_DIR: {WORK_DIR}")
logger.debug(f"RESULT_DIR: {RESULT_DIR}")
logger.debug(f"MODEL_DIR: {MODEL_DIR}")
logger.debug(f"CORRUPTIONS: {CORRUPTIONS}")
logger.debug(f"SEVERITIES: {SEVERITIES}")


def get_configs_and_checkpoints(model_dir):
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
    return config_files, checkpoint_files


config_files, checkpoint_files = get_configs_and_checkpoints(MODEL_DIR)

logger.info(f"found {len(config_files)} config and checkpoint files")

# Set up executor for slurm
executor = submitit.AutoExecutor(folder=WORK_DIR)
executor.update_parameters(
    slurm_partition="gpu_4",
    slurm_gres="gpu:1",
    nodes=1,
    cpus_per_task=1,
    tasks_per_node=1,
    slurm_mem=96_000,
    slurm_mail_type="all",
    slurm_job_name="corruptions",
    slurm_time="30:00",
)
jobs = []

for config_file, checkpoint_file in zip(config_files, checkpoint_files):
    with executor.batch():
        for corruption, severity in itertools.product(CORRUPTIONS, SEVERITIES):
            logger.debug(str(config_file))
            logger.debug(str(checkpoint_file))

            model_name = str(config_file).split("/")[-1][0:-3]
            result_dir = os.path.join(
                f"{RESULT_DIR}/{model_name}/{corruption}_{severity}"
            )

            if os.path.exists(result_dir):
                logger.info(f"skipping {result_dir} as it already exists")
            else:
                logger.info(f"running corruption {corruption} with severity {severity}")
                logger.info(f"saving results to {result_dir}")
                job = executor.submit(
                    run_corruption_val,
                    corruption,
                    severity,
                    config_file,
                    checkpoint_file,
                    result_dir,
                )
                jobs.append(job)

logger.info(
    "Waiting for all jobs to complete. Can be canceled without cancelling the jobs."
)
outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]
