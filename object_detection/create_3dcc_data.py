import os
import subprocess
import argparse
import submitit
import logging
from rich.logging import RichHandler
from dotenv import load_dotenv
import os

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


def download_weights(weights_dir, weights_file, weights_url):
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, weights_file)
    if not os.path.exists(weights_path):
        logger.info("Weight file does not exist, downloading...")
        subprocess.run(["wget", "-P", weights_dir, weights_url])
    else:
        logger.info("Weight file already exists, no need to download.")


def create_depth_info(path_rgb, path_depth):
    logger.info("Starting depth information creation...")
    logger.info("Moving images to DPT input directory...")
    os.makedirs("DPT/input", exist_ok=True)
    for jpg_file in os.listdir(path_rgb):
        if jpg_file.endswith(".jpg"):
            os.rename(
                os.path.join(path_rgb, jpg_file), os.path.join("DPT/input", jpg_file)
            )

    logger.info("Running depth estimation model...")
    subprocess.run(["python", "DPT/run_monodepth.py"])

    logger.info("Creating target directory for depth information...")
    os.makedirs(path_depth, exist_ok=True)

    logger.info("Moving depth output to target directory...")
    for png_file in os.listdir("DPT/output_monodepth"):
        if png_file.endswith(".png"):
            os.rename(
                os.path.join("DPT/output_monodepth", png_file),
                os.path.join(path_depth, png_file),
            )

    logger.info("Restoring original images...")
    for jpg_file in os.listdir("DPT/input"):
        if jpg_file.endswith(".jpg"):
            os.rename(
                os.path.join("DPT/input", jpg_file), os.path.join(path_rgb, jpg_file)
            )

    logger.info("Depth information creation completed.")


def create_3d_corruptions(path_rgb, path_depth, path_target):
    logger.info("Starting 3D corruption creation...")
    subprocess.run(
        [
            "python",
            "3DCommonCorruptions/create_3dcc/create_3dcc.py",
            "--path_rgb",
            path_rgb,
            "--path_depth",
            path_depth,
            "--path_target",
            path_target,
            "--batch_size",
            "1",
        ]
    )
    logger.info("3D corruption creation completed.")


def process_dataset(name, path_rgb, path_depth, path_target, create_depth):
    logger.info(f"Processing {name} dataset...")

    if create_depth:
        create_depth_info(path_rgb, path_depth)

    create_3d_corruptions(path_rgb, path_depth, path_target)
    logger.info(f"{name} dataset processing completed.")


def main(args):
    weights_dir = "DPT/weights"
    weights_file = "dpt_hybrid-midas-501f0c75.pt"
    weights_url = (
        f"https://github.com/intel-isl/DPT/releases/download/1_0/{weights_file}"
    )

    download_weights(weights_dir, weights_file, weights_url)

    datasets = {
        "coco": (
            "data/coco/val2017/val2017",
            "data/coco/val2017_depth",
            "data/coco/3dcc",
        ),
        "voc2007": (
            "data/VOCdevkit/VOC2007/JPEGImages",
            "data/VOCdevkit/VOC2007/depth",
            "data/VOCdevkit/VOC2007/3dcc",
        ),
        "voc2012": (
            "data/VOCdevkit/VOC2012/JPEGImages",
            "data/VOCdevkit/VOC2012/depth",
            "data/VOCdevkit/VOC2012/3dcc",
        ),
    }

    selected_datasets = (
        list(datasets.keys()) if args.dataset is None else [args.dataset]
    )

    # Submit separate jobs for each dataset
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_partition="gpu_4",
        slurm_gres="gpu:1",
        cpus_per_task=4,
        nodes=1,
        tasks_per_node=1,
        slurm_mem=30,
        timeout_min=1800,  # 30 hours
        slurm_mail_type="END,FAIL",
        slurm_mail_user=my_email,
    )

    for dataset in selected_datasets:
        paths = datasets[dataset]
        job_name = f"{dataset}_processing"
        executor.update_parameters(name=job_name)
        job = executor.submit(process_dataset, dataset, *paths, args.create_depth)
        logger.info(
            f"Submitted job for {dataset} dataset with ID: {job.job_id} and name: {job_name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets with depth estimation and 3D corruptions."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["coco", "voc2007", "voc2012"],
        help="Specify the dataset to process.",
    )
    parser.add_argument(
        "--create_depth",
        action="store_true",
        help="Create depth images before 3D corruptions.",
    )

    args = parser.parse_args()
