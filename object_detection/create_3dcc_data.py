import os
import subprocess
import argparse
import submitit
import logging
from rich.logging import RichHandler
from dotenv import load_dotenv

load_dotenv()  # needed to add 3DCommonCorruptions to the path and to get the email
os.chdir("3DCommonCorruptions/create_3dcc")  # deal with relative imports

from create_3dcc import (  # noqa: E402
    create_dof_data,
    create_fog_data,
    create_non3d_data,
    create_flash_data,
    create_shadow_data,
    create_multi_illumination_data,
    create_motion_data,
    create_video_data,
)

corruptions = [
    create_dof_data,
    create_fog_data,
    create_non3d_data,
    create_flash_data,
    create_shadow_data,
    create_multi_illumination_data,
    create_motion_data,
    create_video_data,
]
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
    executor = submitit.AutoExecutor(folder="slurm/logs/cc/%j")
    executor.update_parameters(
        slurm_partition="gpu_4",
        slurm_gres="gpu:1",
        cpus_per_task=4,
        nodes=1,
        tasks_per_node=1,
        slurm_mem=30,
        slurm_time="48:00:00",
        slurm_mail_type="END,FAIL",
        slurm_mail_user=my_email,
    )

    for dataset in selected_datasets:
        path_rgb, path_depth, path_target = datasets[dataset]

        if args.create_depth:
            create_depth_info(path_rgb, path_depth)

        for corruption in corruptions:
            kwargs = {
                "BASE_PATH_RGB": path_rgb,
                "BASE_PATH_DEPTH": path_depth,
                "BASE_TARGET_PATH": path_target,
                "BATCH_SIZE": 1,
            }
            if corruption.__name__ == "create_non3d_data":
                del kwargs["BASE_PATH_DEPTH"]

            job_name = f"3dcc_{dataset}_{corruption.__name__}"
            executor.update_parameters(name=job_name)
            job = executor.submit(
                corruption,
                **kwargs,
            )
            logger.info(f"{job.job_id}: {corruption.__name__} - {kwargs}")


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
    main(args)
