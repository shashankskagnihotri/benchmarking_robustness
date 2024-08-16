import os
from tqdm import tqdm
from pycocotools.coco import COCO
from imagecorruptions import get_corruption_names, corrupt
import mmcv
import logging
from rich.logging import RichHandler
import submitit
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


def common_corruptions(dataset_type, input_dir, output_dir):
    """
    Apply common corruptions to images in COCO or Pascal VOC dataset.

    :param dataset_type: Type of dataset ('coco' or 'voc').
    :param input_dir: Path to the input dataset directory.
    :param output_dir: Path to the output directory to save corrupted images.
    """
    logger.info(f"Applying common corruptions to {dataset_type} dataset.")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    if dataset_type == "coco":
        # Load the COCO annotations
        ann_file = os.path.join(input_dir, "annotations", "instances_val2017.json")
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        img_dir = os.path.join(input_dir, "val2017")

    elif dataset_type == "voc":
        # Load the VOC image set file
        img_set_file = os.path.join(input_dir, "ImageSets", "Main", "test.txt")
        with open(img_set_file, "r") as f:
            img_ids = [line.strip() for line in f.readlines()]
        img_dir = os.path.join(input_dir, "JPEGImages")

    else:
        raise ValueError("dataset_type must be either 'coco' or 'voc'")

    # Apply corruptions and save to folders
    corruptions = get_corruption_names()
    severities = [1, 2, 3, 4, 5]
    for corruption in corruptions:
        for severity in severities:
            logger.info(f"Applying corruption '{corruption}' with severity {severity}.")

            # Create folders for each corruption and severity
            output_folder = os.path.join(
                output_dir, "cc", corruption, f"severity_{severity}"
            )
            os.makedirs(output_folder, exist_ok=True)

            # Check if folder already contains all images
            existing_images = len(os.listdir(output_folder))
            if existing_images >= len(img_ids):
                print(
                    f"Skipping corruption '{corruption}' with severity {severity} as all {existing_images} images already exist."
                )
                continue

            # Apply corruption and save corrupted images
            for img_id in img_ids:
                if dataset_type == "coco":
                    img_info = coco.loadImgs(img_id)[0]
                    img_path = os.path.join(img_dir, img_info["file_name"])
                    output_path = os.path.join(output_folder, img_info["file_name"])
                else:  # 'voc'
                    img_path = os.path.join(img_dir, f"{img_id}.jpg")
                    output_path = os.path.join(output_folder, f"{img_id}.jpg")

                image = mmcv.imread(img_path)
                corrupted_image = corrupt(
                    image, corruption_name=corruption, severity=severity
                )
                mmcv.imwrite(corrupted_image, output_path)
    logger.info("Common corruptions applied successfully.")


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="slurm/logs/cc/%j")
    executor.update_parameters(
        slurm_partition="gpu_4",
        slurm_gres="gpu:1",
        cpus_per_task=4,
        nodes=1,
        tasks_per_node=1,
        slurm_mem="4GB",
        slurm_time="10:00:00",
        slurm_mail_type="END,FAIL",
        slurm_mail_user=my_email,
    )
    jobs = []

    dataset_type = "voc"
    input_dir = "data/VOCdevkit/VOC2012/"
    output_dir = "data/VOCdevkit/VOC2012/cc/"
    job_name = "voc2012_cc_processing"
    executor.update_parameters(name=job_name)
    job = executor.submit(common_corruptions, dataset_type, input_dir, output_dir)
    jobs.append(job)
    logger.info(f"Submitted job with ID: {job.job_id} and name: {job_name}")

    dataset_type = "voc"
    input_dir = "data/VOCdevkit/VOC2007/"
    output_dir = "data/VOCdevkit/VOC2007/cc/"
    job_name = "voc2017_cc_processing"
    executor.update_parameters(name=job_name)
    job = executor.submit(common_corruptions, dataset_type, input_dir, output_dir)
    jobs.append(job)
    logger.info(f"Submitted job with ID: {job.job_id} and name: {job_name}")

    dataset_type = "coco"
    input_dir = "data/coco/"
    output_dir = "data/coco/cc/"
    job_name = "coco_cc_processing"
    executor.update_parameters(name=job_name)
    job = executor.submit(common_corruptions, dataset_type, input_dir, output_dir)
    jobs.append(job)
    logger.info(f"Submitted job with ID: {job.job_id} and name: {job_name}")

    outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]
    logger.info("All jobs completed successfully.")
    logger.info(outputs)
