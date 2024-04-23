import submitit
from adv_attack import run_attack_val, pgd_attack, fgsm_attack, bim_attack
from collect_attack_results import collect_results
from tqdm import tqdm


WORK_DIR = "slurm/work_dir/%j"
CHECKPOINT_FILE = (
    "mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
)
CONFIG_FILE = "mmdetection/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"
TARGETED = False
RANDOM_START = False
RESULT_DIR = "slurm/results"

executor = submitit.AutoExecutor(folder=WORK_DIR)

executor.update_parameters(
    slurm_partition="gpu_4",
    slurm_gres="gpu:1",
    slurm_time="01:00:00",
    nodes=1,
    cpus_per_task=4,
    tasks_per_node=1,
    slurm_mem="10G",
    slurm_mail_type="NONE",
)

submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []
for steps in [1, 4]:
    for attack in [pgd_attack, fgsm_attack]:
        for epsilon in [16, 32]:
            for alpha in [
                2,
                8,
            ]:
                if attack == pgd_attack:
                    attack_kwargs = {
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "targeted": TARGETED,
                        "steps": steps,
                        "random_start": RANDOM_START,
                    }

                    job = executor.submit(
                        run_attack_val,
                        attack,
                        CONFIG_FILE,
                        CHECKPOINT_FILE,
                        attack_kwargs,
                        RESULT_DIR,
                    )
                    jobs.append(job)

                elif attack == fgsm_attack:
                    for norm in ["inf", "two"]:
                        attack_kwargs = {
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "targeted": TARGETED,
                            "norm": norm,
                        }

                        job = executor.submit(
                            run_attack_val,
                            attack,
                            CONFIG_FILE,
                            CHECKPOINT_FILE,
                            attack_kwargs,
                            RESULT_DIR,
                        )
                        jobs.append(job)

                elif attack == bim_attack:
                    for norm in ["inf", "two"]:
                        attack_kwargs = {
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "targeted": TARGETED,
                            "norm": norm,
                            "steps": steps,
                        }

                        job = executor.submit(
                            run_attack_val,
                            attack,
                            CONFIG_FILE,
                            CHECKPOINT_FILE,
                            attack_kwargs,
                            RESULT_DIR,
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
    slurm_mail_user="jonas.jakubassa@students.uni-mannheim.de",
)

executor.submit(collect_results, RESULT_DIR)
