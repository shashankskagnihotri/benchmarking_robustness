import submitit
from create_corrupted_data import common_corruptions_coco

executor = submitit.AutoExecutor(folder="slurm/logs/%j")

executor.update_parameters(
    timeout_min=4,
    slurm_partition="single",
    slurm_time="20:00:00",
    nodes=1,
    cpus_per_task=8,
    tasks_per_node=1,
    slurm_mem="10G",
    slurm_mail_type="END, FAIL",
    slurm_mail_user="jonas.jakubassa@students.uni-mannheim.de",
)

job = executor.submit(common_corruptions_coco)
job.result()
