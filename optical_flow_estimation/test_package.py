from test import evaluate
#attack_args = {"attack_epsilon": 8/255}
evaluate(model_name='raft', pretrained_ckpt='kitti', dataset='kitti-2015', dataset_config_path="./ptlflow/datasets.yml")