
cfg_options = [
    "model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20",
    "model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20",
] # TODO segpgd

all_segformer_cityscapes_models = {
    "MIT-B0": {
        "config": "../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py",
        "checkpoint": "../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth"
    },
    "MIT-B1": {
        "config": "../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py",
        "checkpoint": "../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth"
    },
    "MIT-B2": {
        "config": "../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py",
        "checkpoint": "../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth"
    },
    "MIT-B3": {
        "config": "../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py",
        "checkpoint": "../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth"
    },
    "MIT-B4": {
        "config": "../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py",
        "checkpoint": "../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth"
    },
    "MIT-B5": {
        "config": "../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py",
        "checkpoint": "../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth"
    },
}

sh_array = """"""
counter = 0
for model in all_segformer_cityscapes_models:
    config = all_segformer_cityscapes_models[model]["config"]
    checkpoint = all_segformer_cityscapes_models[model]["checkpoint"]
    work_dir = "../work_dirs/" + config.split('/')[-1].split('.py')[0]
    for cfg_option in cfg_options:
        counter += 1
        if counter == 1:
            start_if = "if"
        else:
            start_if = "elif"
        sh_array += f"""{start_if} [[ $SLURM_ARRAY_TASK_ID -eq {counter} ]]
then
    python tools/test.py {config} {checkpoint} --work-dir {work_dir} --cfg-options {cfg_option}
"""
sh_array += """else
    echo "All submitted"
fi
"""

with open("../scripts/all_untargeted_attacks_cityscapes_segformer.sh", "a") as fp:
    fp.write(sh_array)

print(counter)