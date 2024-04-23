import os
from mmengine.runner import Runner
from mmengine.config import Config
from rich.traceback import install

install(show_locals=False)


for configs_train in os.listdir("./configs_to_train"):
    print(configs_train)
    cfg = Config.fromfile(f"./configs_to_train/{configs_train}")
    cfg.work_dir = "./work_dirs/"
    cfg.total_epochs = 1
    runner = Runner.from_cfg(cfg)
    runner.train()


#! something to check if done correctly -> move into other folder?
#! and something if fails -> stays in folder but how that it does not always runs the same failing ones

#! slurm per python bei Jonas schauen in submit attacks
#! braucht ne function also training als function machen und slurm übergeben
#! wenn slurm error macht dann irgendwo saven
