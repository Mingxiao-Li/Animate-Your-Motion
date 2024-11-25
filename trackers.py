from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional
from omegaconf.dictconfig import DictConfig
import wandb 


class MyWandbTracker(GeneralTracker):
    name = "my_wandb"
    requires_logging_director = False 

    @on_main_process
    def __init__(self, config):
        run = wandb.init( 
                entity=config.entity,
                project=config.project,
                group=config.group,
                anonymous="allow",
                reinit=True
                )


    @property
    def tracker(self):
        return self.run.run 
    

    @on_main_process
    def store_init_configuration(self, values:dict):
        new_config = {}
        for k,v in values.items():
            if isinstance(v,DictConfig):
                new_config[k] = dict(v)
            else:
                new_config[k] = v 
        wandb.config.update(new_config)

    @on_main_process
    def log(self, values: dict=None, step:Optional[int]=None,):
        log_type = list(values.keys())[0]
        if "loss" in log_type:
            wandb.log(values, step=step)
        elif "video" in log_type:
            import ipdb 
            ipdb.set_trace()
            wandb.log(
               {"video": wandb.Video(values["video"], fps=8, format="gif")},
               step=step
            )
        else:
            raise ValueError("value is None, and video is None")
    
    @on_main_process
    def watch(self, model):
        wandb.watch(model)