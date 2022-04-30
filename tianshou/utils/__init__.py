"""Utils package."""
import os

from tianshou.utils.config import tqdm_config
from tianshou.utils.logger.base import BaseLogger, LazyLogger
from tianshou.utils.logger.tensorboard import BasicLogger, TensorboardLogger
from tianshou.utils.logger.wandb import WandbLogger
from tianshou.utils.lr_scheduler import MultipleLRSchedulers
from tianshou.utils.statistics import MovAvg, RunningMeanStd
from tianshou.utils.warning import deprecation

__all__ = [
    "MovAvg",
    "RunningMeanStd",
    "tqdm_config",
    "BaseLogger",
    "TensorboardLogger",
    "BasicLogger",
    "LazyLogger",
    "WandbLogger",
    "deprecation",
    "MultipleLRSchedulers",
    "use_morl",
]

def set_morl(enable: bool = True) -> None:
    os.environ['MORL_ENV'] = str(enable).lower()


def use_morl() -> bool:
    return os.environ.get('MORL_ENV', 'false') == 'true'
