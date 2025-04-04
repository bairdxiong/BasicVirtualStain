from Register import Registers

from .GANBased import *
from .DiffBased import *



def get_runner(runner_name, config):
    runner = Registers.runners[runner_name](config)
    return runner