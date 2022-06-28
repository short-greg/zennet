import sys
import os

def set_env():
    print(os.chdir('../'))
    
    dir_to_add = os.path.realpath(os.getcwd()) 
    # dir_to_add = os.path.realpath(os.path.join(os.getcwd(), '')) 
    sys.path.insert(0, dir_to_add)

from . import datasets
from . import modules
