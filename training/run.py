import logging
import os
import sys
import inspect
from allennlp.commands import main

if __name__ == "__main__":

    # insert models to current path of the program
    sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    
    from models import *
    main(prog=("python run.py")) #call main program