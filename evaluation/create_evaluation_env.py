import logging
import os
import sys

# insert the parent directory of the current directory in index 0 of current's program path
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

# log any message with info sevirity or higher will be logged (not DEBUG or NOTSET)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

"""
    delete current split directory if exist. Otherwise, just create it
"""
def CreateResultDir(current_dir, evaluation_input, result_dir = "results"):
    abs_dir = os.path.join(current_dir,result_dir)
    abs_dir = os.path.join(abs_dir,evaluation_input)
    os.makedirs(abs_dir)

if __name__ == "__main__":

    evaluation_input = sys.argv[1]
	
    current_dir = os.path.dirname(os.path.abspath(__file__))

    CreateResultDir(current_dir,evaluation_input)