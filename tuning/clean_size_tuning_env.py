import os
import sys
import shutil

def CleanFolders(current_dir, field_name, size_to_tune, experiment_dir = "configuration/", serialization_dir = "serialization/"):
    
    # clean configuration file for experiment
    experiment_configuration_path = os.path.join(current_dir, experiment_dir + field_name + "_" + str(size_to_tune) +  ".json")
    if os.path.exists(experiment_configuration_path) == True:
        os.remove(experiment_configuration_path)
        		
if __name__ == "__main__":

    # get number of splits from cross-validation
    field_to_tune = sys.argv[1]
    is_embedding = int(sys.argv[3])
    if is_embedding == 1:
        size_to_tune = int(sys.argv[2])
    else:
        size_to_tune = float(sys.argv[2])

    current_dir = os.path.dirname(os.path.abspath(__file__))
	
    # create experiment configuration
    CleanFolders(current_dir, field_to_tune, size_to_tune)