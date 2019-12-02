import os
import sys
import shutil
import json
import time

def PrepareFolders(current_dir, experiment_dir = "configuration/", serialization_dir = "serialization/"):
    # if not exist, create folder for experiment
    path = os.path.join(current_dir, experiment_dir)
    if os.path.exists(path) == False:
        os.makedirs(path)
    
    serialization_path = os.path.join(current_dir,serialization_dir)
    # check if serialization folder is empty. If not - clean items
    if os.path.exists(serialization_path):
        shutil.rmtree(serialization_path) 
    # create serialization for experiment
        os.makedirs(serialization_path)

def AccessConfigurationField(field, data):
    splitted_field = field.split('.')
    if len(splitted_field) == 1:
        return data[splitted_field[0]]
    elif len(splitted_field) == 2:
        return data[splitted_field[0]][splitted_field[1]]
    else:
        return data[splitted_field[0]][splitted_field[1]][splitted_field[2]]

def UpdateConfigurationField(field,value,data):
    splitted_field = field.split('.')
    if len(splitted_field) == 1:
        data[splitted_field[0]] = value
    elif len(splitted_field) == 2:
        data[splitted_field[0]][splitted_field[1]] = value
    else:
        data[splitted_field[0]][splitted_field[1]][splitted_field[2]] = value

 
def CreateExperimentConfiguration(current_dir, base_dir, configuration_path, cuda_device, field_to_tune,size_to_tune, is_embedding, experiment_dir = "configuration/"):
    with open(os.path.join(base_dir,configuration_path)) as f:
        data = json.load(f)
        if is_embedding == 1:
            current_size = AccessConfigurationField(field_to_tune,data)
            currentLstmInputDim = data["model"]["internal_sentence_encoder"]["input_size"]

        # set cuda device
        data["trainer"]["cuda_device"] = int(cuda_device)
        data["model"]["model_sentence_field_embedder"]["cuda_device"] = int(cuda_device)

        # set size of field
        UpdateConfigurationField(field_to_tune,size_to_tune,data)
        
        # calculate input dim for lstm
        if is_embedding == 1:
            offset = size_to_tune - current_size
            data["model"]["internal_sentence_encoder"]["input_size"] = data["model"]["internal_sentence_encoder"]["input_size"] + offset
        
        with open(os.path.join(current_dir, experiment_dir + field_to_tune + "_" + str(size_to_tune) +  ".json"), 'w') as outfile:
            json.dump(data, outfile, indent=4)

if __name__ == "__main__":

    # get number of splits from cross-validation
    configuration_path = sys.argv[1]
    cuda_device = sys.argv[2]
    field_to_tune = sys.argv[3]
    is_embedding = (int)(sys.argv[5])
    if is_embedding == 1:
        size_to_tune = int(sys.argv[4])
    else: 
        size_to_tune = float(sys.argv[4])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir,"../")
	
    # create experiment configuration
    PrepareFolders(current_dir)
    CreateExperimentConfiguration(current_dir, base_dir, configuration_path, cuda_device, field_to_tune,size_to_tune, is_embedding)