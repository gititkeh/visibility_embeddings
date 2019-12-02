import logging
import os
import sys
import shutil
import time
import json

from sklearn.model_selection import KFold # to use cross-validation on the same input

# insert the parent directory of the current directory in index 0 of current's program path
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

# log any message with info sevirity or higher will be logged (not DEBUG or NOTSET)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

"""
    delete current split directory if exist. Otherwise, just create it
"""
def CreateSplitDir(current_dir, dirName = "experiments"):
    abs_dir = os.path.join(current_dir,dirName)
    DeleteSplitDir(abs_dir)
    # wait a few seconds for the folder to be 
    os.makedirs(abs_dir)

""" 
Delete split directory if exist
"""	
def DeleteSplitDir(abs_dirName = "experiments"):
    if os.path.exists(abs_dirName) and os.path.isdir(abs_dirName):
	    shutil.rmtree(abs_dirName)
    
"""
    This method reads the input file which is a csv, and split it to numberOfSplits pairs of
    training-validation data for with X iterations
    The format of file names will be train*Index*/validate*Index* where Index is the iteration index
    If number of splits > 1, this is cross validation.
"""
def PrepareExperimentFiles(current_dir, training_file_path, numberOfSplits = 1, shuffle_cross_validation = 0, validation_file_path=None, split_dir = "experiments", encoding = 'utf-8'):

    # prepare training files (which is also preparing validation files if cross validation)
    with open(training_file_path, encoding=encoding) as f:
        # open the file and skip the header
        lines = [line.rstrip('\n') for line in f]
        lines = lines[1:] # skip first header line
            
        print('Dataset for Training:', len(lines)) # print number of lines read from csv file
      
        split_path = os.path.join(current_dir,split_dir)
		
        if numberOfSplits == 1:
            # write the lines for training
            WriteToSplit(split_path,lines,1,True)
        else:
            kf = KFold(numberOfSplits,bool(shuffle_cross_validation))
            counter = 1 # count now splits
		
            for train_index, validate_index in kf.split(lines):
                train = [lines[i] for i in list(train_index)] 
                WriteToSplit(split_path,train,counter,True,encoding)
				
                validate = [lines[i] for i in list(validate_index)] 
                WriteToSplit(split_path,validate,counter,False,encoding)
                counter = counter + 1
				
        # prepare validation files
        if numberOfSplits == 1:
            with open(validation_file_path,encoding=encoding) as f:
                # open the file and skip the header
                lines = [line.rstrip('\n') for line in f]
                lines = lines[1:] # skip first header line
            
                print('Dataset For Validating:', len(lines)) # print number of lines read from csv file
                split_path = os.path.join(current_dir,split_dir)
                # write the lines for training
                WriteToSplit(split_path,lines,1,False,encoding)
	
"""
    This function write train/validate file to split folder
"""
def WriteToSplit(splitDir,lines,index,isTraining, encoding = 'utf-8'):
    if isTraining:
        fileName = "train"
    else:
        fileName = "validate"
        
    fileName = fileName + str(index)
    fileName = fileName + ".csv"
		
    with open(os.path.join(splitDir,fileName),'w', encoding=encoding) as f:
        for line in lines:
            f.write(line + '\n')

def LoadConfiguration(base_dir, current_dir, base_config_path, splitDir = "experiments"):
    split_dir = os.path.join(current_dir,splitDir)
    head, tail = os.path.split(base_config_path)
    configuration_file_name = tail
    with open(os.path.join(base_dir,base_config_path)) as f:
        data = json.load(f)
        with open(os.path.join(split_dir,configuration_file_name), 'w') as outfile:
            json.dump(data, outfile, indent=4)

def ReturnTrainingDataPath(base_dir, current_dir, base_config_path, splitDir = "experiments"):
    split_dir = os.path.join(current_dir,splitDir)
    head, tail = os.path.split(base_config_path)
    configuration_file_name = tail
    with open(os.path.join(base_dir,base_config_path)) as f:
        data = json.load(f)
        return data["train_data_path"]
		
def ReturnValidationDataPath(base_dir, current_dir, base_config_path, splitDir = "experiments"):
    split_dir = os.path.join(current_dir,splitDir)
    head, tail = os.path.split(base_config_path)
    configuration_file_name = tail
    with open(os.path.join(base_dir,base_config_path)) as f:
        data = json.load(f)
        return data["validation_data_path"]

def SetupSerialization(currentDir, counter, serializationDir = "serialization"):
    abs_dir = os.path.join(currentDir,serializationDir)
    os.makedirs(os.path.join(abs_dir,str(counter)))

def UpdateConfiguration(currentDir, base_config_path, counter, cuda_device, splitDir = "experiments"):
    abs_dir = os.path.join(currentDir,splitDir)
    head, tail = os.path.split(base_config_path)
    configuration_file_name = tail
    with open(os.path.join(abs_dir,configuration_file_name)) as f:
        data = json.load(f)
        suffix = str(counter) + ".csv"
        # initialize train/validate
        data["train_data_path"] = os.path.join(abs_dir,"train" + suffix)
        data["validation_data_path"] = os.path.join(abs_dir,"validate" + suffix)

        # configure cuda device
        data["trainer"]["cuda_device"] = int(cuda_device)
        data["model"]["model_sentence_field_embedder"]["cuda_device"] = int(cuda_device)
		
        # accumulate embedding dimensions to write to lstm input dimensions
        embedding_dim = 0
        if data["model"]["model_sentence_field_embedder"]["use_glove_embedding"] == True:
            #embedding_dim = embedding_dim + data["model"]["model_sentence_field_embedder"]["glove_dimension_size"]
            embedding_dim = embedding_dim + data["model"]["model_sentence_field_embedder"]["glove_embedder"]["embedding_dim"]
			
        if data["model"]["model_sentence_field_embedder"]["use_elmo_embedding"] == True:
            embedding_dim = embedding_dim + 1024
			
        if data["model"]["model_sentence_field_embedder"]["use_verb_index_embedding"] == True:
            embedding_dim = embedding_dim + data["model"]["model_sentence_field_embedder"]["verb_index_embedding_dimension"]
			
        if data["model"]["model_sentence_field_embedder"]["use_visual_score_embedding"] == True:
            embedding_dim = embedding_dim + data["model"]["model_sentence_field_embedder"]["visual_embedder"]["embedding_dim"]
			
        # set lstm dimensions
        data["model"]["internal_sentence_encoder"]["input_size"] = embedding_dim

        with open(os.path.join(abs_dir,str(counter) +".json"), 'w') as outfile:
            json.dump(data, outfile, indent=4)

if __name__ == "__main__":

    # get number of splits for cross-validation
    # if number of splits is 1, this is not a cross validation, only train on training file
    # and validate with validation file
    number_of_splits = (int)(sys.argv[1])
	
    path_of_configuration = sys.argv[2]
	
    print(path_of_configuration)

    # cuda device - (-1) if don't use GPU
    cuda_device = sys.argv[3]

    # is to shuffle data in cross validation
    shuffle_cross_validation = int(sys.argv[4])
	
    # encoding of file
    encoding_of_dataset = sys.argv[5]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "../")
    # create experiment folder
    CreateSplitDir(current_dir)
    # initialize configuration in work folder
    LoadConfiguration(base_dir, current_dir,path_of_configuration)
    training_file = ReturnTrainingDataPath(base_dir, current_dir,path_of_configuration)
    training_file = os.path.join(base_dir,training_file)
    validation_file = None
    
    if number_of_splits == 1:
        # train on one training file and validate with a validation file.
        # if number of splits > 1, we don't have a validation file - we do cross validation
        validation_file = ReturnValidationDataPath(base_dir, current_dir,path_of_configuration)
        validation_file = os.path.join(base_dir,validation_file)
     
    # number of splits > 1 means cross validation
    if number_of_splits == 1:
        PrepareExperimentFiles(current_dir,training_file, number_of_splits, 0, validation_file, "experiments", encoding_of_dataset)
    else:
        PrepareExperimentFiles(current_dir,training_file, number_of_splits, shuffle_cross_validation,None,"experiments",encoding_of_dataset)

    # create serialization directory
    serialization_dir = os.path.join(current_dir,"serialization/")
    # check if serialization folder is empty. If not - clean items
    if os.path.exists(serialization_dir) == True:
        shutil.rmtree(serialization_dir)
    os.makedirs(serialization_dir)
	
    for i in range(1,number_of_splits + 1):
        UpdateConfiguration(current_dir, path_of_configuration, i, cuda_device, splitDir = "experiments")
        SetupSerialization(current_dir, i, serializationDir = "serialization")