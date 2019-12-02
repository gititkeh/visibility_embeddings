import logging
import os
import sys
import shutil
import datetime
import json

# insert the parent directory of the current directory in index 0 of current's program path
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

# log any message with info sevirity or higher will be logged (not DEBUG or NOTSET)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
					
best_epoch = 0.0
validation_accuracy = 0.0
validation_precision_literal = 0.0
validation_recall_literal = 0.0
validation_f1_literal = 0.0
validation_precision_metaphore = 0.0
validation_recall_metaphore = 0.0
validation_f1_metaphore = 0.0
validation_avg_precision = 0.0
validation_avg_recall = 0.0
validation_avg_f1 = 0.0
validation_loss = 0.0

def SaveAvgMetricsToTuningResults(base_dir, serialization_dir, result_file_name, f_metaphore):

    # find field name
    head = "_".join(result_file_name.split('_')[:-1])
    tail = result_file_name.split('_')[-1]
    tail = os.path.splitext(tail)[0]
    
    result_folder = os.path.join(base_dir, "tuning/results/" + head)
    # create destination folder
    if os.path.exists(result_folder) == False:
        os.makedirs(result_folder)

    #copy result
    #create source
    path_source = os.path.join(serialization_dir, "AvgMetrics.json")
    
    # create destination
    result_file = tail + "_f_met_" + str(f_metaphore) + ".json"
    path_destination = os.path.join(result_folder,result_file)
    shutil.move(path_source, path_destination)
	
    # copy also the configuration file
    # create source
    path_source = os.path.join(serialization_dir, "1/" + "config.json")

    # create destination
    config_file = tail + "config.json"
    path_destination = os.path.join(result_folder,config_file)
    shutil.move(path_source, path_destination)

def CleanFolders(current_dir, is_tuning, serialization_dir = "serialization/", trained_model_dir ="trainedModels/"):
    
    serialization_dir = os.path.join(current_dir,serialization_dir)
    # move result file to experiment folder- only if not using tuning
    if is_tuning == 0:
        # trained model name will contain training time
        time_str = "{date:%Y_%m_%d_%H_%M_%S}".format(date=datetime.datetime.now())
    
        # create trained model folder
        trained_model_dir = os.path.join(trained_model_dir,time_str)
        os.makedirs(trained_model_dir)

        # copy result
        path_source = os.path.join(serialization_dir)
        path_destination = os.path.join(trained_model_dir)
    
        # move all the serialization files to trained_model dir 
        files = os.listdir(path_source)
        for f in files:
            shutil.move(path_source+f, path_destination)
    
    # clean serialization folder
    # check if serialization folder is empty. If not - clean items
    if any(os.scandir(serialization_dir)) == True:
        shutil.rmtree(serialization_dir)

"""
    Write result to json file
"""
def SaveAvgScoreToResult(resultFile, number_of_splits):

    global best_epoch
    global validation_accuracy
    global validation_precision_literal
    global validation_recall_literal
    global validation_f1_literal
    global validation_precision_metaphore
    global validation_recall_metaphore
    global validation_f1_metaphore
    global validation_avg_precision
    global validation_avg_recall
    global validation_avg_f1
    global validation_loss

    resultFile["best_epoch"] = best_epoch
    resultFile["validation_accuracy"] = validation_accuracy / number_of_splits
    resultFile["validation_precision_literal"] = validation_precision_literal / number_of_splits
    resultFile["validation_recall_literal"] = validation_recall_literal / number_of_splits
    resultFile["validation_f1_literal"] = validation_f1_literal / number_of_splits
    resultFile["validation_precision_metaphore"] = validation_precision_metaphore / number_of_splits
    resultFile["validation_recall_metaphore"] = validation_recall_metaphore / number_of_splits
    resultFile["validation_f1_metaphore"] = validation_f1_metaphore / number_of_splits
    resultFile["validation_avg_precision"] = validation_avg_precision / number_of_splits
    resultFile["validation_avg_recall"] = validation_avg_recall / number_of_splits
    resultFile["validation_avg_f1"] = validation_avg_f1 / number_of_splits
    resultFile["validation_loss"] = validation_loss / number_of_splits

	
"""
    Add contenct of current result file to result
"""
def AddScoreIterationToResult(loadedFile):

    global best_epoch
    global validation_accuracy
    global validation_precision_literal
    global validation_recall_literal
    global validation_f1_literal
    global validation_precision_metaphore
    global validation_recall_metaphore
    global validation_f1_metaphore
    global validation_avg_precision
    global validation_avg_recall
    global validation_avg_f1
    global validation_loss
	
    best_epoch = max(best_epoch,loadedFile["best_epoch"])
    validation_accuracy = validation_accuracy + loadedFile["best_validation_accuracy"]
    validation_precision_literal = validation_precision_literal + loadedFile["best_validation_precision_literal"]
    validation_recall_literal = validation_recall_literal + loadedFile["best_validation_recall_literal"]
    validation_f1_literal = validation_f1_literal + loadedFile["best_validation_f1_literal"]
    validation_precision_metaphore = validation_precision_metaphore + loadedFile["best_validation_precision_metaphore"]
    validation_recall_metaphore = validation_recall_metaphore + loadedFile["best_validation_recall_metaphore"]
    validation_f1_metaphore = validation_f1_metaphore + loadedFile["best_validation_f1_metaphore"]
    validation_avg_precision = validation_avg_precision + loadedFile["best_validation_avg_precision"]
    validation_avg_recall = validation_avg_recall + loadedFile["best_validation_avg_recall"]
    validation_avg_f1 = validation_avg_f1 + loadedFile["best_validation_avg_f1"]
    validation_loss = validation_loss + loadedFile["best_validation_loss"]
	

if __name__ == "__main__":

    result_file_name=None
    # get number of splits. If number of splits > 1 this is cross validation
    number_of_splits = (int)(sys.argv[1])
    is_tuning = (int)(sys.argv[2])
    if is_tuning == 1:
       result_file_name = sys.argv[3]
    result_dir = "serialization/"

    # get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir,"../")
    abs_dir_result = os.path.join(current_dir,"serialization")
	
    f_met = 0.0
    if number_of_splits == 1:
        # no multiple number of experiments. Just load metrics.json and save again to AvgMetrics.json
        abs_dir = os.path.join(current_dir,result_dir + str(1))
        with open(os.path.join(abs_dir,"metrics.json"), encoding="utf8") as f:
            data = json.load(f)
            # get f_metaphore score
            f_met = data["test_f1_metaphore"]
		
        # write result
        with open(os.path.join(abs_dir_result,"AvgMetrics.json"), 'w', encoding="utf8") as outfile:
            json.dump(data, outfile, indent=4)
    else:
        # load results from each experiment
        for i in range(1,number_of_splits+1):
            abs_dir = os.path.join(current_dir,result_dir + str(i))
            with open(os.path.join(abs_dir,"metrics.json"), encoding="utf8") as f:
                data = json.load(f)
                AddScoreIterationToResult(data)
	
        # write result
        data = {}

        SaveAvgScoreToResult(data,float(number_of_splits))
	
        # get f_metaphore score
        f_met = data["validation_f1_metaphore"]
        with open(os.path.join(abs_dir_result,"AvgMetrics.json"), 'w', encoding="utf8") as outfile:
            json.dump(data, outfile, indent=4)

    if is_tuning == 1:
       SaveAvgMetricsToTuningResults(base_dir, abs_dir_result, result_file_name, f_met)

    CleanFolders(current_dir, is_tuning)