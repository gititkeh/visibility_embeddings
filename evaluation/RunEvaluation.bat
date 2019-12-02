echo off

rem first argument is the folder containing the trained models in folder "trainedModels" in training folder
rem second argument is the location of the test file for evaluation

python create_evaluation_env.py %1

python run.py evaluate ..\\training\\trainedModels/%1/model.tar.gz ..\\%2 --output-file results/%1/metrics.json --include-package models
