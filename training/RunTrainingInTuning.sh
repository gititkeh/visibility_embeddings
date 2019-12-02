#!/bin/sh

#First argument is the number of splits in cross validation.
#Second argument is the path for the configuration file.
#Third argument is cuda device. should be -1 if running in machine with no GPU. or Gpu device (as 0) otherwise.
#Fourth argumemt is 1 if to shuffle data file before splitting for cross validation, 0 otherwise
#Fifth argument is base folder in case we are running training from somewhare else (as in tuning)
#Sixth argument is the name of the file to save results
#Seventh argument is the encoding of data files. For eample, utf-8 or latin-1
python "$5"create_training_env.py "$1" "$2" "$3" "$4" "$7"

i=1
while [ "$i" -le "$1" ]; do
   python "$5"run.py train -s "$5"serialization/"$i" --include-package models "$5"experiments/"$i".json
   i=$(( i + 1))
done

python "$5"run_training_result.py "$1" 1 "$6"