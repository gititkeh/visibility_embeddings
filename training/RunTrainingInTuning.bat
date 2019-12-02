echo off

rem First argument is the number of splits in cross validation.
rem Second argument is the path for the configuration file.
rem Third argument is cuda device. should be -1 if running in machine with no GPU. or Gpu device (as 0) otherwise.
rem Fourth argumemt is 1 if to shuffle data file before splitting for cross validation, 0 otherwise
rem Fifth argument is base folder in case we are running training from somewhare else (as in tuning)
rem Sixth argument is the name of the file to save results
rem Seventh argument is the encoding of data files as utf-8 or latin-1

python %5create_training_env.py %1 %2 %3 %4 %7

FOR /L %%A IN (1,1,%1) DO (
   echo Processing %%A
   python %5run.py train -s %5serialization/%%A --include-package models %5experiments/%%A.json
 )

python %5run_training_result.py %1 1 %6