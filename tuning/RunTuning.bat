echo off

rem First argument is the number of splits in cross validation.
rem Second argument is the path for the configuration file.
rem Third argument is cuda device. should be -1 if running in machine with no GPU. or Gpu device (as 0) otherwise.
rem Fourth argumemt is 1 if to shuffle data file before splitting for cross validation, 0 otherwise
rem Fifth argument is the field name to tune - full field name in json file splitted with "."
rem Sixth argument is 0 or 1 - is this field dimension size of embedding or not
rem Seventh argument is the lowest value in tuning range
rem Eighth argument is the highest value in range
rem Ninth argument is the step size
rem Tenth argument is the encoding of data file as utf-8 or latin-1

set start=%7

FOR /L %%A IN (%start%,%9,%8) DO (
   python create_size_tuning_env.py %2 %3 %5 %%A %6
   ..\\training\\RunTrainingInTuning.bat %1 tuning\\configuration\\%5_%%A.json %3 %4 ..\\training\\ %5_%%A.json %10
   python clean_size_tuning_env.py %5 %%A %6
 )