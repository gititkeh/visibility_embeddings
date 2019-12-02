echo off

rem -s is the number of splits for cross validation.
rem -c is the path for the configuration file
rem -g is the cuda device number
rem -u is if to shuffle cross validation 
rem -e is the encoding of data file as utf-8 or latin-1

SET numof_splits=10
SET config_path="configuration/mohx.json"
SET cuda_device=-1
SET is_shuffle=0
SET encoding="utf-8"


:loop
IF NOT "%1"=="" (
     IF "%1"=="-s" (
         SET numof_splits=%2
         SHIFT
     )
     IF "%1"=="-c" (
         SET config_path=%2
        SHIFT
    )
     IF "%1"=="-g" (
        SET cuda_device=%2
        SHIFT
     )
     IF "%1"=="-u" (
        SET is_shuffle=%2
        SHIFT
    )
     IF "%1"=="-e" (
        SET encoding=%2
        SHIFT
    )
    SHIFT
    GOTO :loop
)

python create_training_env.py %numof_splits% %config_path% %cuda_device% %is_shuffle% %encoding%

FOR /L %%A IN (1,1,%numof_splits%) DO (
   echo Processing %%A
   python run.py train -s serialization/%%A --include-package models experiments/%%A.json
 )

python run_training_result.py %numof_splits% 0