#!/bin/sh

# flags are:
# number_of_splits - splits for cross validation.
# configuration_path is the path for the configuration file
# cuda_device is the cuda device number
# shuffle_cross_validation is if to shuffle cross validation
# encoding is the encoding of data files. For example - utf-8 or latin-1

usage()
{
  echo "Usage: [-s number_of_splits] [-c configuration_path] [-g cuda_device] [-u shuffle_cross_validation] [-e encoding]"
  exit 2
}

# initialize default arguments
numof_splits=10
configuration_path="configuration/mohx.json"
cuda_device=0
shuffle_cross_validation=0
encoding="utf-8"

while getopts 's:c:g:u:e:' b
do
  case $b in
    s) numof_splits=$OPTARG ;;
    c) configuration_path=$OPTARG ;;
    g) cuda_device=$OPTARG ;;
    u) shuffle_cross_validation=$OPTARG ;;
    e) encoding=$OPTARG ;;
  esac
done

python create_training_env.py $numof_splits $configuration_path $cuda_device $shuffle_cross_validation $encoding

i=1
while [ "$i" -le $numof_splits ]; do
    python run.py train -s serialization/"$i" --include-package models experiments/"$i".json
    i=$(( i + 1))
done

python run_training_result.py $numof_splits 0