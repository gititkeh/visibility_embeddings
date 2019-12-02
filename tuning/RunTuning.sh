#!/bin/sh

# flags are:
# number_of_splits - splits for cross validation.
# configuration_path is the path for the configuration file
# cuda_device is the cuda device number
# shuffle_cross_validation is if to shuffle cross validation
# field_name is the field name to tune - full field name in json file splitted with "."
# is_dimension - is the tuned field a dimension size of embedding or not
# low_lim - the lowest value in tuning range
# high_lim - the highest value in tuning range
# step_size - the step size from low to high
# encoding - the encoding of data files. For example 'utf-8' or 'latin-1'

usage()
{
  echo "Usage: [-s number_of_splits] [-c configuration_path] [-g cuda_device] [-u shuffle_cross_validation] [-f field_name] [-n is_dimension] [-l low_lim] [-h high_lim] [-z step_size] [-e encoding]"
  exit 2
}

# initialize default arguments
numof_splits=10
configuration_path="configuration/mohx.json"
cuda_device=0
shuffle_cross_validation=0
field_name="trainer.optimizer.lr"
is_dimension=0
low_lim=0.016
high_lim=0.016
step_size=0.001
encoding="utf-8"

while getopts 's:c:g:u:f:n:l:h:z:e:' b
do
  case $b in
    s) numof_splits=$OPTARG ;;
    c) configuration_path=$OPTARG ;;
    g) cuda_device=$OPTARG ;;
    u) shuffle_cross_validation=$OPTARG ;;
	f) field_name=$OPTARG ;;
	n) is_dimension=$OPTARG ;;
	l) low_lim=$OPTARG ;;
	h) high_lim=$OPTARG ;;
	z) step_size=$OPTARG ;;
    e) encoding=$OPTARG ;;
  esac
done

start=$low_lim
for i in $(seq "$start" $step_size $high_lim)
do
   python create_size_tuning_env.py $configuration_path $cuda_device $field_name "$i" $is_dimension
   no_zeros=`echo $i | awk ' { sub("\\.*0+$","");print} '`
   if [ $is_dimension = 0 ];
   then
       sh ../training/RunTrainingInTuning.sh $numof_splits tuning/configuration/$field_name_"$no_zeros".json $cuda_device $shuffle_cross_validation ../training/ $field_name_"$i".json $encoding
   else
       sh ../training/RunTrainingInTuning.sh $numof_splits tuning/configuration/$field_name_"$i".json $cuda_device $shuffle_cross_validation ../training/ $field_name_"$i".json $encoding
   fi
   python clean_size_tuning_env.py $field_name "$i" $is_dimension
done