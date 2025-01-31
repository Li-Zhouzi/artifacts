#! /bin/bash
# 
#
mkdir ${1}
for i in $(seq 1 64); 
do
    ./scripts/profile_fixed_gpu.sh $i ${1}
done
python3 profile_cleaning.py ${1}