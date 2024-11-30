#!/bin/bash
#################################################################
# A40 1GPU Job Script for HPC System "KAGAYAKI" 
#                                       2022.3.3 k-miya
#################################################################

#PBS -N chokosen-loveti
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=2:ngpus=1

cd {CURRENTDIR}
source /etc/profile.d/modules.csh

module load singularity
mkdir -p ./tmp
chmod 755 ./tmp

current_dir=$(pwd)
echo "==== make singularity container ====="

singularity exec --nv --bind ./tmp:/container/tmp ./singularity/cardsformer.sif /bin/bash  <<EOF
cd {CURRENTDIR}/Cardsformer
echo "==== run python ====="

echo "==== start train ===="
pwd
python train_prediction.py

echo "==== start prediction ===="
# python train_policy.py

echo "==== start evaluation ===="
# eval.py


EOF
