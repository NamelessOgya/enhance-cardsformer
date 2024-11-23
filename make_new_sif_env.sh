#!/bin/csh
csh << EOF
source /etc/profile.d/modules.csh
module load singularity
mkdir singularity
singularity pull ./singularity/cardsformer.sif docker://namelessogya/cardsformer_env
EOF