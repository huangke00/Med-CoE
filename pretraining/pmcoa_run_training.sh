#!/usr/bin/env bash
#DSUB -n culture
#DSUB -N 1
#DSUB -A root.wuhkjdxjsjkxyjsxyuan
#DSUB -R "cpu=64;gpu=2;mem=120000"
#DSUB -oo %J.out
#DSUB -eo %J.err

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/9.3.0
module load compilers/cuda/11.3.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

conda activate torch1.11


export NGPU=1;
CUDA_VISIBLE_DEVICES=0 python pmcoa_main.py \
	--model ../flan-alpaca-base \
    --user_msg answer --img_type clip \
    --bs 8 --eval_bs 2 --eval_acc 10 --output_len 512 \
    --final_eval --prompt_format Q-A \
