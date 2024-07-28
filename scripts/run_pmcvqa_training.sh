#!/usr/bin/env bash
#DSUB -n culture
#DSUB -N 1
#DSUB -A root.z5g2b2tn
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

export NGPU=3;
CUDA_VISIBLE_DEVICES=0,1,2  python pmc_vqa_main.py \
    --model model/flan-alpaca-base\
    --user_msg answer --img_type clip \
    --bs 8 --eval_bs 14 --eval_acc 10 --output_len 512  --input_len 512 --epoch 5 --lr 5e-5\
    --final_eval --prompt_format QCM-LE \


# export start=0
# export end=3
# export json=test_cot_1.json
# CUDA_VISIBLE_DEVICES=0,1 python pmc_vqa_main.py \
#     --model checkpoint-658612\
#     --user_msg answer --img_type clip \
#     --bs 4 --eval_bs 1 --eval_acc 10 --output_len 512  --input_len 512 --epoch 6 --lr 5e-5\
#     --final_eval --prompt_format QCM-EA \




#训练
# '''
# CUDA_VISIBLE_DEVICES=0,1,2 python pmc_vqa_main.py \
#      --model flan-alpaca-base\
#     --user_msg answer --img_type clip \
#     --bs 4 --eval_bs 1 --eval_acc 10 --output_len 512  --input_len 512 --epoch 30 --lr 5e-5\
#     --final_eval --prompt_format QCMEA \
# '''



#
#CUDA_VISIBLE_DEVICES=0,1 python okvqa_main.py \
#    --model allenai/unifiedqa-t5-base \
#    --user_msg rationale --img_type clip \
#    --bs 2 --eval_bs 1 --eval_acc 10 --output_len 512 \
#    --final_eval --prompt_format QCM-LE\
#    --evaluate_dir experiments/rationale_allenai-unifiedqa-t5-base_clip_QCM-LE_lr5e-05_bs4_op512_ep20_okvqa_attention
### answer inference
#CUDA_VISIBLE_DEVICES=0,1 python okvqa_main.py \
#    --model t5-base \
#    --user_msg rationale --img_type clip \
#    --bs 4 --eval_bs 2 --eval_acc 10 --output_len 64 \
#    --final_eval --prompt_format QCG-A \


