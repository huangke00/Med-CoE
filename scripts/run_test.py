import subprocess

uids = list(range(0, 20))

for uid in uids:
    cmds = [
        'python', 'pmc_vqa_main.py',
        f'--start={uid * 100}',
        f'--end={(uid+1)*100}',
        f'--test_json_name=test_cot_{uid+1}.json',
        '--model', 'flan-alpaca-base',
        '--user_msg', 'rationale',
        '--img_type', 'clip',
        '--bs', '4',  # Convert to string
        '--eval_bs', '4',  # Convert to string
        '--eval_acc', '10',  # Convert to string
        '--output_len', '512',  # Convert to string
        '--final_eval',
        '--prompt_format', 'QCMG-A',  # Add a comma
        '--evaluate_dir', '/home/share/z5g2b2tn/home/chen/22xzq/mm-cot/mm-cot-main/med_vqa1/experiments/answer_model-flan-alpaca-base_clip_QCMG-A_lr5e-05_bs16_op512_ep10_novision/checkpoint-110600'
    ]
    subprocess.run(cmds)
