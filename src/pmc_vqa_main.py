import numpy as np
import re
import json
import argparse
import random

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from transformers.utils import (
    logging,
)
import os
import torch

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

from model import T5ForConditionalGeneration, T5ForMultimodalGeneration
from pmc_vqa_utils_data import load_pmc_vqa_data_img, PMC_VQA_DatasetImg, img_shape

from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)

import nltk
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])

    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet'],
                        help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE', 'QCG-A', 'QC-EA', 'QCM-EA',
                                 'QM-LE','QC-LE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--start', type=int, default=42, help='random seed')
    parser.add_argument('--end', type=int, default=42, help='random seed')
    parser.add_argument('--test_json_name', type=str, default=42, help='random seed')

    args = parser.parse_args()
    return args


def T5Trainer(
        # dataframe：problems, qids, name_maps, image_features
        dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)  # T5的预训练模型

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']  # 问题
    train_problems = problems['train_problems']
    test_problems = problems['test_problems']
    val_problems = problems['val_problems']
    qids = dataframe['qids']  # 问题id
    train_qids = qids['train']  # 训练
    test_qids = qids['test']  # 测试
    val_qids = qids['val']  # 验证

    if args.evaluate_dir is not None:
        console.log(f"""[evaluate_dir]: Loading {args.evaluate_dir}...\n""")
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)  # 填充的padding
    if args.img_type is not None:  # img_type不为空
        patch_size = img_shape[args.img_type]  # 根据图片处理的模型设置patch_size
        console.log(args.model)
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size)

        # model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, padding_idx=padding_idx,
        #                                                   save_dir=save_dir,
        #                                                   torch_dtype=torch.float16,
        #                                                   # device_map={'': 0},
        #                                                   # quantization_config=BitsAndBytesConfig(
        #                                                   #     load_in_4bit=True,
        #                                                   #     bnb_4bit_compute_dtype=torch.float16,
        #                                                   #     bnb_4bit_use_double_quant=True,
        #                                                   #     bnb_4bit_quant_type="nf4",
        #                                                   #     llm_int8_threshold=6.0,
        #                                                   #     llm_int8_has_fp16_weight=False,
        #                                                   # ),
        #                                                   )  # T5为多模态训练设置的模型

        # peft_config = {
        #     # "peft_type": args.peft_method,
        #     "task_type": "SEQ_2_SEQ_LM",
        #     "inference_mode": False,
        #     "r": 4,
        #     "target_modules": ["q", "v"],
        #     "lora_alpha": 32,
        #     "lora_dropout": 0.1,
        #     "fan_in_fan_out": False,
        #     "bias": "none",
        # }
        # lora_config = LoraConfig(
        #     r=4,
        #     lora_alpha=32,
        #     target_modules=["q", "v"],
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type=TaskType.SEQ_2_SEQ_LM
        # )
        # peft_config = get_peft_config(peft_config)
        # model = prepare_model_for_int8_training(model)
        # print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')

        # model = PeftModelForSeq2SeqLM(model, lora_config)
        # model.print_trainable_parameters()
        name_maps = dataframe['name_maps']
        train_name_maps = name_maps['train_name_maps']
        test_name_maps = name_maps['test_name_maps']
        val_name_maps = name_maps['val_name_maps']
        train_image_features = dataframe['train_image_features']
        test_image_features = dataframe['test_image_features']
        val_image_features = dataframe['val_image_features']

        train_set = PMC_VQA_DatasetImg(
            train_problems,
            train_qids,
            train_name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            train_image_features,

            'train',

        )
        test_set = PMC_VQA_DatasetImg(
            test_problems,
            test_qids,
            test_name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            test_image_features,

            'test',
            args.test_le,

        )

        eval_set = PMC_VQA_DatasetImg(
            val_problems,
            val_qids,
            val_name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            val_image_features,

            'val',

            args.eval_le,

        )

    # else:
    #     model = T5ForConditionalGeneration.from_pretrained(args.model)
    #     train_set = ScienceQADatasetStd(
    #         problems,
    #         train_qids,
    #         tokenizer,
    #         args.input_len,
    #         args.output_len,
    #         args,
    #     )
    #     eval_set = ScienceQADatasetStd(
    #         problems,
    #         val_qids,
    #         tokenizer,
    #         args.input_len,
    #         args.output_len,
    #         args,
    #         args.eval_le,
    #     )
    #
    #     test_set = ScienceQADatasetStd(
    #         problems,
    #         test_qids,
    #         tokenizer,
    #         args.input_len,
    #         args.output_len,
    #         args,
    #         args.test_le,
    #     )
    # label_pad_token_id = -100
    # datacollator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=8
    # )
    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)

        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"
        return answer

    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct += 1
        return {'accuracy': 1.0 * correct / len(targets)}

    # rougel for rationale generation
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    # 在Seq2SeqTrainingArguments中定义训练超参数。
    # 将训练参数与模型、数据集、标记器和数据校对器一起传递给Seq2SeqTrainer。
    # 调用train()模型进行微调。
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=5,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            report_to="none",
            adafactor="true"

        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=5,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format != "QCM-LE" else "rougeL",
            predict_with_generate=args.use_generate,
            load_best_model_at_end=True,
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics_acc if args.prompt_format != "QCM-LE" else compute_metrics_rougel,
        compute_metrics=compute_metrics_rougel
    )

    if args.evaluate_dir is None:
        trainer.train()
        trainer.save_model(save_dir)

    # metrics = trainer.evaluate(eval_dataset=test_set)
    #
    # trainer.log_metrics("test", metrics)
    #
    # trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}

        num_fail = 0
        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)
            if extract_pred != "FAILED":
                if extract_pred in args.options:
                    extract_pred = args.options.index(extract_pred)
                else:
                    extract_pred = random.choice(range(0, len(args.options)))
            else:
                num_fail += 1
                extract_pred = random.choice(range(len(args.options)))  # random choose one option
            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref

        preds = [pred.strip() for pred in preds]
        # print(preds)
        output_data = {
            "num_fail": num_fail,
            # "scores": scores,
            "preds": preds,
            "labels": targets}
        output_prediction_file = os.path.join(save_dir, test_json_name)
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))


if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()
    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.img_type is not None:
        test_json_name, problems, qids, name_maps, train_image_features, val_image_features, test_image_features = load_pmc_vqa_data_img(
            args)  # probelms, test question ids, shot example ids

        dataframe = {'problems': problems, 'qids': qids, 'name_maps': name_maps,
                     'train_image_features': train_image_features, 'val_image_features': val_image_features,
                     'test_image_features': test_image_features}

    T5Trainer(
        dataframe=dataframe,
        args=args
    )
