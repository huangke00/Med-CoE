'''
Adapted from https://github.com/lupantech/ScienceQA
'''
import json
from dataclasses import dataclass
from typing import List, Optional


def get_question_text(problem):
    question = problem['Question']
    return question


def get_context_text(problem):
    context = ''
    # if type == 'train':
    #     with open('data/a_okvqa/aok_vqa_id_train_caption.json', encoding='utf-8') as f1:
    #         data = json.load(f1)
    #     context = data.get(qid)
    # if type == 'test':
    #     with open('data/a_okvqa/aok_vqa_id_train_caption.json', encoding='utf-8') as f1:
    #         data = json.load(f1)
    #     context = data.get(qid)
    # if type == 'val':
    #     with open('data/a_okvqa/aok_vqa_id_val_caption.json', encoding='utf-8') as f1:
    #         data = json.load(f1)
    #     context = data.get(qid)
    # return solution
    context = problem['Caption']
    # img_context = problem['caption'] if use_caption else ""
    # context = " ".join([txt_context, img_context]).strip()
    # if context == "":
    #     context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['Choice']

    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
    return choice_txt


def get_origin_answer(problem, options):
    return problem['Answer']


def get_answer(problem, options):
    answer = options[problem['Answer_label']]

    return answer

def get_direct_answer(problem):
    answer = problem['direct_answers']

    return answer
def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture']
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['Solution']

    return solution


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True,
                       WithOutput=False, curr_le_data=None):
    # train  QCM LE
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer the following question by reasoning step by step.\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    elif input_format == "QCG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nSolution: {lecture} {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Solution: {solution} Answer: The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"

    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        else:
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def build_prompt(problems, shot_qids, test_qid, args):
    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    # solution = ""
    # solutions = get_solution_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])
    # solution = solutions[0]+solutions[1]
    # QCM - LE
    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input


def build_train_pair(problems, test_qid, args,type, curr_le_data=None,):
    examples = []

    # test example
    # 'Which of these states is farthest north?'
    question = get_question_text(problems[test_qid])
    # problem['hint']
    context = get_context_text(problems[test_qid])
    # context="N/A"
    # '(A) West Virginia (B) Louisiana (C) Arizona (D) Oklahoma'
    choice = get_choice_text(problems[test_qid], args.options)
    # problem['lecture']

    # solution
    solution = ''
    # if type=='test':
    #     solution = rationals.get(test_qid)
    # else:
    solution = get_solution_text(problems[test_qid])
    # solution = solutions[0] + solutions[1]
    lecture = ''
    # solution = solution1[1]
    # i = 0
    # for s in solution1:
    #     i = i + 1
    #     if i <= 2:
    #         solution = solution + s
    # solution = solution1[0]


    # answer_text = get_origin_answer(problems[test_qid], args.options)
    # answer_option = ''

    answer_option = get_answer(problems[test_qid], args.options)
    # answer = get_direct_answer(problems[test_qid])[0]
    answer = "(" + answer_option + ")"
    # answer = get_origin_answer(problems[test_qid], args.options)

    # test_example ('Question: Which of these states is farthest north?
    # Context: N/A
    # Options: (A) West Virginia (B) Louisiana (C) Arizona (D) Oklahoma
    # Solution:')
    # target Solution:
    # Maps have four cardinal directions, or main directions.
    # Those directions are north, south, east, and west.\nA
    # compass rose is a set of arrows that point to the cardinal directions.
    # A compass rose usually shows only the first letter of each cardinal direction.
    # \nThe north arrow points to the North Pole. On most maps, north is at the top of the map.
    # To find the answer, look at the compass rose. Look at which way the north arrow is pointing.
    # West Virginia is farthest north..
    test_example, target = create_one_example(args.prompt_format,
                                              question,
                                              context,
                                              choice,
                                              answer,
                                              lecture,
                                              solution,
                                              test_example=False, WithOutput=True, curr_le_data=curr_le_data)
    examples.append(test_example)

    # target = target.replace("Answer:", "").strip()
    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input, target


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
