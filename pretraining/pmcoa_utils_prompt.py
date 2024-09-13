'''
Adapted from https://github.com/lupantech/ScienceQA
'''
from dataclasses import dataclass
from typing import List, Optional


def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['caption']
    # img_context = problem['caption'] if use_caption else ""
    context = txt_context
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']

    if len(choices) > 5:
        choices = choices[0:5]

    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
    return choice_txt


def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]


def get_answer(problem):
    answer = problem["choices"][0]

    return answer


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    # solution = problem['rationales']
    solution = problem['solutions']
    return solution


def create_one_example(format,t_t_v, question, caption,
                       WithOutput=False, curr_le_data=None):
    # train  QCM LE
    input_format, output_format = format.split("-")
    input = ""
    output = ""
    if input_format == "Q":
        input = f"Context: {question}\n"

    if output_format == 'A':
        output = f"This is the Caption: {caption}."

    return input, output


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
    solution = get_solution_text(problems[test_qid])
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


def build_train_pair(t_t_v,problems, test_qid, args, curr_le_data=None):
    examples = []

    # test example

    question = "Introduce this image in detail"

    # problem['hint']
    caption = get_context_text(problems[test_qid], args.use_caption)

    test_example, target = create_one_example(args.prompt_format,
                                              t_t_v,
                                              question,
                                              caption,
                                              # choice,
                                              # answer,
                                              # lecture,
                                              # solution,
                                             WithOutput=True, curr_le_data=curr_le_data)

    return test_example, target


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
