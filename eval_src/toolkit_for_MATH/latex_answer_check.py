# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Based on ToRA (https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py)
# Modified by Weiqi Wang
# ---------------------------------------------------------

import sys
sys.path.append(".")
import re
from eval_src.toolkit_for_MATH.parsing_lib import *

from typing import Union, Any

from copy import deepcopy
from math import isclose
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex

import pprint

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    if string.isdigit() or (string.replace(".", "").isdigit() and string.endswith(".0")):
        string = string.rstrip(".0")

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)

        if verbose:
            print(ss1, ss2)

        if ss1=="\\text{no}" or ss1=="no":
            if ss2 == "False":
                return True   
        
        if ss1=="\\text{yes}" or ss1=="yes":
            if ss2 == "True":
                return True  

        if ss2=="\\text{no}" or ss2=="no":
            if ss1 == "False":
                return True   
        
        if ss2=="\\text{yes}" or ss2=="yes":
            if ss1 == "True":
                return True         

        match = re.search(r"\\text{([a-zA-Z])\)}", ss1)  # Match \text{X) where X is a letter
        if match:
            letter = match.group(1)  # Extract the letter
            if f"({letter})" in ss2:  # Check if (X) is in ss2
                return True


                         
        return ss1 == ss2
    except:
        return str1 == str2


def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string


def has_numbers(input_string: str) -> bool:
    """
    Checks if a string contains a number.
    """
    return any(char.isdigit() for char in input_string)


def has_structure(input_string: str) -> bool:
    """
    Checks if a string contains structured content.
    """
    if "(" in input_string or ")" in input_string or "[" in input_string or "]" in input_string or "\\" in input_string or "<" in input_string or ">" in input_string or "," in input_string or 'x' in input_string or 'y' in input_string or 'z' in input_string:
        return True
    return False


def sympy_parse(input_string: str) -> Any:
    """
    Parsing strings into mathematical expressions using sympy
    """
    for f in [parse_latex, parse_expr]:
        try:
            return f(input_string)
        except:
            pass
    return input_string


def symbolic_equal(a: str, b: str) -> Union[bool, None]:
    """
    Check if two strings are symbolic equal.
    """
    a = sympy_parse(a)
    b = sympy_parse(b)

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), float(N(a)), rel_tol=1e-9) and isclose(N(a), float(N(a)), rel_tol=1e-9):
            return False
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return None


def convert_to_int(input_string: str) -> Union[int, None]:
    """
    Try to convert a string into int. Return `None` if an error occurs.
    """
    try:
        float_s = float(input_string)
        int_s = int(float_s)

        # If a floating-point number is converted to an integer that is very close to itself, then we consider it to be an integer.
        if isclose(int_s, float_s, rel_tol=1e-9):
            return int_s
        return None
    except:
        return None


def convert_to_float(input_string: str) -> Union[float, None]:
    """
    Try to convert a string into float. Return `None` if an error occurs.
    """
    try:
        float_s = float(input_string)
        return float_s
    except:
        return None


def numerical_equal(a: str, b: str) -> Union[bool, None]:
    """
    Check if two strings are numerical equal.
    """
    a_int = convert_to_int(a)
    b_int = convert_to_int(b)

    if a_int is not None and b_int is not None:
        return a_int == b_int

    a_float = convert_to_float(a)
    b_float = convert_to_float(b)

    if a_float is not None and b_float is not None:
        return isclose(a_float, b_float, rel_tol=1e-3)

    return None


def literal_check(model_generated_answer: str, ground_truth: str) -> Union[bool, None]:
    """
    Check if two strings are the same character by character
    """
    model_remove = deepcopy(model_generated_answer).replace(",", " ").replace(" ", "").replace(" ", "")
    gt_remove = deepcopy(ground_truth).replace(",", " ").replace(" ", "").replace(" ", "")

    if model_remove == gt_remove:
        return True

    if has_numbers(model_generated_answer) == False and has_numbers(ground_truth) == False:
        model_generated_answer = model_remove.strip("[]() ")
        ground_truth = gt_remove.strip("[]() ")
        if model_generated_answer == ground_truth:
            return True

    return None


def number_check(model_generated_answer: str, ground_truth: str) -> None:
    """
    Check if two strings have the same mathematical meaning.
    """
    if "," in model_generated_answer or "," in ground_truth:
        return None

    model_generated_answer = remove_prefix_and_suffix(remove_equals(model_generated_answer))
    ground_truth = remove_prefix_and_suffix(remove_equals(ground_truth))

    numerical_equal_result = numerical_equal(model_generated_answer, ground_truth)
    if numerical_equal_result is not None:
        return numerical_equal_result

    symbolic_equal_result = symbolic_equal(model_generated_answer, ground_truth)

    if symbolic_equal_result is not None:
        return symbolic_equal_result

    return None


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def simple_mathcomparison(solution_a,solution_b,gold_answerb):
    answer_a = last_boxed_only_string(solution_a)
    final_answer_a = remove_boxed(answer_a)
    if gold_answerb is False:
        answer_b = last_boxed_only_string(solution_b)
        final_answer_b = remove_boxed(answer_b)
        equiv = is_equiv(final_answer_a, final_answer_b)

    elif gold_answerb is True:
        print("-"*50)
        print("Checking answers using exact match!")
        print("Model final answer:",final_answer_a)
        print("Gold answer:", solution_b)
        equiv = is_equiv(final_answer_a, solution_b)
        print("Are they equivalent? ", equiv)
        print("-"*50)

    
    #print("Is it correct? ", equiv)
    if equiv:
      return True
    
    return False   


def latex_answer_check(model_output, gt_answer, split=None, extract_policy: str="flex", eval_policy: str="aggressive"):
    assert gt_answer is not None
    assert len(gt_answer) > 0

    if model_output is None or model_output == "":
        return False

    # Step 1: Extract answer from response
    if split is not None:
        model_output = extract_answer(model_output, split, extract_policy = extract_policy)
   
    if model_output is None or model_output == "":
        return False

    # Step 2: Remove boxes and perform literal check
    # Compare strings character by character after simple processing including remove $%.
    # First we remove the boxes in the string but keeps the content
    # \boxed{\frac{13}{4}} --> \frac{13}{4}
    model_ans_norm = string_normalization(model_output)
    model_ans_norm_wo_boxes = remove_boxes_keep_content(model_ans_norm)
    gt_norm = string_normalization(gt_answer)
    gt_norm_wo_boxes = remove_boxes_keep_content(gt_norm)

    literal_check_result = literal_check(remove_prefix_and_suffix(model_ans_norm_wo_boxes), remove_prefix_and_suffix(gt_norm_wo_boxes))
    if literal_check_result is not None:
        return literal_check_result

    # Step 3: Attempt to parse -- single
    # Treat a string as a single number/extract a single number from a string and then compare.
    #
    # If we can accept a few mistakes, we try to extract numbers from the answers and compare them
    if eval_policy == "aggressive":
        # We wan't to use raw model_output to keep the $$
        # $13$ meters --> $13$ --> 13
        model_ans_num_lst = search_for_numbers(model_output)

        # We want the original answer has $$
        # This way we are able to consider the answer as a whole
        # We don't want \frac{13}{4} --> [13, 4] to be considered as 2 numbers
        if gt_answer[0] != "$" or gt_answer[-1] != "$":
            gt_num_lst = search_for_numbers("$" + gt_answer + "$")
        else:
            gt_num_lst = search_for_numbers(gt_answer)

        # We want to judge only those answers that contain only one number that represents the full meaning of the original string.
        # If the string still has LaTeX components or variables in addition to this number, then we believe that this number may not represent the meaning of the answer.
        # Here we must be really really careful.
        # x \\leq -5 vs. x \\geq -5
        # (-\\infty, 5) vs. (5, +\\infty)
        # TODO: We may have better methods to check if the numbers are simple enough
        if len(model_ans_num_lst) == 1 and len(gt_num_lst) == 1 and \
            not has_structure(model_output.replace(model_ans_num_lst[0], "")) and \
            not has_structure(gt_answer.replace(gt_num_lst[0], "")):

            model_num = remove_prefix_and_suffix(remove_boxes_keep_content(remove_text_box_only(model_ans_num_lst[0])))
            gt_num = remove_prefix_and_suffix(remove_boxes_keep_content(remove_text_box_only(gt_num_lst[0])))
            parse_result = number_check(model_num, gt_num)  #! problematic

            # As an additional method of judgment, even if it returns False we can't say that the answer is wrong, it could be caused by an unreasonable extraction of numbers
            if parse_result is True:
                return True

    # Here we do the same thing to the whole string
    model_wo_text = remove_prefix_and_suffix(model_ans_norm)
    gt_wo_text = remove_prefix_and_suffix(gt_norm)
    parse_result = number_check(model_wo_text, gt_wo_text)
    if parse_result is not None:
        return parse_result

    # If none of the above ways can determine whether the answer is correct or incorrect, then return incorrect
    return False
