
import sys
sys.path.append(".")
import re
import json 
import pprint
from models.vLLM_server_API import generate_n_with_vLLM_server

prompt_equiv = r'''You are given a mathematical question, the gold/correct answer to the question, and a model generated solution with the final answer in boxed notation. Your job is to predict 'True' if the gold answer and the answer from the model generated solution are equivalent, otherwise 'False' if they are not equivalent.

Question: Imagine you're programming a video game that simulates the flight of a magical bird. The bird's flight path is determined by a spell that affects its speed and direction. According to the spell, the change in the bird's acceleration, minus twice the change in its speed, plus twice its current position equals zero. At the start of the game (time = 0), the bird is at the starting point (position = 0). When the game clock reaches the halfway point of the first day (time = π/2 hours), the bird has reached a position of 1 unit above the starting point. What is the bird's position at a quarter of the way through the first day (time = π/4 hours)?
Gold Answer: 0.322
Model Generated Soution: the given differential equation is:\n\n\\[\n\\frac{d^2x}{dt^2} - 2\\frac{dx}{dt} + 2x = 0\n\\]\n\nthis is a second-order linear homogeneous differential equation with constant coefficients. the characteristic equation for this differential equation is:\n\n\\[\nr^2 - 2r + 2 = 0\n\\]\n\nsolving the characteristic equation:\n\n\\[\nr = \\frac{2 \\pm \\sqrt{4 - 8}}{2} = \\frac{2 \\pm \\sqrt{-4}}{2} = \\frac{2 \\pm 2i}{2} = 1 \\pm i\n\\]\n\nthe roots are complex, so the general solution to the differential equation is of the form:\n\n\\[\nx(t) = e^t (a \\cos t + b \\sin t)\n\\]\n\nnext, we use the initial conditions to find the constants \\(a\\) and \\(b\\). at \\(t = 0\\), the position \\(x(0) = 0\\):\n\n\\[\nx(0) = e^0 (a \\cos 0 + b \\sin 0) = a = 0\n\\]\n\nso, \\(a = 0\\), and the solution simplifies to:\n\n\\[\nx(t) = b e^t \\sin t\n\\]\n\nnext, we use the second initial condition at \\(t = \\frac{\\pi}{2}\\), where the position \\(x\\left(\\frac{\\pi}{2}\\right) = 1\\):\n\n\\[\nx\\left(\\frac{\\pi}{2}\\right) = b e^{\\frac{\\pi}{2}} \\sin \\left(\\frac{\\pi}{2}\\right) = b e^{\\frac{\\pi}{2}} \\cdot 1 = b e^{\\frac{\\pi}{2}} = 1\n\\]\n\nsolving for \\(b\\):\n\n\\[\nb = \\frac{1}{e^{\\frac{\\pi}{2}}}\n\\]\n\nthus, the position function is:\n\n\\[\nx(t) = \\frac{1}{e^{\\frac{\\pi}{2}}} e^t \\sin t = e^{t - \\frac{\\pi}{2}} \\sin t\n\\]\n\nfinally, we need to find the position of the bird at \\(t = \\frac{\\pi}{4}\\):\n\n\\[\nx\\left(\\frac{\\pi}{4}\\right) = e^{\\frac{\\pi}{4} - \\frac{\\pi}{2}} \\sin \\left(\\frac{\\pi}{4}\\right) = e^{-\\frac{\\pi}{4}} \\sin \\left(\\frac{\\pi}{4}\\right) = e^{-\\frac{\\pi}{4}} \\cdot \\frac{\\sqrt{2}}{2} = \\frac{\\sqrt{2}}{2} e^{-\\frac{\\pi}{4}}\n\\]\n\ntherefore, the bird's position at a quarter of the way through the first day (time = \\(\\frac{\\pi}{4}\\) hours) is:\n\n\\[\n\\boxed{\\frac{\\sqrt{2}}{2} e^{-\\frac{\\pi}{4}}}\n\\]
Prediction: True

Question: A company's profit (in thousands of dollars) from selling x units of a product is given by the function P(x) = 3x^2 + 2x - 1. Find the profit when 4 units are sold.
Gold Answer: 54.0
Model Generated Solution: Substitute x = 4 into the profit function:\n\nP(4)=3(4)2+2(4)-1=3(16)+8-1=48+8-1= \\boxed{55}.
Prediction: False

Question: Find the values of x that satisfy the equation x^2 - 7x + 6 = 0.
Gold Answer: [1, 6]
Model Generated Soution: \\[ \\boxed{6 \\text{ and } 1}. \\]
Prediction: True

Question: Imagine you're a data analyst working on a new predictive model. You've created a function that takes complex numbers as inputs to forecast stock prices. However, no matter what input you use, the forecasted price seems to remain the same. Assuming the forecasted price is determined by a mathematical function that takes complex numbers as inputs and the forecasted price never exceeds a certain level for any input, does this mean your forecasting function is applying a uniform prediction across all inputs?
Gold Answer: true
Model Generated Soution: given that \( f = u(z) + iv(z) \) is an entire function and \( |f(z)| \leq 1000 \) for all \( z \in \mathbb{c} \), we can use liouville's theorem. liouville's theorem states that any bounded entire function must be constant. here's the detailed reasoning: entire function: since \( f \) is entire, it is holomorphic (complex differentiable) everywhere in the complex plane \( \mathbb{c} \). bounded: the function \( f \) is bounded by 1000, meaning \( |f(z)| \leq 1000 \) for all \( z \in \mathbb{c} \).application of liouville's theorem: liouville's theorem directly applies here because \( f \) is an entire function that is also bounded. therefore, by liouville's theorem, \( f \) must be a constant function. so, the function \( f \) is constant, and the specific value it takes is some constant \( c \) such that \( |c| \leq 1000 \).\( \\boxed{f \\text{ is a constant function}} \)
Prediction: True
'''


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


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def is_equiv(str1, str2, verbose=False):
    #print("In is_equiv:", str1, str2)
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)

       #print("After stripping:", ss1, ss2)

        if verbose:
            print(ss1, ss2)

        if ss1=="\\text{no}" or ss1=="no":
            if ss2 == "False":
                return True   
        
        if ss1=="\\text{yes}" or ss1=="yes":
            if ss2 == "True":
                return True  

        match = re.search(r"\\text{([a-zA-Z])\)}", ss1)  # Match \text{X) where X is a letter
        if match:
            letter = match.group(1)  # Extract the letter
            if f"({letter})" in ss2:  # Check if (X) is in ss2
                return True


                         
        return ss1 == ss2
    except:
        return str1 == str2


def simple_mathcomparison(solution_a,solution_b,gold_answerb):
    answer_a = last_boxed_only_string(solution_a)
    final_answer_a = remove_boxed(answer_a)
    
    answer_b = last_boxed_only_string(solution_b)
    final_answer_b = remove_boxed(answer_b)
    equiv = is_equiv(final_answer_a, final_answer_b)



def get_final_boxed_answer(string):
    answer = last_boxed_only_string(string)
    return remove_boxed(answer)

def evaluate_accuracy(dataset_path: str, prediction_path: str, verbose: bool = False):
    # Load dataset (gold answers)
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # gold_answers = {item["id"]: item["gold_answer"] for item in dataset}
    # questions = {item["id"]: item["problem"] for item in dataset}
    gold_answers = {item["id"]: item["Answer"] for item in dataset}
    questions = {item["id"]: item["query"] for item in dataset}
    
    # Load generated answers
    generated_answers = {}
    with open(prediction_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            #generated_answers[data["id"]] = data["answer"]
            generated_answers[data["id"]] = data["pred"]
            
    
    total = 0
    correct = 0
    incorrect_cases = []

    for id_, gold in gold_answers.items():
        if id_ not in generated_answers:
            if verbose:
                print(f"[WARNING] Missing ID in generated outputs: {id_}")
            continue

        question =  questions[id_]   
        
        pred_solution = generated_answers[id_]
        gold_str = str(gold)
        pred_str = str(get_final_boxed_answer(pred_solution))

        if is_equiv(pred_str, gold_str, verbose=verbose):
            correct += 1
        else:

            # Use the generator LLM server to evaluate 
            
            prompt2 = f"\n\nQuestion:{question}\nGold Answer:{gold_str}\nModel Generated Soution:{pred_solution}\nPrediction: "
            prompt = prompt_equiv + prompt2
            
            print("Using LLM to determine correctness!")
            print("Model generated solution: ", pred_solution)
            print("Gold Answer: ", gold_str)
            

            is_equiv_llm = generate_n_with_vLLM_server(prompt, max_tokens=20)
            
        

            print("Are they equivalent? LLM Ans:", is_equiv_llm)
            print("-"*50)
            #ch = input("Pause initiated! Enter key to continue:")
            
            
            if "True" in is_equiv_llm or "true" in is_equiv_llm:
                correct+=1     
            else:
                incorrect_cases.append((id_, pred_str, gold_str, pred_solution))
        
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Accuracy: {accuracy*100:.2f}% ({correct}/{total})")


    if verbose and incorrect_cases:
        print("\nIncorrect predictions:")
        for id_, pred, gold, pred_solution in incorrect_cases:
            print(f"ID: {id_}\nPredicted: {pred}\nGold: {gold}\n---")

    return accuracy, incorrect_cases


# === Example usage ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated outputs against gold answers.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--prediction_path", type=str, required=True, help="Path to the generated output JSONL file.")
    parser.add_argument("--verbose", action="store_true", help="Print mismatches and debug info.")

    args = parser.parse_args()
    evaluate_accuracy(args.dataset_path, args.prediction_path, verbose=args.verbose)
