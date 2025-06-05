import json
def has_retrieval_key(d):
    return any("retrieval" in key for key in d)


file_name = "outputs_MCTS/run_outputs/BRIGHT/Qwen2.5-7B-Instruct/TESTRUN/answer_sheets/Question 0009 - Final Solutions.json"
if __name__ == "__main__":
    
    # Open and load JSON file
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)  # Loads JSON into a Python dictionary
    
    question = (data[0]['trace']['0']['user_question'])
    print(f"Question: {question}")
    total_solutions = 0 
    retrieval_solutions = 0
    # Print or process the data
    for i in range(len(data)):
        total_solutions +=1
        question = (data[i]['trace']['0']['user_question'])
        answer = ""
        
        if not has_retrieval_key(data[i]['trace']['0']['ost_step']):
            continue
            
        retrieval_solutions+=1

        if 'ost_step' not in data[i]['trace']['0']:
           continue

        else:
            ost_steps =  data[i]['trace']['0']['ost_step'] 
           
            for step_no in ost_steps:
                
                
                if 'retrieval' in step_no:
                    answer += "\n" + ost_steps[step_no][0]
                    step_sol = ost_steps[step_no][0] 
                    print("Retrieval:", step_sol)
                    print("-"*80)
                
                elif 'querygeneration' in step_no:
                    answer += "\n" + ost_steps[step_no][0]
                    step_sol = ost_steps[step_no][0]
                    print("Query:", step_sol)
                    print("-"*80)
                
                else:
                    if type(ost_steps[step_no]) == list:
                        answer += "\n" + ost_steps[step_no]
                        step_sol = ost_steps[step_no]

                    else:
                        answer += "\n" + ost_steps[step_no]
                        step_sol = ost_steps[step_no]

                    print("OST Step: ", step_no, ",",step_sol) 
               
                
            if 'direct_answer' in data[i]['trace']['0']:
                answer += "\n" + data[i]['trace']['0']['direct_answer']['text']   
                print("Step :", data[i]['trace']['0']['direct_answer']['text'])
                print("*"*50) 
        
        
        #print(f"Full Solution {i}: {answer}")
        print("-"*70)
        ch = input("Enter a key to continue:")

    print("Total number of retrieval solutions:", retrieval_solutions/total_solutions)
