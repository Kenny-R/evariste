import os
import time
import json
from itertools import compress

from openai import OpenAI

client = OpenAI()
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed

from .utils import *
from .QueryTree import *


@retry(wait=wait_fixed(61), stop=stop_after_attempt(6))
def completion_with_backoff_chat(**kwargs):
    return client.chat.completions.create(**kwargs)

def construct_chat_dict(role,content):
  return {"role":role,"content":content}

def construct_message_dict(instruction,question_answers):
    messages=[]
    messages.append(construct_chat_dict("system",instruction))
    for x in question_answers:
        q=x[0]
        messages.append(construct_chat_dict("user",q))
        a=x[1]
        messages.append(construct_chat_dict("assistant",a))    
    return messages

def add_more_seq_scan(temp_unfilt_ans: List[str],
                      model_arch: str,
                      temp_questions: List[str],
                      old_pr: List[Dict],
                      old_ans: str,
                      max_tries=7,
                      increase_threshold=5,
                      verbose=True) -> List[str]:
    """Adds more info to sequential scans by calling the model multiple times

    Args:
        temp_unfilt_ans (List[str]): List answers from the model before parsing
        model_arch (str): OpenAI model architecture
        temp_questions (List[str]): List of questions
        old_pr (List[Dict]): prompt used
        old_ans (str): old answer
        max_tries (int, optional): number of trials. Defaults to 7.
        increase_threshold (int, optional): threshold used to stop calling. Defaults to 5.
        verbose (bool, optional): verbose. Defaults to True.

    Returns:
        List[str]: returns a list of augmented answers
    """
    # Quitar para que se pueda hacer las peticionse a chatgpt
    # return old_ans
    i=0
    final_ans = old_ans

    while i < max_tries:
        if verbose:
            print('#',i+1)

        # construct prompt
        pr = old_pr + [ construct_chat_dict("assistant",old_ans)] + [ construct_chat_dict("user","Give me more.")]
        messages = pr
        temp_questions.append("Give me more.")

        max_len = 400
        # run gpt3.5
        response = completion_with_backoff_chat(model=model_arch, 
                                                messages=messages, 
                                                temperature=0,
                                                max_tokens= max_len)
    	# get answer
        ans = response.choices[0].message.content

        if verbose:
            print(ans)
        temp_unfilt_ans.append(ans)
        
        if ans != 'Unknown':
            final_ans+=f', {ans}'

        # parse old and new answers and check their lengths
        len1 = len(set(old_ans.split(',')))
        len2 = len(set(old_ans.split(',')+ans.split(',')))

        old_ans= ans
        old_pr = pr
        # if not a lot of new info, then stop
        if (len2-len1)<increase_threshold:
            break
        i+=1
    return final_ans

def answer_batch_questions_chat(model_questions: List[str],
                                pr: List,
                                label: str,
                                cache_fn: str,
                                model_arch: str,
                                max_len: int,
                                verbose=False) -> List[str]:
    """Returns answer for a batch request

    Args:
        model_questions (List[str]): Questions
        pr (List): Prompts
        label (str): experiment label
        cache_fn (str): cache file name
        model_arch (str): model architecture
        max_len (int): max length of generation
        verbose (bool, optional): verbose. Defaults to False.

    Returns:
        List[str]: Answers of batch request
    """
    # Quitar para que puedas hacer las peticiones a chatgpt
    # return [""]*len(pr[:])

    # cache file name
    cache_fn=label+'_cache.json'
    # if cache is available then read, otherwise create a new one
    mode = 'r' if cache_fn in os.listdir('.') else 'w'
    cache=json.load(open(cache_fn,'r')) if mode=='r' else dict()

    batch_pr = pr[:]   
    batch_mq = model_questions[:]  
    batch_ans = [None]*len(batch_pr)
    to_fetch_indices=[]

    for ind,pro in enumerate(batch_pr):
        # if prompt in cache then return the answer directly
        pro_key = json.dumps(pro) if isinstance(pro,list) else pro
        if pro_key in cache:
            batch_ans[ind]=cache[pro_key]
        else:
            # otherwise we need to call the API and fetch the answer
            to_fetch_indices.append(ind)
    if verbose:
        print(f'In Cache: {len(batch_ans)-len(to_fetch_indices)}/{len(batch_ans)}')
    if to_fetch_indices:
        batch_pr_to_fetch = [batch_pr[tfi] for tfi in to_fetch_indices]
        batch_mq_to_fetch = [batch_mq[tfi] for tfi in to_fetch_indices]

        for i in range(len(batch_pr_to_fetch)):
            response = completion_with_backoff_chat(model=model_arch, 
                                                    messages=batch_pr_to_fetch[i], 
                                                    temperature=0,
                                                    max_tokens= max_len)
            if hasattr(response,'choices'):
                batch_ans_fetched = response.choices[0].message.content
                
                bpr = batch_pr_to_fetch[i]
                bmq= batch_mq_to_fetch[i]

                # add to cache
                bpr_key = json.dumps(bpr) if isinstance(bpr,list) else bpr
                cache[bpr_key]=batch_ans_fetched
                if verbose:
                    print('Added to cache:',bmq,'\n\n')
                    print('Response: ', batch_ans_fetched)
                    print(f'Len of cache: {len(cache)}')
                json.dump(cache,open(cache_fn,"w"),indent=2)
            else: 
                batch_ans_fetched='qlq'

            batch_ans[to_fetch_indices[i]] =batch_ans_fetched

    return batch_ans

def compute_node(node,
                 model_arch,
                 instr,
                 few_shots,
                 inst_funct,
                 label,
                 augmented_question_maps,
                 verbose=False):
    """Perform computations for a specific node

    Args:
        node (Node): Query Tree Node
        model_arch (tr): model architecture
        instr (str): instruction to be used for the model
        few_shots (List[List[str,str]]): List od few shots
        inst_funct (Callable): function to prepare instruction. not used with chatgpt
        label (str): experiment name
        augmented_question_maps (Dict[str,str]): mapping of node strings to questions
        verbose (bool, optional): verbosity. Defaults to False.
    """

    cache_fn=label+'_cache.json'
    mode = 'r' if cache_fn in os.listdir('.') else 'w'
    if verbose: print('Mode: ',mode)
    cache=json.load(open(cache_fn,'r')) if mode=='r' else dict()

    status = 'FINISHED'

    if node.op=='JOIN':
        # for the JOIN node we will need to compute the child nodes
        pr=construct_message_dict(instr,few_shots)

        # compute left child node
        left_questions = [node.key_left.replace('!!x!!',x) for x in node.l.answers[-1]]
        
        if verbose: print('left questions',left_questions)
        
        lpr = [pr + [ construct_chat_dict("user",x)] for x in left_questions]
        
        batch_left_ans = answer_batch_questions_chat(left_questions,lpr,label,cache_fn,model_arch,50,verbose=verbose)
        
        if verbose: print('left answer',batch_left_ans)

        # compute right child node
        right_questions = [node.key_right.replace('!!x!!',x) for x in node.r.answers[-1]]
        
        if verbose: print('right questions',right_questions)
        
        rpr = [pr + [ construct_chat_dict("user",x)] for x in right_questions]
        
        batch_right_ans = answer_batch_questions_chat(right_questions,rpr,label,cache_fn,model_arch,50,verbose=verbose)
        
        if verbose: print('right answer',batch_right_ans)

        # construct dfs
        left = pd.DataFrame({'left':node.l.answers[-1],"key":batch_left_ans})
        right = pd.DataFrame({'right':node.r.answers[-1],"key":batch_right_ans})

        # perform merge
        ans = list (left.merge(right,on='key',how='inner')[node.filter_key])
        
        if verbose: print('JOINED ANSWER',ans)
        
        node.answers = [ans]
        node.status = status
        
        return

    node_text_list = node.text
    if verbose: print(node_text_list)
    
    adjusted_nodes_list = node.adjusted_nodes
    
    if verbose:
        print('Tree Nodes: ',(["_".join(x) for x in adjusted_nodes_list]))

    for adjusted_node in  adjusted_nodes_list:
        op = adjusted_node[0]    

        if 'AGGREGATE_count_star()' in "_".join(adjusted_node) or 'AGGREGATE_OP_count' in "_".join(adjusted_node):
            try:
                ans = len(node.answers[-1] if len(adjusted_nodes_list)>1 else node.l.answers[-1])
            except:
                status = 'FAILED AGGREGATE OPERATION'
                ans=[]
            
            node.questions.append('COUNT')
            node.status = status
            node.unfiltered_answers.append(ans)
            node.answers.append(ans)
            
            if verbose:
                print('Answer: ',ans)


        elif 'AGGREGATE_OP' in op:

            func = adjusted_node[1]
            node.questions.append(func+'('+",".join(node.answers[-1])+')')

            try:
                prev_ans = node.l.ans[-1]
                prev_ans = [x[:-1] if x[-1]=='.' else x for x in prev_ans]

                if func!='count':
                    # this means we need to do some mathematical operations, so we need to remove as much text as possible
                    numer_ans = [replace_units(x) for x in prev_ans]
                    numer_ans = [re.sub("[^0-9.*]","",x) for x in numer_ans]
                    numer_ans = [x for x in numer_ans if x]
                    numer_ans = [x[:-1] if x[-1]=='.' else x for x in numer_ans]
                    numer_ans = [x for x in numer_ans if x]
                    numer_ans = [eval(x) for x in numer_ans]
                    numer_ans = [ float(x) for x in numer_ans]
                else:
                    numer_ans = [x for x in numer_ans if x]
                    
                if verbose:
                    print('Numerical Parsing: ',numer_ans)
                ans = map_func(func)(numer_ans)
                node.status = 'FINISHED'

            except:
                node.status = 'FAILED AGGREGATE OPERATION'
                ans=[]

            node.unfiltered_answers.append(ans)
            node.answers.append(ans)

            if verbose:
                print('Function used: ',map_func(func))
                print('Unfiltered answer:',ans)
                print('Filtered answer:',ans)
                print('Status: ',status)
        else:
            k = '_'.join(adjusted_node)
            print(k)
            question = node.filled_question if hasattr(node,'filled_question') else augmented_question_maps[k]
            if verbose:
                print('OP: ',k)
                print('Q: ',question)

            # adjust max length generation
            if op == 'SEQ_SCAN':max_len = 400 
            elif op=='FILTER': max_len=1 if 'turbo' not in model_arch else 2
            else: max_len = 50
            
            if op=='AGGREGATE_PROJ':
                prev_ans = node.l.answers[-1]
            elif  op =='PROJECTION':
                prev_ans = node.l.answers[-1]
            else:
                prev_ans = node.l.answers[-1] if node.l and node.l.answers else []

            # replace question placeholders
            if '!!x!!' in question: model_questions = [question.replace('!!x!!',x) for x in prev_ans ]
            else: model_questions = [question]

            node.questions.append(model_questions)


            pr=construct_message_dict(instr,few_shots)
            pr = [pr +[ construct_chat_dict("user",x)] for x in model_questions]
            ans=[]
            
            if op == 'SEQ_SCAN':
                if k in cache: 
                    ans = cache[k]
                else:
                    print('NOT IN CACHE')
                    old_pr = pr[0]
                    if verbose:
                       print('RUNNING SEQUENTIAL SCANS...')
                       print(old_pr)
                    response = completion_with_backoff_chat(model=model_arch, 
                                                            messages=old_pr, 
                                                            temperature=0,
                                                            max_tokens= max_len)
                    
                    old_pr_key = json.dumps(old_pr) if isinstance(old_pr,list) else old_pr
                    cache[old_pr_key] = response.choices[0].message.content
                    json.dump(cache,open(cache_fn,"w"),indent=2)

                    old_ans = response.choices[0].message.content
                    if verbose:
                        print(f'before add more items \n{old_ans}')

                    temp_unfilt_ans = []
                    temp_questions = []
                    
                    ans = [add_more_seq_scan(temp_unfilt_ans,model_arch,temp_questions,old_pr,old_ans,max_tries=10,increase_threshold=5)]
                    
                    node.unfiltered_answers.append(temp_unfilt_ans)
                    node.questions.append(temp_questions)
                    cache[k] = ans
                    json.dump(cache,open(cache_fn,"w"),indent=2)
                    
            else:   
                    # batch questions for LLMs no operations
                    batch_ans = answer_batch_questions_chat(model_questions,pr,label,cache_fn,model_arch,max_len,verbose=verbose)
                    ans.extend(batch_ans)

            node.unfiltered_answers.append(ans)
            if verbose:   print('Unfiltered Answer: ',ans)

            # once node computation is done, we perform some post-processing
            if op == 'SEQ_SCAN':
              if ans:
                  ans = ans[0]
                  ans = ans[:-1] if ans[-1]=='.' else ans
                  ans = ans.replace(' and ','')
                  ans = ans.split(',')
                  ans = [x.strip() for x in ans if '.' not in x] #remove 'India.Yemen'
                  ans = list(set(ans))
                  ans =[x for x in ans if x]
              node.answers.append(ans)
              if verbose:
                  print('Final Answer: ',ans)

            elif op == 'FILTER':
                filtered_ans=[]
                if ans:
                    bool_index = [x.replace('.','').strip() for x in ans]
                    bool_index = [x=='Yes' for x in bool_index]
                    filtered_ans = list(compress(node.l.answers[-1], bool_index))
                if verbose:
                    print('Final Answer: ', filtered_ans)
                node.answers.append(filtered_ans)
                if not filtered_ans: 
                    status = 'EMPTY'
                    if verbose:
                        print('EMPTY')
                    break

            elif op != 'AGGREGATE_OP' and op!='AGGREGATE_count_star()':
                  node.answers.append(ans)
                  if verbose:
                      print('Final Answer: ', ans)
            if verbose: print('\n')
            node.status = status

def compute_tree(node,
                 model_arch,
                 instr,
                 few_shots,
                 inst_funct,
                 label,
                 augmented_question_maps,
                 verbose=False):
    """Perform Post-order tree traversal and compute nodes"""
    if node and node.text:
        compute_tree(node.l,model_arch,instr,few_shots,inst_funct,label,augmented_question_maps,verbose=verbose)
        compute_tree(node.r,model_arch,instr,few_shots,inst_funct,label,augmented_question_maps,verbose=verbose)
        compute_node(node,model_arch,instr,few_shots,inst_funct,label,augmented_question_maps,verbose=verbose)

def GPT_SPWJ(model_arch,
            query,
            instr,
            few_shots,
            inst_funct,
            label,
            augmented_question_maps,
            query_plan_dict,
            verbose=False):
    '''
    Queries LLM using query tree

    Args:
        model_arch (str): model architecture
        query: A string that contains the SQL query to execute
        instr (str): instruction to be used for the model
        few_shots (List[List[str,str]]): List od few shots
        inst_funct (Callable): function to prepare few shot examples
        label (str): experiment name
        augmented_question_maps (Dict[str,str]): mapping of node strings to questions
        query_plan_dict (Dict[str,Node]): maps query to root of query tree
        verbose (bool, optional): verbosity. Defaults to False.
    '''
    
    global b

    # create file for answers
    json.dump([],open(label+".json","w"),indent=3)

    #get logical execution plan
    if verbose:
        print(query)
        print('\n')
    if query in query_plan_dict: 
        print('Query in dict')
        root = query_plan_dict[query]
    else:
       print('Query not in dict')

    # adjust query tree nodes
    tree_adjust_nodes(root)
    
    # compute nodes (Aqui es donde se hace las peticiones a chatGpt)
    compute_tree(root,model_arch,instr,few_shots,inst_funct,label,augmented_question_maps,verbose=verbose)
    
    # get outputs
    tree_nodes,questions,answers,unfiltered_answers = get_snippet(root,[],[],[],[])

    snippet = {'Tree Nodes':tree_nodes,
               'LP Questions':questions,
               'LP Answers':answers, 
               'LP Unfiltered Answers':unfiltered_answers,
               'Status':root.status}

    log = json.load(open(label+".json","r"))
    log.append(snippet)
    json.dump(log,open(label+".json","w"),indent=3)
    time.sleep(3)
    if verbose: print("===================================================================================")

def GPT_SPWJ_seq(model_arch,
                df,
                instr,
                few_shots,
                inst_funct,
                label,
                augmented_question_maps,
                query_plan_dict,
                verbose=False):
    """Queries LLM using query tree

    Args:
        model_arch (str): model architecture
        df (pandas DataFrame): dataframe containing thr questions, queries, answers, and database names
        instr (str): instruction to be used for the model
        few_shots (List[List[str,str]]): List od few shots
        inst_funct (Callable): function to prepare few shot examples
        label (str): experiment name
        augmented_question_maps (Dict[str,str]): mapping of node strings to questions
        query_plan_dict (Dict[str,Node]): maps query to root of query tree
        verbose (bool, optional): verbosity. Defaults to False.
    """
    global b
    # create file for answers
    json.dump([],open(label+".json","w"),indent=3)
    
    for index,row in df.iterrows():
        # get query
        query = row.Query

        # get duckdb con
        con = run_db(db_files[row.Database])
        con.execute("PRAGMA enable_profiling='query_tree';")
        con.execute("PRAGMA explain_output='ALL';")

        #get logical execution plan
        if verbose:
            print(query)
            print('\n')
        if query in query_plan_dict: 
            print('Query in dict')
            root = query_plan_dict[query]
        else:
            try:
                con.execute("EXPLAIN "+query.replace('"',"'"))
                s = con.fetchall()[0][1].split('\n')
                if verbose:
                    print("\n".join(s))
                    print('\n')
                root = parse_query_tree(s)
            except:
                print('Canot parse query')
                continue
        # adjust query tree nodes
        tree_adjust_nodes(root)
        # compute nodes
        compute_tree(root,model_arch,instr,few_shots,inst_funct,label,augmented_question_maps,verbose=verbose)
        # get outputs
        tree_nodes,questions,answers,unfiltered_answers = get_snippet(root,[],[],[],[])

        snippet = {'Gold Question':row.Question,'Gold Answer':row.Answer,'Query':row.Query,
                   'Tree Nodes':tree_nodes,'LP Questions':questions,'LP Answers':answers,
                   'LP Unfiltered Answers':unfiltered_answers,'Status':root.status}

        log = json.load(open(label+".json","r"))
        log.append(snippet)
        json.dump(log,open(label+".json","w"),indent=3)
        time.sleep(3)
        if verbose: print("===================================================================================")

def run_question(df,inst_chatgpt,fewshot_chatgpt):
    single_question_answers=[]
    for i in range(len(df)):
        question = df.iloc[i].Question
        message = construct_message_dict(inst_chatgpt,fewshot_chatgpt)
        question_message = message + [ construct_chat_dict("user",question)]
        response =  completion_with_backoff_chat(model='gpt-3.5-turbo', messages=question_message, temperature=0,max_tokens= 400)
        ans = response['choices'][0]['message']['content']
        single_question_answers.append(ans)
    return single_question_answers


def run_CoT(df,inst_chatgpt,cot_ex):
    cot=[]
    for i in (range(len(df))):
        question = df.iloc[i].Question
        pr = construct_message_dict(inst_chatgpt+' Think step-by-step. Here is an example:',[['What is the largest continent',cot_ex]]) +[ construct_chat_dict("user",question)] +[ construct_chat_dict("assistant",'Let us think step by step.')]
        response =  completion_with_backoff_chat(model='gpt-3.5-turbo', messages=pr, temperature=0,max_tokens= 400)
        ans = response['choices'][0]['message']['content']
        cot.append(ans)
    return cot