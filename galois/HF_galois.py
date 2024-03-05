import json
import os
import time
from itertools import compress
from math import ceil

import pandas as pd

from src.QueryTree import *
from src.utils import *


def add_more_seq_scan(model,
                      tokenizer,
                      temp_unfilt_ans,
                      temp_questions,
                      old_pr,
                      old_ans,
                      device,
                      max_tries=7,
                      increase_threshold=5,
                      verbose=True):
    """Adds more info to sequential scans by calling the model multiple times

    Args:
        model:  HuggingFace Model
        tokenizer: HuggingFace Tokenizer
        temp_unfilt_ans (List[str]): List answers from the model before parsing
        temp_questions (List[str]): List of questions
        old_pr (List[Dict]): prompt used
        old_ans (str): old answer
        device: cuda or cpu
        max_tries (int, optional): number of trials. Defaults to 7.
        increase_threshold (int, optional): threshold used to stop calling. Defaults to 5.
        verbose (bool, optional): verbose. Defaults to True.

    Returns:
        List[str]: returns a list of augmented answers
    """
    i=0
    final_ans = old_ans

    while i < max_tries:
        if verbose:
            print('#',i+1)
        pr = old_pr +old_ans+'\n\nQ: Give me more.\nA:'
        temp_questions.append("Give me more.")

        #run HF model
        max_len = 400
        input_ids = tokenizer(pr, return_tensors="pt",padding='max_length', max_length = 512).input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(input_ids,temperature=0,max_length=max_len)
        ans = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs][0]

        if verbose:
            print(ans)
        temp_unfilt_ans.append(ans)

        final_ans+=ans

        len1 = len(set(old_ans.split(',')))
        len2 = len(set(old_ans.split(',')+ans.split(',')))
        old_ans= ans
        old_pr = pr
        if (len2-len1)<increase_threshold:
            break
        i+=1
    return final_ans

def answer_batch_questions(model,
                           tokenizer,
                           model_questions,
                           pr,
                           label,
                           cache_fn,
                           max_len,
                           device,
                           verbose=False):
    """Returns answer for a batch request

    Args:
        model: HF model
        tokenizer: HF tokenizer
        model_questions (List[str]): Questions
        pr (List): Prompts
        label (str): experiment label
        cache_fn (str): cache file name
        max_len (int): max length of generation
        device: cuda or cpu
        verbose (bool, optional): verbose. Defaults to False.

    Returns:
        List[str]: Answers of batch request
    """
    cache_fn=label+'_cache.json'
    mode = 'r' if cache_fn in os.listdir('.') else 'w'
    cache=json.load(open(cache_fn,'r')) if mode=='r' else dict()
    final_ans=[]
    for i in range(ceil(len(pr)/10)):
        batch_pr = pr[i*10:i*10+10]
        batch_mq = model_questions[i*10:i*10+10]
        batch_ans = [None]*len(batch_pr)
        to_fetch_indices=[]

        for ind,pro in enumerate(batch_pr):
            pro_key = json.dumps(pro) if isinstance(pro,list) else pro
            if pro_key in cache:
                batch_ans[ind]=cache[pro_key]
            else:
                to_fetch_indices.append(ind)
        if verbose:
            print(f'In Cache: {len(batch_ans)-len(to_fetch_indices)}/{len(batch_ans)}')
        if to_fetch_indices:
            batch_pr_to_fetch = [batch_pr[tfi] for tfi in to_fetch_indices]
            batch_mq_to_fetch = [batch_mq[tfi] for tfi in to_fetch_indices]

            input_ids = tokenizer(batch_pr_to_fetch, return_tensors="pt",padding='max_length', max_length = 512).input_ids
            input_ids = input_ids.to(device)
            outputs = model.generate(input_ids,temperature=0,max_length=max_len)
            ans = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]

            batch_ans_fetched = ans 

            for bmq,bpr,bans in zip(batch_mq_to_fetch,batch_pr_to_fetch,batch_ans_fetched):
                bpr_key = json.dumps(bpr) if isinstance(bpr,list) else bpr
                cache[bpr_key]=bans
                if verbose: print('Added to cache:',bmq,'\n\n')
            json.dump(cache,open(cache_fn,"w"),indent=2)

            for k in range(len(batch_ans_fetched)):
                batch_ans[to_fetch_indices[k]] =batch_ans_fetched[k]
        final_ans.extend(batch_ans)
    return final_ans

def compute_node(model,
                 tokenizer,
                 node,
                 instr,
                 few_shots,
                 inst_funct,
                 label,
                 augmented_question_maps,
                 device,
                 verbose=False):
    """Perform computations for a specific node

    Args:
        model: HF model
        tokenizer: HF tokenizer
        node (Node): Query Tree Node
        instr (str): instruction to be used for the model
        few_shots (List[List[str,str]]): List od few shots
        inst_funct (Callable): function to prepare instruction. not used with chatgpt
        label (str): experiment name
        augmented_question_maps (Dict[str,str]): mapping of node strings to questions
        device: cuda or cpu
        verbose (bool, optional): verbosity. Defaults to False.
    """
    cache_fn=label+'_cache.json'
    mode = 'r' if cache_fn in os.listdir('.') else 'w'
    if verbose: print('Mode: ',mode)
    cache=json.load(open(cache_fn,'r')) if mode=='r' else dict()
    status = 'FINISHED'

    if node.op=='JOIN':
        pr= instr+few_shots+""

        left_questions = [node.key_left.replace('!!x!!',x) for x in node.l.answers[-1]]
        if verbose: print('left questions',left_questions)
        lpr = [pr + inst_funct(x) for x in left_questions]
        batch_left_ans = answer_batch_questions(model,tokenizer,left_questions,lpr,label,cache_fn,50,device,verbose=verbose)
        if verbose: print('left answer',batch_left_ans)

        right_questions = [node.key_right.replace('!!x!!',x) for x in node.r.answers[-1]]
        if verbose: print('right questions',right_questions)
        rpr = [pr + inst_funct(x) for x in right_questions]
        batch_right_ans = answer_batch_questions(model,tokenizer,right_questions,rpr,label,cache_fn,50,device,verbose=verbose)
        if verbose: print('right answer',batch_right_ans)


        left = pd.DataFrame({'left':node.l.answers[-1],"key":batch_left_ans})
        right = pd.DataFrame({'right':node.r.answers[-1],"key":batch_right_ans})
        ans = list (left.merge(right,on='key',how='inner')[node.filter_key])
        if verbose: print('JOINED ANSWER',ans)
        node.answers.append(ans)
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
                print('Unfiltered answer:',ans)
                print('Filtered answer:',ans)
                print('Status: ',status)
        else:
            k = '_'.join(adjusted_node)
            question = node.filled_question if hasattr(node,'filled_question') else augmented_question_maps[k]
            if verbose:
                print('OP: ',k)
                print('Q: ',question)

            # adjust max length generation
            if op == 'SEQ_SCAN':max_len = 400 
            elif op=='FILTER': max_len=2
            else: max_len = 50
            
            if op=='AGGREGATE_PROJ':
                prev_ans = node.l.answers[-1]
            elif  op =='PROJECTION':
                prev_ans = node.l.answers[-1]
            else:
                prev_ans = node.l.answers[-1] if node.l and node.l.answers else []

            if '!!x!!' in question: model_questions = [question.replace('!!x!!',x) for x in prev_ans ]
            else: model_questions = [question]

            node.questions.append(model_questions)
            pr= instr+few_shots+""
            pr = [pr + inst_funct(x) for x in model_questions]

            ans=[]
            if op == 'SEQ_SCAN':
                if k in cache: 
                    ans = cache[k]
                else:
                    old_pr = pr[0]
                    input_ids = tokenizer(old_pr, return_tensors="pt",padding='max_length', max_length = 512).input_ids
                    input_ids = input_ids.to(device)
                    outputs = model.generate(input_ids,temperature=0,max_length=max_len)
                    ans = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]

                    old_pr_key = json.dumps(old_pr) if isinstance(old_pr,list) else old_pr
                    cache[old_pr_key] = ans

                    json.dump(cache,open(cache_fn,"w"),indent=2)

                    old_ans = ans
                    if verbose:
                        print(old_ans)
                    temp_unfilt_ans = []
                    temp_questions = []
                    ans = [add_more_seq_scan(model,tokenizer,temp_unfilt_ans,temp_questions,old_pr,old_ans[0],device,max_tries=7,increase_threshold=5)]
                    node.unfiltered_answers.append(temp_unfilt_ans)
                    node.questions.append(temp_questions)
                    cache[k] = ans
                    json.dump(cache,open(cache_fn,"w"),indent=2)
                    
            else:
                    batch_ans = answer_batch_questions(model,tokenizer,model_questions,pr,label,cache_fn,max_len,device,verbose=verbose)
                    ans.extend(batch_ans)

            node.unfiltered_answers.append(ans)
            if verbose:   print('Unfiltered Answer: ',ans)

            if op == 'SEQ_SCAN':
              if ans:
                  ans = ans[0]
                  ans = ans[:-1] if ans[-1]=='.' else ans
                  ans = ans.replace(' and ','')
                  ans = ans.split(',')
                  ans = [x for x in ans if '.' not in x] #remove 'India.Yemen'
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
                  #ans = ans[0]
                  node.answers.append(ans)
                  if verbose:
                      print('Final Answer: ', ans)
            node.status = status

def compute_tree(model,
                 tokenizer,
                 node,
                 instr,
                 few_shots,
                 inst_funct,
                 label,
                 augmented_question_maps,
                 device,
                 verbose=False):
    """Perform Post-order tree traversal and compute nodes"""
    if node and node.text:
        compute_tree(model,tokenizer,node.l,instr,few_shots,inst_funct,label,augmented_question_maps,device,verbose=verbose)
        compute_tree(model,tokenizer,node.r,instr,few_shots,inst_funct,label,augmented_question_maps,device,verbose=verbose)
        compute_node(model,tokenizer,node,instr,few_shots,inst_funct,label,augmented_question_maps,device,verbose=verbose)

def HF_SPWJ_seq(model,
               tokenizer,
               df,
               instr,
               few_shots,
               inst_funct,
               label,
               augmented_question_maps,
               query_plan_dict,
               device,
               verbose=False):
    """_summary_

    Args:
        model: HF model
        tokenizer: HF tokenizer
        df (pandas DataFrame): dataframe containing thr questions, queries, answers, and database names
        instr (str): instruction to be used for the model
        few_shots (List[List[str,str]]): List od few shots
        inst_funct (Callable): function to prepare instruction
        label (str): experiment name
        augmented_question_maps (Dict[str,str]): mapping of node strings to questions
        query_plan_dict (Dict[str,Node]): maps query to root of query tree
        device: cuda or cpu
        verbose (bool, optional): verbosity. Defaults to False.
    """
    
    
    global b
    # create file for answers
    json.dump([],open(label+".json","w"),indent=3)
    
    # check if there is cache data
    cache_fn=label+'_cache.json'
    mode = 'r' if cache_fn in os.listdir('.') else 'w'
    if verbose: print('Mode: ',mode)
    cache=json.load(open(cache_fn,'r')) if mode=='r' else dict()
    start=time.time()
    for index,row in df.iterrows():

        # get query
        query = row.Query
        # get duckdb con
        con = run_db(db_files[row.Database])
        con.execute("PRAGMA enable_profiling='query_tree';")
        con.execute("PRAGMA explain_output='ALL';")
        print("##################")
        print(query)
        print(row.Database)
        #get logical execution plan
        if verbose:
            print(query)
            print('\n')
        if query in query_plan_dict: 
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
                continue
        
        b=root
        tree_adjust_nodes(root)        
        compute_tree(model,tokenizer,root,instr,few_shots,inst_funct,label,augmented_question_maps,device,verbose=verbose)
        tree_nodes,questions,answers,unfiltered_answers = get_snippet(root,[],[],[],[])

        snippet = {'Gold Question':row.Question,'Gold Answer':row.Answer,'Query':row.Query,
                   'Tree Nodes':tree_nodes,'LP Questions':questions,'LP Answers':answers,
                   'LP Unfiltered Answers':unfiltered_answers,'Status':root.status}

        log = json.load(open(label+".json","r"))
        log.append(snippet)
        json.dump(log,open(label+".json","w"),indent=3)
        print("===================================================================================")
    end=time.time()
    print(f'Time Taken: {(end-start)/60}  seconds.')

def run_question(df,model,tokenizer,device):
    single_question_answers=[]
    for i in range(len(df)):
        question = df.iloc[i].Question
        max_len = 400
        input_ids = tokenizer(question, return_tensors="pt",padding='max_length', max_length = 512).input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(input_ids,temperature=0,max_length=max_len)
        ans = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs][0]
        single_question_answers.append(ans)
    return single_question_answers
