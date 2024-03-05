import re
import subprocess
from fractions import Fraction
from subprocess import run
from typing import Callable, Dict

import duckdb
import numpy as np
import pandas as pd


db_files = {'geo':'data/spider_files/spider/database/geo/geography-db.added-in-2020.sqlite',
            'world_1':'data/spider_files/spider/database/world_1/world_1.sqlite',
            'flight_4':'data/spider_files/spider/database/flight_4/flight_4.sqlite',
            'flight_2':'data/spider_files/spider/database/flight_2/flight_2.sqlite',
            'singer':'data/spider_files/spider/database/singer/singer.sqlite'
            }

def run_db(db_name):
  con = duckdb.connect(database=':memory:')
  table_names = get_table_names(db_name)
  for table_name in table_names:
      bash_cmd = f'sqlite3 -header -csv {db_name} "SELECT * from {table_name};">data/spider_files/csv_files/{table_name}.csv'
      data = run(bash_cmd,capture_output=True,shell=True)
      con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('data/spider_files/csv_files/{table_name}.csv');")
  return con


def get_table_names(db_name: str):
    """Gets table names of a DB"""
    cmd = f'sqlite3 {db_name} .tables'
    cmd = cmd.split(' ')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    a=proc.stdout.readlines()
    a=" ".join([x.decode("utf-8").replace('\n','')  for x in a])
    table_names = a.split(' ')
    table_names = [x.strip() for x in table_names if x.strip()]
    return table_names


def get_parsable_queries(x):
    try:
      db=x.Database
      q=x.Query
      con = run_db(db_files[db])
      con.execute("PRAGMA enable_profiling='query_tree';")
      con.execute("PRAGMA explain_output='ALL';")
      con.execute("EXPLAIN "+q.replace('"',"'")) 
      s = con.fetchall()[0][1].split('\n')
      return True
    except:
      return False

def adjust_nodes(node):
    """adjusts node operation

    Args:
        node (Node)
    """
    if node.op=='JOIN': 
        node.adjusted_nodes = ['JOIN']
        return 
    # split multiple attributes
    adjusted_nodes=[]
    if node.op in {'FILTER','PROJECTION','AGGREGATE'} and len(node.args)>1:
        adjusted_nodes = [[node.op,x] for x in node.args]
    else:   
        adjusted_nodes = [node.text]


    # split aggregates
    # ['AGGREGATE', 'sum(SurfaceArea)']  ==> ['AGGREGATE_PROJ', 'SurfaceArea','AGGREGATE_OP', 'SurfaceArea']
    
    adjusted_nodes2=[]

    for n in adjusted_nodes:
        if n[0]=='AGGREGATE' and n[1]!='count_star()':
            func, arg = re.findall("([a-zA-Z_]+)\((.*)\)",n[1])[0]
            adjusted_nodes2.extend([['AGGREGATE_PROJ',arg],['AGGREGATE_OP',func]])

        else: adjusted_nodes2.append(n)

    node.adjusted_nodes = adjusted_nodes2


def augment_questions(question_dict: Dict[str,str]) -> Dict[str,str]:
    """Augments questions with commands to better parse LLM output.

    Args:
        question_dict (Dict[str,str])

    Returns:
        Dict[str,str]: augmented questions
    """
    d={}
    for k in question_dict:
        question = question_dict[k]
        first_word = question.split(' ')[0]
        if first_word in {'Is','Does'}: question +=' Answer with Yes or No only.'
        if first_word =='List': question +=' Separate them by a comma. List as much as you can.'
        d[k] = question
    return d


def tree_adjust_nodes(head):
    """Passes through the query tree and adjusts the nodes

    Args:
        head (Node)
    """
    if head and head.text:
        adjust_nodes(head)
        tree_adjust_nodes(head.l)
        tree_adjust_nodes(head.r)

def replace_units(x:str) -> str:
    x=x.lower()
    return x.replace('thousand','*10**3').replace('million','*10**6').replace('billion','*10**9').replace('trillion','*10**12')

def map_func(func: str) -> Callable:
    if func=='sum': return np.sum
    if func=='avg': return np.mean
    if func=='max': return np.max
    if func == 'count': return len

def get_cardinality(results):
    answer_dims = []
    for i in range(len(results)):
        x=results[i]
        ans = x['LP Answers'][-1]
        num_rows = len(ans)
        cols=[]
        if len(ans) ==2: indices=[1]
        elif len(ans)==4: indices=[1,3]
        else: indices=[0]
        for k in indices:
            if isinstance(ans[k],int) or isinstance(ans[k],float): 
                if ans[k]:  
                    cols.append(1)
                else:
                    cols.append(0)
                num_rows-=1
            else:
                t = [x for x in ans[k] if 'Unknown' not in x]
                cols.append(len(t) if t else 0)
        num_cols = max(cols)
        answer_dim = (num_rows,num_cols)
        answer_dims.append(answer_dim)
    return answer_dims

def compute_metric(df,col_prefix,metric):
    if metric == 'NUM_QUESTIONS':
        num = len(df[f'{col_prefix} Answer'].dropna())

        return f'{col_prefix}_num_qa',num

    else:
        mean_metric = np.mean(df[f'{col_prefix} {metric}'].dropna().apply(lambda x:float(Fraction(x))))
        return f'{col_prefix}_mean_{metric}',mean_metric

def compute_metric_type(df,col_prefix,metric,type_):
    if type_ == 'ALL':
         k,v = compute_metric(df,col_prefix,metric)
         return 'ALL',k,v

    sel_df = df[df['Final Type']==type_]
    k,v = compute_metric(sel_df,col_prefix,metric)
    return type_,k,v


def get_type_metric_df(df,col_prefix,metric):
    all_types = ['ALL','Sel','Agg','Join']
    d=[]
    for t in all_types:
        v = compute_metric_type(df, col_prefix,metric,t)[-1]
        d.append(['Mean ' + metric,t,v])
    d = pd.DataFrame(d)
    d.columns=['Metric','Type',col_prefix]
  
    return d

def get_final_type(x):
    if 'J' in x: return 'Join'
    if 'A' in x: return 'Agg'
    return 'Sel'