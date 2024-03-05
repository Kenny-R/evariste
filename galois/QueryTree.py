from typing import List

box_head1 = '┌───────────────────────────┐'

box_head2 = '┌─────────────┴─────────────┐'

box_end1  = '└─────────────┬─────────────┘'

box_end2  = '└───────────────────────────┘'

class Node:
  def __init__(self):
    self.text=None
    self.l=None
    self.r=None
    self.op=None
    self.args=None
    self.questions = []
    self.unfiltered_answers = []
    self.answers = []
    
def parse_query_tree(s: str) -> Node:
    root = Node()
    p=root
    for ss in s:
        if ss ==box_head1 or ss== box_head2:
          seach_text=True
          text=[]
        elif  box_end1 in ss or box_end2 in ss:
          search_text=False
          p.text = text
          p.op = p.text[0]
          p.args = p.text[1:]
          p.l=Node()
          p=p.l
        else:
          t=ss.replace('|','').replace('─','').replace('│','').replace('└┬┘','').replace('└┘','').strip()#.replace(' ','')
          if t:
              text.append(t)
              if ')' in text[-1] and '(' not in text[-1]: text = text[:-2]+[" ".join(text[-2:])]
    # remove projection after aggregation
    if root.op=='PROJECTION' and root.l and root.l.op=='AGGREGATE' and root.args == root.l.args: root=root.l
    
    return root

def print_tree(p: Node):
  """prints tree

  Args:
      p (Node)
  """
  if p and p.text:
    print(p.text)
    print_tree(p.l)
    print_tree(p.r)



def get_tree_elements(p: Node,a: List) -> List[str]:
  """Travserse tree and append to List"""
  if p and p.text:
    a.append(p.text)
    get_tree_elements(p.l,a)
    get_tree_elements(p.r,a)
  return a


def get_snippet(p: Node,tree_nodes: List,questions: List,answers: List,unfiltered_answers: List):
  """Traverses tree and stores Node, questions, answers and unfiltered answers"""
  if p and p.text:
      get_snippet(p.l,tree_nodes,questions,answers,unfiltered_answers)
      get_snippet(p.r,tree_nodes,questions,answers,unfiltered_answers)
      tree_nodes.append(p.adjusted_nodes)
      questions.append(p.questions)
      answers.append(p.answers)
      unfiltered_answers.append(p.unfiltered_answers)
  return tree_nodes,questions,answers,unfiltered_answers