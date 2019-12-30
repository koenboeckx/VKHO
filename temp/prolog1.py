"""
http://www.openbookproject.net/py4fun/prolog/prolog1.py
"""

import sys, re, copy

def fatal(mesg):
    sys.stdout.write("Fatal: %s\n" % mesg)
    sys.exit(1)
rules = []
trace = False
# In[]:
class Term:
    def __init__(self, s): # expect "x(y, z, ...)"
        if s[-1] != ')': fatal("Syntax error in term: %s" % [s])
        flds = s.split('(')
        self.args = flds[1][:-1].split(',')
        self.pred = flds[0]
    
    def __repr__(self):
        return "{}({})".format(self.pred, ",".join(self.args))

class Rule:
    def __init__(self, s): # expect "term :- term; term; ..."
        flds = s.split(':-')
        self.head = Term(flds[0])
        self.goals = []
        if len(flds) == 2:
            flds = re.sub("\),", ");", flds[1]).split(";")
            for fld in flds:
                self.goals.append(Term(fld))
    
    def __repr__ (self) :
        rep = str(self.head)
        sep = " :- "
        for goal in self.goals :
            rep += sep + str(goal)
            sep = ","
        return rep

class Goal:
    """
    A Goal is a rule at a certain point in its computation.
    'env' contains definitions (so far), 'inx' indexes the current term
    being satisfied, parent is another Goal which spawned this one and
    which we will unify back to when this Goal is complete.
    """
    goalID = 100
    def __init__(self, rule, parent=None, env={}):
        Goal.goalID += 1
        self.id = Goal.goalID
        self.rule = rule
        self.parent = parent
        self.env = copy.deepcopy(env)
        self.inx = 0 # start search with 1st subgoal
    
    def __repr__(self):
        return f"Goal {self.id} rule={self.rule} inx={self.inx} env={self.env}"

def unify(src_term, src_env, dst_term, dst_env):
    """Update dst_env from src. Returns True is unification succeeds."""
    nargs = len(src_term.args)
    if nargs != len(dst_term.args): return 0
    if src_term.pred != dst_term.pred: return 0
    for i in range(nargs):
        src_arg = src_term.args[i]
        dst_arg = dst_term.args[i]
        if src_arg <= 'Z': # meaning src_arg is a capital letter, thus not grounded
            src_val = src_env.get(src_arg) # only not None is value is already set
        else:
            src_val = src_arg
        if src_val: # either constant or defined Variable in source, thus not None
            if dst_arg <= 'Z':
                dst_val = dst_env.get(dst_arg)
                if not dst_val: # value is not yet set
                    dst_env[dst_arg] = src_val     # unification !!
                elif dst_val != src_val: return 0  # unify not possible
            elif dst_arg != src_val: return 0      # unify not possible
    return 1

def search(term, rules, trace=False):
    if trace: print(f'Search {term}')
    goal = Goal(Rule("got(goal):-x(y)"))     # Anything- just get a rule object
    goal.rule.goals = [term]                 # target is the single goal
    if trace: print(f"stack {goal}")
    stack = [goal]                           # Start our search
    while stack:
        c = stack.pop()
        if trace: print(f'    pop {c}')
        if c.inx >= len(c.rule.goals):       # Is this one finished?
            if c.parent == None:             # Yes. Is it our original goal?
                if c.env: print(c.env)     # Yes. Tell user we have a solution
                else    : print("Yes")
                continue
            parent = copy.deepcopy(c.parent) # Otherwise, resume parent goal
            unify(c.rule.head, c.env, 
                  parent.rule.goals[parent.inx], parent.env)
            parent.inx += 1                  # Advance to next goal in body
            if trace: print(f'stack {parent}')
            stack.append(parent)             # let it waits its turn
            continue
        
        # Nothing more to do with this goal
        term = c.rule.goals[c.inx]          # What we want to solve
        for rule in rules:                  # Walk down the rule database
            if rule.head.pred != term.pred: continue
            if len(rule.head.args) != len(term.args): continue
            child = Goal(rule, c)           # A possible subgoal
            ans = unify(term, c.env, rule.head, child.env)
            if ans:                         # if it unifies, stack it up
                if trace: print(f'stack {child}')
                stack.append(child)              
        
def process_file(f, rules):
    env = []
    for sent in f.split('\n'):
        s = re.sub("#.*","",sent[:])   # clip comments and newline
        s = re.sub(" ", "",s)           # remove spaces
        if s == "" : continue
        
        if s[-1] in '?.':
            punc = s[-1];
            s = s[:-1]
        else:
            punc = '.'
            
        if punc == '?' :
            print(s)
            search(Term(s), rules)
        else:
            rules.append(Rule(s))
    return rules

##

if __name__ == '__main__':

    f = """
    boy(bill)
    boy(frank)
    mother(alice, bill)
    father(alex, bill)

    child(J, K) :- mother(K, J)
    child(G, H) :- father(H, G)
    son(X, Y) :- child(X, Y), boy(X)

    son(X, alice)?

    son(bill, X)?
    """

    process_file(f, rules)


# %%
