# Project 1

[TOC]

关于多种search

```python
class Node:
    def __init__(self, state, pre, action, pri=0):
        self.state = state
        self.pre = pre  #记录前一个节点 以便到达终点之后回溯出路径
        self.action = action
        self.pri = pri #记录priority(h(n))
```

## P1 Finding a Fixed Food Dot using Depth First Search 

```python
def depthFirstSearch(problem: SearchProblem):
    visit = []
    fringe = util.Stack()
    fringe.push(Node(problem.getStartState(), None, None))
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state) is True:
            actions = []
            while node.action:
                actions.append(node.action)
                node = node.pre
            actions.reverse()
            return actions
        visit.append(node.state)
        for s in problem.getSuccessors(node.state):
            if s[0] not in visit:
                fringe.push(Node(s[0], node, s[1]))
    return []
```

## P2 Breadth First Search                      

```python
def breadthFirstSearch(problem: SearchProblem):
    visit = []
    fringe = util.Queue()
    fringe.push(Node(problem.getStartState(), None, None))
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state) is True:
            actions = []
            while node.action:
                actions.append(node.action)
                node = node.pre
            actions.reverse()
            return actions
        if node.state not in visit:     #不能丢
            visit.append(node.state)
            for s in problem.getSuccessors(node.state):
                if s[0] not in visit:
                    fringe.push(Node(s[0], node, s[1]))
    return []
```

## P3 Varying the Cost Function 

```python
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    visit = []
    fringe = util.PriorityQueue()
    fringe.push(Node(problem.getStartState(), None, None),0)
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state) is True:
            actions = []
            while node.action:
                actions.append(node.action)
                node = node.pre
            actions.reverse()
            return actions
        if node.state not in visit:     #不能丢
            visit.append(node.state)
            for s in problem.getSuccessors(node.state):
                if s[0] not in visit:
                    fringe.update(Node(s[0], node, s[1], node.pri+s[2]), node.pri+s[2])
    return []
```

## P4 A* search                      

```python
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visit = []
    fringe = util.PriorityQueue()
    fringe.push(Node(problem.getStartState(), None, None),0)
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state) is True:
            actions = []
            while node.action:
                actions.append(node.action)
                node = node.pre
            actions.reverse()
            return actions
        if node.state not in visit:     #不能丢
            visit.append(node.state)
            for s in problem.getSuccessors(node.state):
                if s[0] not in visit:
                    fringe.update(Node(s[0], node, s[1], s[2]+node.pri), s[2]+node.pri+heuristic(s[0],problem))
    return []
```


