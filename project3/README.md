## P1 Value Iteration                      

```python
    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        while self.iterations != 0:#递归深度
            newValues = util.Counter() # store new value of a state
            updateFlag = util.Counter() # store whether a state has been updated
            for state in self.mdp.getStates():#对每一个状态进行搜索
                bestAction = self.computeActionFromValues(state) 
                #选出各个Action中的V值最大的
                if bestAction:
                    newValue = self.computeQValueFromValues(state, bestAction)
                    #计算出该行动的q值作为该状态的值
                    newValues[state] = newValue
                    updateFlag[state] = 1
            for state in self.mdp.getStates():#更新
                if updateFlag[state]:
                    self.values[state] = newValues[state]
            self.iterations -= 1
```

## P2 Policies                      

analysis.py 设置参数使agent采取固定最佳策略

question2b甚是刁钻 难绷

## P3 Q-Learning     

 和P1差不多只是算法不一样

```python
    def update(self, state, action, nextState, reward: float):
        oldValue = self.getQValue(state, action)
        maxNewQ = self.computeValueFromQValues(nextState)
        self.values[(state, action)] = (1 - self.alpha) * oldValue + self.alpha * (reward + self.discount * maxNewQ)
```

## P4 Epsilon Greedy   

只需简单修改                   

```python
	def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action
```

## P5 Q-Learning and Pacman                      

无需操作 只是将Q-Learning运用在了Pacman

## P6 Approximate Q-Learning

 和Q-Learning类似

```python
	def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        currentValue = self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        # weights = self.getWeights()
        maxNewQ = self.computeValueFromQValues(nextState)
        newValue = reward + self.discount * maxNewQ
        diff = newValue - currentValue
        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]
```

