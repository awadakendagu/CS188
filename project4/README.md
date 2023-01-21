# Project4

三种agent:

**exact inference**

**Approximate Inference(Particle)**

**Joint Particle Filter(DBN)**

evidence:

**Observation**

**elapseTime**

## P0 DiscreteDistribution Class

```python
    def normalize(self):
        totalSum = self.total()
        if self.keys() is None or totalSum == 0:
            return
        for key in self.keys():
            self[key] /= totalSum
        #>>> dist = DiscreteDistribution()
        #>>> dist['a'] = 1
        #self[key]=value
    def sample(self):
        prob = random.random() * self.total() 
        #random 0-1 不一定归一化
        cumulativeSum = 0
        for key in self:
            if cumulativeSum <= prob < cumulativeSum + self[key]:
                return key
            cumulativeSum += self[key]
```

## P1 Observation Probability

return `P(noisyDistance | pacmanPosition, ghostPosition)`

一定要用None 0的干扰

```python
    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        if ghostPosition == jailPosition and noisyDistance is None:
            return 1
        if noisyDistance is not None and ghostPosition == jailPosition:
            return 0
        if noisyDistance is not None and ghostPosition != jailPosition:
            distance = manhattanDistance(pacmanPosition, ghostPosition)
            return busters.getObservationProbability(noisyDistance, distance)
        return 0
```

## P2 Exact Inference Observation

```python
    def observeUpdate(self, observation, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        for ghostPosition in self.allPositions:
            self.beliefs[ghostPosition] *= self.getObservationProb(observation, pacmanPosition,
                                            ghostPosition, jailPosition)
        self.beliefs.normalize()
```

加入ObserationProb的处理

## P3 Exact Inference with Time Elapse

```python
    def elapseTime(self, gameState):
        newProb = DiscreteDistribution() #空字典
        for position in self.allPositions:
            newPosDist = self.getPositionDistribution(gameState, position)
            for pos in newPosDist: #计算新的belief
                newProb[pos] += newPosDist[pos] * self.beliefs[position]
        newProb.normalize()
        self.beliefs = newProb
```

根据前一步估计后一步改写belief

## P4 Exact Inference Full Test

```python
    def chooseAction(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        #可以做出的action
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = [beliefs for i, beliefs in enumerate(self.ghostBeliefs) if livingGhosts[i+1]] 
        #找到没被抓住的幽灵的beliefs
        targets = [ghostPosition.argMax() for ghostPosition in livingGhostPositionDistributions]
        #没被抓住的幽灵的最可能出现的地方
        trueTarget = min(targets, key=lambda target: self.distancer.getDistance(pacmanPosition, target))
        #真正要抓的ghost
        bestAction = min(legal, key=lambda action: self.distancer.getDistance(Actions.getSuccessor(pacmanPosition, action), trueTarget))
        #采取action可以离要抓的ghost最近
        return bestAction
```

**min**的使用

## P5 Approximate Inference Initialization and Beliefs

**particle filtering algorithm** 

```python
    def initializeUniformly(self, gameState):
        self.particles = []
        for i in range(self.numParticles):
            self.particles.append(self.legalPositions[i%len(self.legalPositions)])
        random.shuffle(self.particles)
        #将particle均匀分散到legalPosition
    def getBeliefDistribution(self):
        beliefDistribution = DiscreteDistribution()
        for particle in self.particles:
            beliefDistribution[particle] += 1
        beliefDistribution.normalize()
        return beliefDistribution
        #计算belief
```

## P6 Approximate Inference Observation

  Update beliefs based on the distance observation and Pacman's position.

```python
    def observeUpdate(self, observation, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        predictions = self.getBeliefDistribution()
        for ghostPosition in self.allPositions:
            predictions[ghostPosition] *= self.getObservationProb(observation, pacmanPosition,ghostPosition, jailPosition)
        if predictions.total() == 0:
            self.initializeUniformly(gameState)
            return
        predictions.normalize()
        self.particles = [predictions.sample() for _ in range(self.numParticles)]
```

## P7 Approximate Inference with Time Elapse

```python
    def elapseTime(self, gameState):
        self.particles = [self.getPositionDistribution(gameState, particle).sample() for particle in self.particles]
# 调用函数
```

## P8 Joint Particle Filter Observation

**a dynamic Bayes net**

同时跟踪多个幽灵。每个粒子将代表一个鬼魂位置的元组，这是所有鬼魂当前所在位置的一个样本。

```python
    def initializeUniformly(self, gameState):
        self.particles = []
        samples = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        random.shuffle(samples)
        self.particles = samples[:self.numParticles]
        #itertools.product 笛卡尔积
```

## P9 Joint Particle Filter Observation

和P6相似 只是状态是一个元组 

```python
    def observeUpdate(self, observation, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        ghostStates = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        #获取所有状态
        predictions = self.getBeliefDistribution()
        for state in ghostStates:
            for i in range(self.numGhosts):
                #乘以每一个ghost的ObservationProbability
                jailPosition = self.getJailPosition(i)
                observationI = observation[i]
                ghostPosition = state[i]
                predictions[state] *= self.getObservationProb(observationI,pacmanPosition, ghostPosition, jailPosition)
        predictions.normalize()
        if predictions.total() == 0:
            self.initializeUniformly(gameState)
        else:
            self.particles = [predictions.sample() for _ in range(self.numParticles)]

```

## p10 Joint Particle Filter Time Elapse and Full Test

```python
    def elapseTime(self, gameState):
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions
            # now loop through and update each entry in newParticle...
            newParticle = [self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i]).sample() for i in range(self.numGhosts)]
            newParticles.append(tuple(newParticle))
        self.particles = newParticles
```

