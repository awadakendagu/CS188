# Project 2

[TOC]

## P1 Reflex Agent         

根据下一状态评估

```python
    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action) #后继状态
        newPos = successorGameState.getPacmanPosition() #Pacman的后继位置
        newFood = successorGameState.getFood() 
        newGhostStates = successorGameState.getGhostStates() #后继状态的幽灵状态
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #幽灵被恐吓的时间
        foods = newFood.asList()
        nearestGhostDis = 1e9
        for ghostState in newGhostStates:
            ghostX, ghostY = ghostState.getPosition()
            ghostX = int(ghostX)
            ghostY = int(ghostY)
            if ghostState.scaredTimer == 0: #没被吓着的
                nearestGhostDis = min(nearestGhostDis,manhattanDistance((ghostX, ghostY), newPos))
        nearestFoodDis = 1e9
        for food in foods:
            nearestFoodDis = min(nearestFoodDis, manhattanDistance(food, newPos))
        if not foods:
            nearestFoodDis = 0
        return successorGameState.getScore() - nearestFoodDis / 3   - 7 / (nearestGhostDis + 1)
```

## P2 Minimax   

```python
def getAction(self, gameState: GameState):
        def minimaxSearch(gameState, agentIndex, depth):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), Directions.STOP
            if agentIndex == 0:
                ret = maximizer(gameState, agentIndex, depth)
            else:
                ret = minimizer(gameState, agentIndex, depth)
            return ret

        def maximizer(gameState, agentIndex, depth):
            legalMoves = gameState.getLegalActions(agentIndex)
            maxScore = -1e9
            maxAction = Directions.STOP
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newScore = minimaxSearch(successorGameState, (agentIndex+1) % gameState.getNumAgents(), depth-1)[0]
                if newScore > maxScore:
                    maxScore, maxAction = newScore, action
            return maxScore, maxAction

        def minimizer(gameState, agentIndex, depth):
            legalMoves = gameState.getLegalActions(agentIndex)
            minScore = 1e9
            minAction = Directions.STOP
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newScore = minimaxSearch(successorGameState, (agentIndex+1) % gameState.getNumAgents(), depth-1)[0]
                if newScore < minScore:
                    minScore, minAction = newScore, action
            return minScore, minAction

        return minimaxSearch(gameState, 0, self.depth*gameState.getNumAgents())[1]
```

## P3 Alpha-Beta Pruning   

```python
    def getAction(self, gameState: GameState):
        def alphaBetaSearch(gameState, agentIndex, depth, alpha, beta):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), Directions.STOP
            if agentIndex == 0:
                ret = maximizer(gameState, agentIndex, depth, alpha, beta)
            else:
                ret = minimizer(gameState, agentIndex, depth, alpha, beta)
            return ret

        def maximizer(gameState, agentIndex, depth, alpha, beta):
            legalMoves = gameState.getLegalActions(agentIndex)
            maxScore = -1e9
            maxAction = Directions.STOP
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newScore = alphaBetaSearch(successorGameState, (agentIndex+1) % gameState.getNumAgents(), depth-1, alpha, beta)[0]
                if newScore > maxScore:
                    maxScore, maxAction = newScore, action
                    if maxScore > beta:
                        return maxScore, maxAction
                    alpha = max(alpha, maxScore)
            return maxScore, maxAction

        def minimizer(gameState, agentIndex, depth, alpha, beta):
            legalMoves = gameState.getLegalActions(agentIndex)
            minScore = 1e9
            minAction = Directions.STOP
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newScore = alphaBetaSearch(successorGameState, (agentIndex+1) % gameState.getNumAgents(), depth-1, alpha, beta)[0]
                if newScore < minScore:
                    minScore, minAction = newScore, action
                    if minScore < alpha:
                        return minScore, minAction
                    beta = min(beta, minScore)
            return minScore, minAction

        return alphaBetaSearch(gameState, 0, self.depth*gameState.getNumAgents(), -1e9, 1e9)[1]     
```

## P4 Expectimax

```python
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimaxSearch(gameState, agentIndex, depth):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), Directions.STOP
            if agentIndex == 0:
                ret = maximizer(gameState, agentIndex, depth)
            else:
                ret = expectation(gameState, agentIndex, depth)
            return ret

        def maximizer(gameState, agentIndex, depth):
            legalMoves = gameState.getLegalActions(agentIndex)
            maxScore = -1e9
            maxAction = Directions.STOP
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newScore = expectimaxSearch(successorGameState, (agentIndex+1) % gameState.getNumAgents(), depth-1)[0]
                if newScore > maxScore:
                    maxScore, maxAction = newScore, action
            return maxScore, maxAction

        def expectation(gameState, agentIndex, depth):
            legalMoves = gameState.getLegalActions(agentIndex)
            expScore = 0
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                expScore += expectimaxSearch(successorGameState, (agentIndex+1) % gameState.getNumAgents(), depth-1)[0]
            expScore /= len(legalMoves)
            return expScore, Directions.STOP

        return expectimaxSearch(gameState, 0, self.depth*gameState.getNumAgents())[1] 
```

## Q5 Evaluation Function

```python
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition() #Pacman的后继位置
    newFood = currentGameState.getFood() 
    newGhostStates = currentGameState.getGhostStates() #后继状态的幽灵状态
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #幽灵被恐吓的时间
    foods = newFood.asList()
    nearestGhostDis = 1e9
    ghostCount = 0
    for ghostState in newGhostStates:
        ghostX, ghostY = ghostState.getPosition()
        ghostX = int(ghostX)
        ghostY = int(ghostY)
        if ghostState.scaredTimer == 0: #没被吓着的
            nearestGhostDis = min(nearestGhostDis,manhattanDistance((ghostX, ghostY), newPos))
            ghostCount += 1
        else:
            nearestGhostDis = min(nearestGhostDis,manhattanDistance((ghostX, ghostY), newPos)*3)
    if ghostCount == 0:
        nearestGhostDis *=3
    nearestFoodDis = 1e9
    for food in foods:
        nearestFoodDis = min(nearestFoodDis, manhattanDistance(food, newPos))
    if not foods:
        nearestFoodDis = 0
    return currentGameState.getScore() + 10 / (nearestFoodDis + 1) - 100 / (nearestGhostDis + 1)
```

