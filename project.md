得分最高的中随机选择一项的index

```python
legalMoves = gameState.getLegalActions()
scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
bestScore = max(scores)
bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
chosenIndex = random.choice(bestIndices) # Pick randomly among the best
return legalMoves[chosenIndex]
```

  `shift+tab` 可以把所选代码统一往前4个空格

