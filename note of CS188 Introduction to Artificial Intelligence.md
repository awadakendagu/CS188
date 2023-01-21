# note of CS188 Introduction to Artificial Intelligence

## Intro to AI

AI:Computational Rationality

rational:maximize expected utility

design rational agents

- An agent is an entity that perceives and acts.
- A rational agent selects actions that maximize
  its (expected) utility.
- Characteristics of the percepts, environment,
  and action space dictate techniques for
  selecting rational actions
- This course is about:
  - General AI techniques for a variety of problem
    types
  -  Learning to recognize when and how a new
    problem can be solved with an existing technique
    Agent

在人工智能中，目前的主要问题是创建一个理性（rational）  agent，一个有特定目标或偏好并会针对这些目标试图执行一系列操作（actions）来得到最优解的实体。理性agent存在于为给定的agent示例而量身定制的特定环境中。举一个简单的例子，一个西洋棋agent的环境就是它用来与对手对弈的虚拟棋盘，而对应的操作就是棋子的移动。一个环境和其中的agents一起创造了一个世界。

## Search

### agent

#### **reflex agents**

- Choose action based on current percept (and maybe memory)
- May have memory or a model of the world’s current state
- Do not consider the future consequences of their actions
- Consider how the world IS

#### **Planning agents:**

- Ask “what if”
- Decisions based on (hypothesized) consequences of actions
- Must have a model of how the world evolves in response to actions
- Must formulate a goal (test)
- Consider how the world WOULD BE

反射（reflex）agent不会考虑它的操作的后果，而只是根据世界的当前状态而选择一个操作。而计划（planning）agent则有一个世界模型并用这个模型来模拟执行不同的操作，远远胜过了反射agent。于是，agent就能确定操作的假想结果并选择最佳操作。这样的模拟“智能”在这个意义上符合人类在任何情况下尝试决定最佳动作的行为——预测。

A **search problem** consists of:

- A state space
- A successor function (with actions, costs)
- A start state and a goal test

A **solution** is a sequence of actions (a plan) which transforms the start state to a goal state

The **world state** includes every last detail of the environment

A **search state** keeps only the details needed for planning (abstraction)

世界状态包含一个给定状态的所有信息，而搜索状态仅包含对计划（主要是为了空间效率）必要的信息。我们将介绍本课程的一大亮点——吃豆人Pacman来解释这些概念。这个游戏很简单：吃豆人必须探索一个迷宫，吃掉所有的的小豆子并且不被游荡的恶灵吃掉。如果它吃了大豆子，它将在一段时间内无敌并且能吃掉恶灵赚取分

 我们来考虑这个游戏的一个变式，迷宫中只有吃豆人和豆子。在这种情况下我们能给出两种不同的搜索问题：**路径规划**和**光盘行动**（**pathing** and **eat-all-dots**）。路径规划需要找出从位置 1 到 位置2 的最佳方案，而光盘行动要求用尽可能短的时间吃掉迷宫中所有的豆子。下面分别列出了两个问题的状态、操作、后继函数和目标测试函数：

 Problem: Pathing

- States: (x,y) location
- Actions: NSEW
- Successor: update location only
- Goal test: is (x,y)=END

 Problem: Eat-All-Dots

- States: {(x,y), dot booleans}
- Actions: NSEW
- Successor: update location and possibly a dot boolean
- Goal test: dots all false

### Uninformed Search Methods

#### **Depth-First Search**

**What nodes DFS expand?**

- Some left prefix of the tree.
- Could process the whole tree!
- If m is finite, takes time O(b^m)

**How much space does the fringe take?**

- Only has siblings on path to root, so O(bm)

**Is it complete?**

- m could be infinite, so only if we prevent
  cycles (more later)


**Is it optimal?**

- No, it finds the “leftmost” solution, regardless of depth or cost

#### Breadth-First Search

**What nodes does BFS expand?**
- Processes all nodes above shallowest solution
-  Let depth of shallowest solution be s
- Search takes time O(b^s)

**How much space does the fringe take?**

- Has roughly the last tier, so O(b^s)

**Is it complete?**

- s must be finite if a solution exists, so yes!

**Is it optimal?**

- Only if costs are all 1 (more on costs later)

#### Uniform-Cost Search

**Strategy**: expand a cheapest node first:(**距离起点**)
Fringe is a priority queue (priority: cumulative cost)

**What nodes does UCS expand?**

-  Processes all nodes with cost less than cheapest solution!
- if that solution costs C* and arcs cost at least
  e , then the “effective depth” is roughly C*/e*
- *Takes time O(b^C*/e) (exponential in effective depth)

**How much space does the fringe take?**

- Has roughly the last tier, so O(b C*/e)

**Is it complete?**

- Assuming best solution has a finite cost and minimum arc cost is positive, yes!

**Is it optimal?**

- Yes! (Proof next lecture via A*)

### Informed Search

#### Heuristics*（启发式搜索）*

**A heuristic is:**

- A function that estimates how close a state is to a goal

- Designed for a particular search problem

- Examples: Manhattan distance, Euclidean distance for
  pathing

  Manhatten Distance: |x1-x2|+|y1-y2|

**priority**: h(n)

#### Greedy Search

- **描述**：贪婪搜索总是选择有**最小启发值（lowest heuristic value）**的节点来扩展，这些节点对应的是它认为最接近目标的状态。
- **边缘描述**：贪婪搜索的操作和UCS相同，具有优先队列边缘表示。不同之处在于，贪婪搜索使用**估计前进代价（estimated forward cost）**，而不是**计算后退代价（computed backward cost）**（通往状态的路径上各边的权重之和）。
- **完备性和最优性**：如果存在一个目标状态，贪婪搜索无法保证能找到它，它也不是最优的，尤其是在选择了非常糟糕的启发函数的情况下。在不同场景中。它的行为通常是不可预测的，有可能一路直奔目标状态，也有可能像一个被错误引导的DFS一样并遍历所有的错误区域。

**priority**: g(n)

#### A*Search

- **描述**：A*搜索总是选择有**最低估计总代价（lowest estimated total cost）**的边缘节点来扩展，其中总代价是指从起始节点到目标节点的全部代价。
- **边缘表示**：和贪婪搜索及UCS一样，A*搜索也用一个优先队列来表示它的边缘。和之前一样，唯一的区别在于优先权选择的方法。A*搜索将UCS使用的全部后退代价（到达状态的路径上各边权重之和）和贪婪搜索使用的估计前进代价（启发值）通过相加联合起来，有效地得到了从起始到目标的**估计总代价（estimated total value）**。由于我们想将从起始到目标的总代价最小化，这是一个很好的选择。
- **完备性和最优性**：在给定一个合适的启发函数（我们很快就能得到）时，A*搜索既是完备的又是最优的。它结合了目前我们发现的所有其他搜索策略的优点，兼备贪婪搜索的高搜索速度以及UCS的完备性和最优性。

**priority**: f(n) = g(n) + h(n)

#### Admissibility 可纳性

g(n) 精确的 但是h(n)不能精确 但是可以找h*(n) 使最优

**key: relax problem -> h***

**定义.** 为从一个给定节点n到达目标状态的真正的最佳前进代价h*(n)，我们能将可纳性约束数学表示为：

任意n, 0 <= h(n) <= h*(n)

**定理.** 对于一个给定的搜索问题，如果一个启发式函数h满足可纳性约束，使用含有h的A*树搜索能得到最优解。

对于一个特定的问题可能会想到很多满足的h*

越接近h 走的路径越少 但是h*计算量可能更大

#### Graph Search

将封闭集存储为一个不相交集合而不是一个队列。将它存储为队列需要花费O(n)个操作来检查**资格（membership）**，这会抵消图搜索本来的优势。

另一个需要注意的是即使是在可纳的启发式下，它也会**破坏A*的最优性**。

#### consistency 一致性

**定义.** 任意A,C  h(A) - h(C) <= cost(A,C)

**定理.** 对于一个给定的搜索问题，如果启发式函数h满足一致性约束，对这个搜索问题使用有h的A*图搜索能得到一个最优解。

## Constraint Satisfaction Problems 约束满足问题 CSP

**Search** Assumptions about the world: a single agent, deterministic actions, fully observed state, discrete state space

**CSP** 黑盒 只知道后继和是否是goal

约束满足问题是**NP-hard**问题，意味着没有已知的算法能在多项式时间内找到该问题的解。

![](.\image\C1.jpg)

**Example: N-Queens**

![](.\image\C2.jpg)

![](.\image\C3.jpg)

### **Constraint Graphs** **约束图**

每一条边表示一种约束

二元关系

![](.\image\C4.jpg)

多元关系

![](.\image\C5.jpg)

### **Solving Constraint Satisfaction Problems**

#### **Backtracking search**

约束满足问题的传统解法是使用一种叫做**回溯搜索（Backtracking search）**的算法。回溯搜索是对专门针对约束满足问题的DFS的最优化，主要改善了两个法则：

1. 规定变量的顺序，并按照这个顺序选择变量的赋值。因为赋值是可以互换的（例如，令WA=Red，NT=Green 等价于 令NT=Green，WA=Red），这是有效的。
2. 在给变量选择赋值时，只选择不与已经分配了的值冲突的赋值。如果不存在这样的值，回溯并返回前一个变量，并改变其赋值。

![](.\image\C6.jpg)

### Filtering

我们要讲的CSP的第一个改进是**过滤(filtering)**，它通过移除我们已知一定会进行回溯的值来检查我们能否提前将一些还未赋值的变量的域修剪掉。

#### forward checking 前向检测法

一个简洁的过滤的方法是**前向检测法(forward checking)**，每当一个值被赋给了一个变量  ，就删去与  共享一个约束的未赋值变量的域，这个域如果被赋值的话会违反约束。每当一个新的变量被赋值，我们就能前向检测，并删去约束图中与刚赋值的变量相邻的未赋值变量的域。回到我们地图染色的例子，有一些未赋值的变量以及它们可能的值

![](.\image\C7.jpg)

#### Arc Consistency  弧相容

**每次检查所有的约束关系**

将一个CSP的约束图每条无向边理解为两条指向相反方向的有向边。每个这样的有向边叫做**弧（arc）**。弧相容算法的工作过程如下：

![](.\image\C8.jpg)

![](.\image\C9.jpg)

![](.\image\C10.jpg)

### Ordering

我们已经提到过，在解决CSP问题时，我们对涉及到的变量和值都进行排序。在实践中，使用两个基本原则——**最小剩余值（minimum remaining values）**和**最少约束值（least constraining value）**来“动态”地计算下一个变量和相应的值，通常要高效得多：

- **最小剩余值Minimum Remaining Values (MRV)**：当选择下一个要赋值的变量时，用MRV策略能选择有效剩余值最少的未赋值变量（即约束最多的变量）。这很直观，因为受到约束最多的变量也最容易耗尽所有可能的值，并且如果没有赋值的话最终会回溯，所以最好尽早赋值。
![](.\image\C11.jpg)
- **最少约束值Least Constraining Value (LCV)**：同理，当选择下一个要赋值的值时，一个好的策略就是选择从剩余未分配值的域中淘汰掉最少可能的那一个。要注意的是，这要求更多的计算（比方说，对每个值重新运行弧相容/前向检测或是其他过滤方法来找到其LCV），但是取决于用途，仍然能获得更快的速度。

![](.\image\C12.jpg)

### Structure

#### tree-structured CSP

弧一致性是一个更广义的一致性概念——**k-相容（k-consistency）**的子集，它在强制执行时能保证对于CSP中的任意k个节点，对于其中任意k-1个节点组成的子集的赋值都能保证第k个节点能至少有一个满足相容的赋值。这个想法可以通过**强k-相容（strong k-consistency）**的思想进行进一步拓展。一个具有强k-相容的图拥有这样的性质，任意k个节点的集合不仅是k-相容的，也是k-1, k-2, …,1-相容的。于是，在CSP中更高阶的相容计算的代价也更高。在这一广义的相容的定义下，我们可以看出弧相容等价于2-相容。

![](.\image\C13.jpg)

![](.\image\C14.jpg)

![](.\image\C15.jpg)

![](.\image\C16.jpg)

![](.\image\C17.jpg)

#### cutset conditioning 割集条件设置

树形结构算法能推广到树形结构相近的的CSP中。割集条件设置包括首先找到一个约束图中变量的最小子集，这样一来删去他们就能得到一棵树（这样的子集称为**割集（cutset）**）。举个例子，在我们的地图染色例题中，South Australia（SA）是可能的最小的割集：

![](.\image\C18.jpg)

![](.\image\C19.jpg)

### Iterative Improvement 迭代改进

作为我们感兴趣的最后一个话题，回溯搜索并不是唯一的解决约束满足问题的算法。另一个应用广泛的算法是**本地搜索（local search）**，它的思想非常简单直接，但是相当有效。本地搜索通过迭代改进来工作——从一些随机的变量赋值开始，**反复选择违反约束最多的变量并将其重置为违反约束最少的值**（这一策略称为**最小冲突启发式（min-conflicts heuristic）**）。通过这一策略，我们能既节省时间又节省空间地解决类似N皇后问题这样的约束满足问题。比如，在下面这个有四个皇后的例子中，我们仅用两步就得到了一个解

![](.\image\C20.jpg)

其实，不仅是对有任意大的N皇后问题，本地搜索对任意生成的CSP的运行时间几乎是常数并且成功率也非常高！然而，抛开这些优势，本地搜索并不具备完备性和最优性，所以并不一定能得到最优解。另外，有一个关键的比例，在其附近，本地搜索的代价极高。

**限制多限制少都很快** 介于两者之间某个值很慢

![](.\image\C21.jpg)

**对于找到一个可行方法适用，但是不能找到最优解（局部最优停止）**

## Game Tree

### Adversarial Games

Many possible **formalizations**, one is:
§ States: S (start at s0 )
§ Players: P={1...N} (usually take turns)
§ Actions: A (may depend on player / state)
§ Transition Function: SxA -> S
§ Terminal Test: S -> {t,f}
§ Terminal Utilities: SxP -> R

**Solution** for a player is a **policy**: S -> A

**Zero-Sum Games**：争抢同一个东西 一个人的多了 另一个人的就少了

游戏的一方要使其最大化，而另一方要使其最小化，这样就能让双方进行直接竞争。在吃豆人中，这一变量就是玩家得分，玩家通过高效吃豆子来使其最大化，而幽灵试图通过抢先吃掉玩家来使玩家得分最小化。

**General Games**：一个抢绿宝石，一个抢橙色宝石 Cooperation indifference competition and more

### Adversarial Search

#### MiniMax

我们第一个讲到的零和游戏算法是minimax，这一算法基于一种假设，即对手也会进行最优操作并总是会采取最损人利己的策略。首先我们要介绍**终端效益（terminal utilities）**和**状态值（state value）**的概念。

一个状态的值是控制该状态的agent所能得到的最高分数。

![img](https://pic4.zhimg.com/80/v2-d063f3f4366b4f4b96d3ab46c920bbb3_1440w.webp)

假设吃豆人初始分数为10，在吃到豆子之前每一步会扣一分，吃到豆子时游戏会达到**终端状态（terminal state）**并结束。对此，我们能构建一棵如下图所示的游戏树，其中一种状态的子节点是后继状态，与普通搜索问题中的搜索树一样

![img](https://pic3.zhimg.com/80/v2-b9124c069b7d95b125c3d235afeb0b9a_1440w.webp)

通过这棵树可以明显看出，如果吃豆人直接冲向豆子，游戏结束时分数为8，而如果在任何一步反向移动，最终得分都会低于8。现在我们构建了一棵由几个中断状态和中间状态构成的游戏树，接下来就可以得到其中任何一个状态的值的意义。

一个状态的值定义为一个agent从该状态出发能得到的最佳输出（效益）。一个终端状态的值，称为**终端效益（terminal utility）**，是一个能确定的值和一个固定的游戏性质。在吃豆人的例子中，最右端的终端状态为8，这是吃豆人直奔豆子的得分。同样，在这个例子中，一个非终端状态的值定义为其子节点的最大值。定义V(s)为定义一个状态s的值的函数，以上讨论可以总结为：

![img](https://pic2.zhimg.com/80/v2-fdf154dcd2bbbed1c9ec186a3fe04bd1_1440w.webp)

现在介绍一个新的游戏情景，有一个敌方幽灵想阻止吃豆人吃到豆子。

![img](https://pic4.zhimg.com/80/v2-9e91f985a3ca50f40fe9965f88c2f417_1440w.webp)

游戏规则要求双方轮流进行移动，导致最终得到的游戏树中双方agent关闭了他们“控制”的树的层。一个agent控制一个节点的意思是该节点对应着轮到该agent移动的状态，所以这是他来决定行动并改变游戏形势的机会。如下是这一新的双agent游戏对应的游戏树：

![img](https://pic1.zhimg.com/80/v2-84a3ab305865750436cfbc99b49f47a4_1440w.webp)

蓝色节点由吃豆人控制，红色节点由幽灵控制。注意到，幽灵控制节点的所有子节点都是由幽灵向左或向右移动形成的，吃豆人控制的节点也同理。为了简单起见，我们将这个游戏简化为深度为2的树，并给终端状态赋上虚值：

![img](https://pic2.zhimg.com/80/v2-739a9be3d24ce12bb89a578bb8b5ddd1_1440w.webp)

自然而然的，加入幽灵控制的节点会改变吃豆人原本认为最优的行动，而新的最优行动由minimax算法确定。Minimax算法仅仅最大化吃豆人控制节点的子节点并同时将幽灵控制节点的子节点最小化，而不是将树的每一层的子节点的效益最大化。因此，上图中的两个幽灵节点取值分别为 min(-8,-5)=-8 以及 min(-10,+8)=-10。相应的，吃豆人控制的根节点值为  max(-8,-10)=-8。由于吃豆人想让分数最大化，他会向左走来取得-8分，而不是妄图吃豆子而最终只能得-10分。这是通过计算提高行为的一个基础例子——虽然吃豆人想要到达最右子节点状态中的+8分，但是通过minimax他“知道”一个会采取最优行动的幽灵不会让他成功。为了采取最优行动，吃豆人必须止损，并且反直觉地远离豆子来让他的损失最小化。Minimax对状态赋值的方法可以总结如下：

![img](https://pic3.zhimg.com/80/v2-c052463fddad083626d526a6d1a13906_1440w.webp)

在实现时，minimax的行为与**深度优先搜索**有些类似，与DFS采取相同的顺序来计算节点的值，从最左叶子节点开始并向右迭代。更准确的来说，它在游戏树中使用了**后序遍历（postorder traversal）**。Minimax最终的伪代码既优雅，同时又简单直观：

![img](https://pic2.zhimg.com/80/v2-716f7292ce1de125435cf267488013e1_1440w.webp)

#### α-β剪枝 Alpha-Beta Pruning

Minimax看起来几乎完美——简单，最优，直观。然而，它的执行与DFS相似，而且时间复杂度也同样悲催，达到了 O(b^m)。为了改进这一问题，对minimax进行优化后就得到了α-β剪枝。

  概念上来说，α-β剪枝的意思就是：如果你想通过查看一个节点n的后继来决定它的值，当你知道n的值最多也只能达到其父节点的最大值时，就别管它了。我们通过一个例子来解释一下这个小技巧的含义。考虑如下一棵游戏树，方块节点对应终端状态，倒三角表示取子节点最小值，正三角表示取子节点最大值：

![img](https://pic2.zhimg.com/80/v2-4ffb879be1d737b4466bf9d58507cd35_1440w.webp)

先来看看minimax会如何操作：最开始在值为3,12,8的子节点中迭代，并给最左侧的最小值节点赋值为 min(3,12,8)=3，然后给中间的最小值节点赋值为 min(2,4,6)=2，给最右的最小值节点赋值  min(14,5,2)=2。最终给根节点的最大值节点赋值为  max(3,2,2)=3。然而，在这个例子中我们会意识到，**当访问到中间倒三角节点的一个子节点值为2时，无论后面的子节点如何取值，这个最小值节点最大取值也只能是2，因此我们不再需要查看其他子节点了。**从而对这棵搜索树进行剪枝：

![img](https://pic3.zhimg.com/80/v2-301e0f3c86a008984aaeeed4312280c2_1440w.webp)

进行这样的剪枝可以让时间复杂度降为 O(b^m/2)，从而能让我们可以解决的深度直接翻倍。实际上，通常没有这么多，但是也可以让搜索范围减少至少1-2层。这仍然非常有意义，因为能够预测三步的一方比预测两步的一方有更大的胜算。经过α-β剪枝的minimax的做法正是如此，其操作如下：

![img](https://pic2.zhimg.com/80/v2-0e731dd5790a2dc8aadfb3b9cba7c9d1_1440w.webp)

花点时间对比一下这一方法和前面的普通minimax，并且要注意到，我们现在不用遍历每一个子节点，可以更早地返回结果。

剪枝效果和顺序有关 

#### Evaluation Function *(估计函数 )*

虽然α-β剪枝能有效增加minimax的搜索深度，但是对大多数游戏来说这离到达搜索树的底部仍然还差得远。于是，我们转向**估计函数（evaluation functions）**，这种函数输入状态并输出该节点的minimax估计值。简单直接的解释为：一个好的估计函数能给更好的状态赋更高的值。估计函数在**深度限制（depth-limited）minimax**中广泛应用，即将最大可解深度处的非叶子结点都视作叶子节点，然后用仔细挑选的估计函数给其赋虚值。由于估计函数只能用于估计非叶子结点的效益，这使得minimax的最优性不再得到保证。

  在设计一个用于minimax的agent时，大量的设想和实验被用于选择估计函数，更好的估计函数能让agent的行动更接近最优策略。此外，在使用估计函数之前访问的层数更深，也能够带来更好的结果——将计算埋在游戏树的深层能缓和最优性的让步。这种函数在游戏中的做法与启发式在标准搜索问题中的做法非常相似。

 最常用的估计函数被设计为特征的线性组合：

![img](https://pic1.zhimg.com/80/v2-34f2fc2d8c6f646c5236c81258b12cb4_1440w.webp)

每个 f 对应从状态s中提取的一个特征，每个特征被赋予了一个相应的权重 w。特征就是游戏状态中能够提取并量化的一些元素。估计函数的设计可以比较不拘一格，也没必要非得是线性函数。最重要的是，估计函数会尽可能频繁地为更好的位置给出更高的分数。这需要对各种不同特征和权重的agent进行大量的微调和实验。

### Uncertainty and Utilities *（不确定性和效益）*

#### Expectimax  Search

**minimax**相信它面对的是一个最优的对手，在对手不一定会做出最优行动的情况下，它就显得有些杞人忧天了。这种情况包括一些本来具有随机性的情况，比如打牌或骰子，又或者是会进行随机/次优移动的对手。

这种随机性可以用**minimax**的一种泛化来表示，即**expectimax**。**Expectimax**在游戏树中加入了**机会节点（chance nodes）**，与考虑最坏情况的最小化节点不同，机会节点会考虑**平均情况（average case）**。更准确的说，最小化节点仅仅计算子节点的最小效益，而机会节点计算**期望效益（expected utility）**或期望值。Expectimax给节点赋值的规则如下：

![img](https://pic4.zhimg.com/80/v2-c18aa1db14af7ed5d0c1653bbfceab57_1440w.webp)

minimax就是expectimax的一种特例，最小化节点就是认为值最小的子节点概率为1而其他子节点概率为0的机会节点。

 Expectimax的伪代码与minimax很相似，就是把最小效益换成了期望效益，这是因为最小化节点被替换成了机会节点：

![img](https://pic2.zhimg.com/80/v2-8ed56a77dc6dca8dfc1798d7ce18fc25_1440w.webp)

我们先来快速过一下一个简单的例子。看看下面这个expectimax树，其中圆形表示机会节点，代替了最大化/最小化节点的正三角和倒三角。

![img](https://pic1.zhimg.com/80/v2-3ed20eca4f10587870a0eb7c08d424ec_1440w.webp)

为了简单起见，假设机会节点的每个子节点概率都是1/3。于是，根据expectimax的赋值规则，三个机会节点从左到右的取值分别为1/3*3+1/3*12+1/3*9=8，1/3*2+1/3*4+1/3*6=4以及1/3*15+1/3*6+1/3*0=7。最大化节点（根节点）会从中选择最大值8，从而形成如下的完整游戏树：

![img](https://pic1.zhimg.com/80/v2-18754d79048b5d5ee883408290fde3e0_1440w.webp)

每个值都会影响expectimax计算的期望值。不过，当我们已知节点有限的取值范围时，剪枝*(pruning)*也是有可能的。

**minimax**相信它面对的是一个最优的对手，在对手不一定会做出最优行动的情况下，它就显得有些杞人忧天了。**expectimax**考虑平均情况

![图片](.\image\comparison of expectimax and minimax.jpg)

#### Mixed Layer Types 

虽然minimax和expectimax分别使用了最大值/最小值节点和最大值/机会节点，仍然有许多游戏不适用于这两种算法中的模式。即使是在吃豆人中，在吃豆人移动后，通常会有多个而不是一个幽灵进行轮流移动。这可以理解为我们必须在游戏树中不定地加入新的层。在一局有四个幽灵的吃豆人游戏中，就需要在两个吃豆人（最大值）层之间连续插入四个幽灵（最小值）层。其实，这么做肯定会加强所有最小值节点之间的合作，因为它们会轮流地进一步让最大值节点能得到的分数最小化。如果有一局有两个幽灵的吃豆人游戏，其中一个幽灵采取最优行动而另一个采取随机行动，我们可以用交替的最大值-机会-最小值节点层来表示。

![img](https://pic1.zhimg.com/80/v2-3abf66069f574288c461a25589da1d44_1440w.webp)

显然，有相当大的空间来调整节点分层，这使得对于任何零和游戏，由expectimax和minimax混合构成的游戏树和对抗搜索算法能有进一步发展。

#### Multi-Agent Utilities

并非所有游戏都是零和的。其实，在一些并不含有直接对抗的游戏中，不同的agent可能有各自独立的任务。这样的游戏可以用具有**多agent效益（multi-agent utilities）**的树来表示。每个agent会倾向于在各自控制的节点处让自己的效益最大化，而不考虑其他玩家的得分。看看下面这棵树：

![img](.\image\multi-agent tree.jpg)

红色、绿色和蓝色的节点对应三个agent，分别在他们各自的层中取相应颜色的最大值。这棵树最终根节点的结果效益为三元组（5,2,5）。具有多agent效益的普通游戏是通过计算提升行为的基础例子，因为在根节点处的效益会倾向于让所有参与的agent共同合作来得到合理的效益。

#### Utilities 

 理智的agent必须服从**效益最大化原则（principle of maximum utility）**——他们必须总是选择能将效益最大化的操作。然而，服从这一原则只会让有**理智倾向（rational preferences）**的agent收益。要构建一个非理智倾向的例子，我们假设存在三个对象A,B和C，我们的agent现在拥有A。假设我们的agent有以下的非理智倾向：

![image](.\image\irrational preference.jpg)

agent破产

##### rational preferences

![image](.\image\Axioms of Rationality.jpg)

- **有序性（orderability）**

- **传递性（transitivity）**

- **连续性（continuity）**

- **置换性（substitutability）**

  如果一个理性的agent认为A和B一样，那么对于任何两个A和B互相替换的彩票，该agent也认为没区别

- **单调性（monotonicity）**

如果一个agent满足全部以上5条公理，那么可以保证该agent的行为会让期望效益最大化。更具体地说，这说明存在一个实数**效益函数（utility function）**U ，会给偏好的奖励赋予更高的效益值，同时，一个彩票的效益就是该彩票的奖励效益的期望值。这两条可以总结为两个简单的数学方程：

![img](https://pic1.zhimg.com/80/v2-22ac391989969f9c87165c93bba9e12c_1440w.webp)

如果能够满足这些约束并且选择合适的算法，使用这样的效益函数的agent就能保证采取最优行动。我们来通过一个具体例子进一步导论一下效益函数。有一支彩票如下：

L = [0.5, $0;0.5, $1000]

这表示在这支彩票中，获得0$和1000$的概率五五开。现在加入三个agent, A1, A2, A3。如果三个agent都可以选择买彩票或是直接收到500$，他们会选择哪一个呢？他们做出两种选择分别得到的效益的期望如下：

![image](.\image\utility.jpg)

![img](https://pic2.zhimg.com/80/v2-8a463e21fce1af79c46e0249078ad1a5_1440w.webp)

这些彩票的效益值计算过程如下，使用了上面提到的方程：

![img](https://pic1.zhimg.com/80/v2-1d788e78f56c9cb31aa48acfee551844_1440w.webp)

风险中立（risk-neutral）规避风险（risk-averse）寻求风险（risk-seeking）。

效益不等于期望，更多人选择直接拿走400$，保险公司。

risk-seeking preference

Functions that are either decreasing slower than linearly, like -sqrt{r}, or increasing faster than linearly, like r^2, satisfy this.

## MDP (Markov Decision Processes)

###  Non-Deterministic Search 不确定搜索

现在，我们将再次改变模型来考虑另一个影响因素——世界本身的动态变化。Agent所在的环境会迫使agent的行为变得**不确定(nondeterministic)**，意味着**在某一状态下采取的一个行动会有多个可能的后继状态**。实际上，这就是许多像扑克牌或是黑杰克这样的卡牌游戏中会出现的情况，存在着由发牌的随机性导致的固有的不确定性。这种在世界中存在一定程度的不确定性的问题被称为**不确定搜索问题(nondeterministic search problems)**，可以用**马尔科夫决策过程(Markov decision processes)**或者称为MDPs来解决。

### Markov Decision Processes

马尔可夫决策过程包含了几个属性：

![image](.\image\MDP01.jpg)

![image](.\image\MDP02.jpg)

给一种情况构造MDP类似于给一个搜索问题构造状态空间图，另外再补充一些说明。例如一辆赛车的激励机制：

![img](https://pic3.zhimg.com/80/v2-dcfb18d217ad694bc3ef6297a3dc0452_1440w.webp)

有三种可能状态S={cool，warm，overheated}，以及两种可能操作A={slow，fast}。就像在状态空间图中一样，三个状态都用节点表示，行动则用边来表示。Overheated是一个最终状态，因为一旦赛车达到这一状态，它就无法再采取任何行动来获得更多奖励了（这在MDP中是一种**下沉状态(sink  state)**，没有可以转移到其他状态的边）。尤其地，对不确定行动来说，有多条边表示着从同一个状态采取的相同行动，而这些行动指向不同的后继状态。每条边不仅注释了它代表的行动，同时还有转移概率及相应的奖励。以上这些总结如下：

![](.\image\MDP03.jpg)

我们的赛车问题简化为深度为2的搜索树之后表示如下：

![img](https://pic3.zhimg.com/80/v2-395efad1647ea4903bc8ea9495501e82_1440w.webp)

绿色节点表示q状态，意味着已经从一个状态采取了某一行动，但还没有得到一个后继状态。要注意理解的是，在q状态时agent不会消耗时间步，它们只是一种概念，用来更好地对MDP算法进行表示和发展。

#### Discounting

![](.\image\MDP04.jpg)

![](.\image\MDP05.jpg)

**折扣因子r的引入 使得U收敛**

### Solving MDPs

![](.\image\MDP06.jpg)

##### **The Bellman Equation （贝尔曼方程）**

**自上向下**

##### ![](.\image\MDP07.jpg)

##### Value Iteration

**自下向上**

![](.\image\MDP08.jpg)

![](.\image\MDP09.jpg)

r=1 计算 V2(cool)

a=slow 1+2=3

a=fast   0.5(2+2)+0.5(2+1)=3.5

### Policy Methods

#### Policy Iteration

![](.\image\MDP10.jpg)

![](.\image\MDP11.jpg)

![](.\image\MDP12.jpg)

![](.\image\MDP13.jpg)

### Summary

-  值迭代Value iteration：迭代更新直至收敛，用于计算状态的最优值。
-  策略评估 Policy evaluation：用于在一个特定的策略下计算状态的值。
-  策略提取 Policy  extraction：用于在给定一些状态值函数的情况下决定一个策略。如果状态值是最优的，这个策略也就是最优的。这一方法在进行值迭代之后使用，用于从最优状态值计算得到最优状态；或是在值迭代中作为一个子程序，用于计算当前估计状态值的最佳策略。
-  策略迭代 Value iteration：一种将策略评估和策略提取集于一身的技术，用于迭代收敛从而得到一个最优策略。其表现优于值迭代，这是由于策略往往比状态值收敛得快得多。

## Reinforcement Learning 强化学习

![](.\image\RL1.jpg)

在前面的章节中，我们讨论了马尔可夫决策过程，利用值迭代和策略迭代等技术来计算状态的最优值并提取最优策略。解决**马尔可夫决策过程**是**离线计划（offline planning）**的一个例子，其中代理人对转移函数和报酬函数都有**充分的了解**，所有他们需要的信息都用来预先计算由 MDP  编码的世界中的最优行动，而不需要实际采取任何行动。在本章中，我们将讨论在线计划(**online planning**)，在这个过程中，代理没有关于奖励或过渡的事先知识(仍然表示为  MDP)。在在线计划中，代理人必须尝试探索，在这个过程中，代理人采取行动，接收反馈，反馈的形式是**它到达的后继状态和它获得的相应奖励**。代理使用这个反馈通过一个叫做**强化学习**的过程来估计一个最优策略，然后使用这个估计的策略进行开发，或者报酬最大化。

### Model-Based Learing

![](.\image\RL2.jpg)

![](.\image\RL3.jpg)

![](.\image\RL4.jpg)

根据大数定律，当我们通过让agent经历更多的事件来收集越来越多的样本时，我们的 T 和 R 模型将得到改进，随着 估计的T,R 趋向于 真正的T 和  R。只要我们认为合适，我们可以结束我们的agent的训练，通过运行Value Iteration 或 Policy Iteration 来生成策略 π 利用，使用我们目前的 T 和 R 模型，并使用 π 利用进行开发，让我们的agent遍历 MDP  采取行动寻求奖励最大化而不是寻求学习。基于模型的学习非常简单和直观，但是非常有效，只需要计数和归一化就可以生成 T 和  R。然而，维护每个元组的计数可能是昂贵的。

### Model-Free Learning

我们将讨论无模型学习算法其中的三种: **direct evaluation** (直接评估)、**temporal difference learning** (时间差学习)和 **Q-learning** (Q 学习)。直接评估和时间差学习属于一类被称为**passive reinforcement learning**（被动强化学习）的算法。在被动强化学习中，一个主体被给予一个policy 来遵循并学习该policy下的state value，Q 学习属于第二类无模型学习算法，称为**active reinforcement learning** （主动强化学习），在此期间学习agent可以使用它接收到的反馈在学习过程中**迭代更新其策略**，直到在充分探索之后最终确定最优策略。

#### Direct Evaluation

![](.\image\RL5.jpg)

![](.\image\RL6.jpg)

 γ = 1 通过Episode1，从状态 D 到终止，我们获得了总奖励10，从状态 C，我们获得了总奖励(- 1) + 10 = 9，从状态 B，我们获得了总奖励(- 1) + (- 1) + 10 = 8。

![](.\image\RL7.jpg)

**简便但是忽略了状态之间的关系，而且耗时**

#### Temporal Difference Learning

时间差学习(TD Learning)使用从每个经验中学习的思想，而不是简单地跟踪总奖励和访问状态的次数。在policy evaluation中，我们使用由fixed policy和贝尔曼方程生成的方程组来确定该政策下的状态值(或者使用迭代更新，如值迭代)。

![](.\image\RL8.jpg)

![](.\image\RL9.jpg)

上面，α 是一个受0≤ α ≤1约束的参数，称为学习速率，它指定了我们要为 Vπ (s) ，1-α  分配现有模型的权重，以及我们要分配新的抽样估计值 α 的权重。典型的做法是从学习速率 α = 1开始，相应地将 Vπ  (s)分配给任何第一个样本，然后慢慢地将其缩小到0，这时所有后续样本将被归零，并停止影响我们的 Vπ  (s)模型。让我们停下来分析一下更新规则。通过定义 V πk (s)和 samplek 分别作为 kth 更新和 kth 样本后状态 s  的估计值来注释模型在不同时间点的状态，我们可以重新表示我们的更新规则。

由于0≤(1-α)≤1，当(1-α)的数量增加到越来越大的幂时，它变得越来越接近0。通过我们导出的更新规则展开，这意味着给予较老的样本以指数级较小的权重，这正是我们想要的，因为这些较老的样本是使用我们的 V π (s)模型的较老(因此更糟)版本计算的！

![](.\image\RL10.jpg)

#### Q-Learning

![](.\image\RL11.jpg)

Q 学习提出直接学习状态的 q 值，而不需要知道任何值、转换函数或奖励函数。因此，Q 学习是完全没有模型的。Q 学习使用以下更新规则来执行所谓的 q 值迭代:

![](.\image\RL12.jpg)

只要我们花足够的时间进行探索，并以适当的速度降低学习速率 α，Q 学习就能学习到每个 q 态的最优 q 值。这就是为什么 Q  学习如此具有革命性的原因——当 TD 学习和直接评估通过遵循policy来学习政策下的状态价值，然后再通过其他技术来确定policy的最优性时，Q 学习甚至可以通过采取次优或随机行动来直接学习最优政策。这被称为非策略学习(与直接评估和 TD 学习相反，后者是非策略学习的例子)。

#### Exploration and Exploitation

![](.\image\RL13.jpg)

是否停止学习用当前最优，还是去探索新的路径

##### ε-Greedy Policies

![](.\image\RL14.jpg)

ε- 贪婪策略用概率 ε  进行探索。这是一个很容易实现的策略，但是仍然很难处理。如果选择了一个大的 ε  值，那么即使在学习了最优策略之后，Agent 仍然会表现出大部分随机行为。类似地，选择一个小的 ε 值意味着代理将很少探索，导致 Q  学习(或任何其他选择的学习算法)非常缓慢地学习最优策略。为了解决这个问题，ε 必须手动调整，并随着时间的推移降低，以查看结果。

##### Exploration Functions

![](.\image\RL15.jpg)

n较小 探索（乐观）

n较大 没必要重复探索 重复学习

#### Approximate Q-Learning

Q 学习只是存储表格形式的状态的所有 q 值，这并不特别有效，因为大多数强化学习的应用都有几千甚至几百万个状态。这意味着我们不能在训练期间访问所有的状态，也不能存储所有的 q 值。

概括学习经验的关键是基于特征的状态表示，它将每个状态表示为一个向量，称为特征向量。例如，Pacman 的特征向量可以编码

![](.\image\RL16.jpg)

![](.\image\RL17.jpg)

![](.\image\RL18.jpg)

Q改成了特征值的加权线性函数

每进行一次模拟 算出difference = 目前的效益加上以后的最大效益 - 计算出的Q

根据difference更新当前状态的Q值 和 权重

![](.\image\RL19.jpg)

到点的距离 和 到幽灵的距离 作为特征

## Probability

![](.\image\P.jpg)

基础概率论知识 跳过

### Conditional Independence

![](.\image\P2.jpg)

Cavity 有无蛀牙		Catch 探针测出有蛀牙		Toothache和Catch 在 Cavity 条件下相互独立

![](.\image\P3.jpg)

![](.\image\P4.jpg)

链式法则可以化简

![](.\image\P5.jpg)

T B 在G条件下独立 所以根据现有的信息可以得出右边完整的表格

## Bayes' Nets

### Representation

![](.\image\BN1.jpg)

![](.\image\BN2.jpg)

**根据图可以很简单的写出简化后的链式规则，从而非常方便求联合概率** 

![](.\image\BN3.jpg)

![](.\image\BN4.jpg)

![](.\image\BN5.jpg)

**根据因果关系画图（建立模型）的好处** 

### Conditional Independece

#### D-seperation(directed separation)

一个节点在给定其所有父节点的情况下，有条件地独立于图中的所有祖先节点。复杂的图可以分成三元组，我们将先介绍所有三个连通的三节点两边贝叶斯网或三元组的典型情况，以及它们所表达的条件独立关系

##### Causal Chains

![](.\image\BN6.jpg)

**X和Z不独立		X和Z在Y条件下独立**

##### Common Cause

![](.\image\BN7.jpg)

**X和Z不独立		X和Z在Y条件下独立**

##### Common Effect

![](.\image\BN8.jpg)

**X和Y独立		X和Y在Z条件下不独立**

##### General Case

D- 分离(定向分离)是贝叶斯网络图结构的一个属性，它暗示了这种条件独立关系，并且概括了我们上面看到的情况。如果一组变量 Z1，... Zk d- 分离 X 和 Y，那么在所有可能的分布中，可以用贝叶斯网编码的 X 独立Y 在 { Z1，... Zk }条件下。

1. 对图中所有观察到的节点{ Z1，... Zk }进行阴影处理（作为evidence)

2. 枚举从 X 到 Y 的所有无向路径。

3. 对于每个路径: 

   (a)将路径分解为三元组(3个节点的段)。

   (b)如果所有三元组都是active的，则此路径是active的，并且 d-connects X 到 Y。

4. 如果没有路径 d 连接 X 和 Y，那么 X 和 Y 是 d 分离的，所以给定{ Z1，... ，Zk }它们是条件独立的

图中从 X 到 Y  的任何路径都可以分解成一个由3个连续节点和2个边组成的集合——每个节点都被称为三元组。三元组是active的还是inactive取决于是否观察（observe/shadow)到中间节点。如果**路径中的所有三元组都是active**，那么**路径是active**，并且 d-连接 X 到 Y，这意味着给定观察到的节点，X 不能保证在条件上独立于 Y。如果从 X 到 Y  的**所有路径都是inactive**的，那么给定所观察到的节点，X 和 Y 是条件独立的。

![](.\image\BN9.jpg)

###### Example

![](.\image\BN10.jpg)

**一条路径有一个三元组inactive整个路径就是inactive**

![](.\image\BN11.jpg)

**有一个路径是active 那么就是No**

![](.\image\BN12.jpg)

### Probabilistic Inference

#### Enumeration

**把所有相关的子表合成一个大表 再选择有用的计算 指数级**

![](.\image\BN14.jpg)

#### Variable elimination

**边结合便消除无关(hidden)变量**

![](.\image\BN13.jpg)

##### Example

![](.\image\BN15.jpg)

**enumeration**

![](.\image\BN16.jpg)

**variable elimination**

![](.\image\BN17.jpg)

**Math** **presentation**

![](.\image\BN18.jpg)

**evidence的处理**

把无关evidence删去，简化表格

![](.\image\BN19.jpg)

P(+r, L)->P(L|+r)

![](.\image\BN20.jpg)

**求P(B|j,m)**

![](.\image\BN21.jpg)

![](.\image\BN22.jpg)

##### join order

![](.\image\BN23.jpg)

1. 先消除X1,X2...再消除Z合并（每一步计算都很简单）

2. 先消除Z (复杂度高，P(X1,X2...Xn-1) 指数级)再消除X1,X2...

**最优合并方式是NP-hard问题** 但是Polytrees（没有无向环的有向图）可以

![](.\image\BN24.jpg)

### Sampling (Approximate Inference)

**根据已有信息进行大量模拟，利用sample计算出要求的P**

![](.\image\BN25.jpg)

#### Prior Sampling

**模拟C S R W**

![](.\image\BN26.jpg)

![](.\image\BN27.jpg)

#### Rejection Sampling

![](.\image\BN28.jpg)

**发现不符合证据立刻重新开始采样** 比如求P(w|+c) 若采样时出现 -c 丢弃该样本，对下一样本进行采样

#### Likelihood Weighting （可能性加权）

![](.\image\BN30.jpg)

![](.\image\BN29.jpg)

**如果遇到evidence的变量 设置为观测值 这会改变sample的分布和实际情况不同（每个sample不是等可能的） 所以需要同时计算出每个sample的权重**

![](.\image\BN31.jpg)

#### Gibbs Sampling

![](.\image\BN32.jpg)

如果有观测值在很后面，前面有一大堆祖先，那么仍需取大量数据，因为可能取的都是w较小的数据。比如在下雨的条件下，那么前面仍取-低气压意义不大，因为sample的w很小

吉布斯抽样是第四种抽样方法。在这种方法中，我们首先将所有变量设置为某个完全随机的值(不考虑任何 CPT)。然后，我们一次重复选择一个变量，清除它的值，并根据当前分配给所有其他变量的值对它进行重采样。

**抽样过程**: 跟踪完整的实例化 x1，x2，... ，xn。从与证据一致的任意实例开始。一次采样一个变量，以所有其他变量为条件，但保持证据固定。长时间重复这个过程。

**属性**: 在无限多次重复的限制下，得到的样品来自正确的分布。

**理由**: 上游或下游变量都以证据为条件。

**相比之下**: 似然加权仅在上游证据的条件下，因此在似然加权中获得的权重有时可能非常小。所有样品的权重总和表明获得了多少“有效”样品，所以我们要高权重的样品。

![](.\image\BN33.jpg)

**计算更高效**

![](.\image\BN34.jpg)

## Hidden Markov Models

### Markov Models

**可以被认为类似于链状的，无限长的贝叶斯网。**

![](.\image\BN35.jpg)

我们将在本节中使用的运行示例是天气模式的日常波动。我们的天气模型将是依赖于时间的(就像马尔可夫模型一般) ，这意味着我们将有一个单独的随机变量的天气在每一天。

![](.\image\BN36.jpg)

![](.\image\BN37.jpg)

#### Stationary distribution

![](.\image\BN38.jpg)

### Hidden Markov Models

在马尔可夫模型中，我们看到了如何通过一系列随机变量来整合随时间变化的变化。例如，如果我们想用上面的标准马尔可夫模型知道第10天的天气，我们可以从初始分布 Pr (W0)开始，并使用带有转换模型的迷你前向算法来计算 Pr (W10)。然而，在 t = 0和 t =  10之间，我们可能会收集到新的气象学证据，这些证据可能会影响我们对任何给定时间步骤的天气概率分布的信念。简单地说，如果天气预报说第10天有80% 的可能会下雨，但是第9天晚上天气晴朗，那么80%  的可能性就会急剧下降。这正是隐马尔可夫模型帮助我们做到的——它**允许我们在每个时间步骤观察一些证据**，这可能会影响每个state的分布。

![](.\image\BN40.jpg)

我们的天气模型的隐马尔可夫模型可以用贝叶斯网络结构来描述

![](.\image\BN41.jpg)

#### Conditional Independence

![](.\image\BN42.jpg)

#### example

![](.\image\BN43.jpg)

#### The Forward Algorithm

![](.\image\BN44.jpg)

![](.\image\BN45.jpg)

![](.\image\BN46.jpg)

##### example

![](.\image\BN48.jpg)

![](.\image\BN49.jpg)

![](.\image\BN50.jpg)

#### Viterbi Algorithm

该算法由两个步骤组成:

 第一个步骤在时间上向前运行，并根据目前观察到的证据计算到达该(状态，时间)元组的**最佳路径**的概率。

第二次通过向后运行: 首先它找到位于最高概率路径上的终端状态，然后沿着通向这个状态的路径向后遍历时间(这必须是**最佳路径**)

##### state trellis 状态格

![](.\image\HMM7.jpg)

在这个隐马尔可夫模型中有两种可能的隐状态，太阳或雨，我们希望计算从 X1到 XN 的最高概率路径。边权等于 P (Xt | Xt-1) P  (Et | Xt) ，路径概率由边权的乘积计算。权重中的第一项估计特定转换的可能性，第二项权重观察到的证据与结果状态的匹配程度。

![](.\image\HMM8.jpg)

##### example

![](.\image\HMM9.jpg)

**语音识别 找到最相似的单词**

### Particle Filtering 

![](.\image\HMM2.jpg)

我们不存储一个完整的概率表将每个状态映射到它的信念概率，而是存储一个 **n 个粒子的列表**，其中每个粒子都处于依赖于时间的随机变量域中的一个多可能的状态。

![](.\image\HMM3.jpg)

**时间流逝更新**：根据**转换模型**更新每个粒子的值。

**观测更新**：在粒子过滤的观测更新过程中，我们根据观测证据和粒子状态所决定的概率对每个粒子进行**加权**。nomalize权重在各状态的分布，并从这个分布**重新采样粒子列表**。

##### example

[课程网](https://inst.eecs.berkeley.edu/~cs188/su20/)note6最后一部分

### Dynamic Bayes Nets (DBNs)

![](.\image\HMM4.jpg)

![](.\image\HMM5.jpg)

#### DBN Particle Filters 

![](.\image\HMM6.jpg)

## Decision Networks and Value of Information

### Decision Networks

现在我们将讨论 `Bayes’ nets` 和 `expectimax`的组合——`Decision Networks`，我们可以使用这个决策网络来建模各种行动对效用的影响，基于一个总体的图形概率模型。让我们深入剖析一下决策网络:

![](.\image\D1.jpg)

![](.\image\D2.jpg)

机会节点-决策网络中的机会节点与贝叶斯网络的行为相同。机会节点中的每个结果都有一个相关的概率，这可以通过对它所属的底层贝叶斯网络进行推断来确定。我们用椭圆来表示这些。

动作节点-动作节点是我们可以完全控制的节点;  它们代表了我们可以从中选择的任何一个动作。我们将用矩形表示操作节点。

效用节点-效用节点是某些行动节点和机会节点组合的子节点。它们根据父母的价值观输出一个效用，并在我们的决策网络中以菱形的形式表示。

![](.\image\D3.jpg)

![](.\image\D4.jpg)

![](.\image\D6.jpg)

#### Outcome Trees

决策网络包含一些`expectimax`类型的元素，所以让我们讨论一下这到底意味着什么。我们可以将决策网络中与期望效用最大化相对应的行为的选择拆解为一个结果树。我们上面的天气预报例子分解为以下结果树:

顶部的根节点是一个最大化器节点，就像在 `epectimax`  中一样，由我们控制。我们选择一个操作，它将我们带到树中的下一个级别，由机会节点控制。在这个级别上，机会节点在最终级别上解析为不同的效用节点，其概率相当于基础贝叶斯网络上运行的概率推断得到的后验概率。对于结果树，我们在任何给定的时刻(在花括号内)使用我们所知道的内容来注释节点。

![](.\image\D5.jpg)

### Value of Information

在我们到目前为止所做的所有工作中，我们通常都假设我们的代理已经掌握了特定问题所需的所有信息，并且/或者没有办法获取新的信息。在实践中，情况并非如此，决策过程中最重要的部分之一就是知道是否值得收集更多的证据来帮助决定采取何种行动。观察新的证据几乎总是有一些成本，无论是在时间，金钱，或其他媒介方面。在本节中，我们将讨论一个非常重要的概念——**the value of perfect information (VPI)** -——它从数学上量化了如果观察到一些新的证据，代理人的最大期望效用预期会增加的数量。我们可以比较学习一些新信息的 VPI 和观察该信息的成本，以决定是否值得观察。

![](.\image\V1.jpg)

**MEU: maximum expected utility**

#### VPI Properties

![](.\image\V2.jpg)

#### example

![](.\image\V4.jpg)

![](.\image\V3.jpg)

### Partially observable MDP（POMDP）

**每次增加一个observation作为evidence ** **通过比较VPI和cost of observation来决定是否采取action**

![](.\image\V5.jpg)

![](.\image\V6.jpg)

## Machine Learning

在本课程前面的几个笔记中，我们已经了解了各种类型的模型，这些模型可以帮助我们在不确定情况下进行推理。到目前为止，我们一直认为我们所使用的概率模型是**理所当然**的，我们所使用的基础概率表的生成方法已经被抽象出来了。当我们深入讨论机器学习时，我们将开始打破这个抽象的障碍。机器学习是计算机科学的一个广泛领域，涉及构造和/或学习给定数据的特定模型的参数。

有许多机器学习算法处理许多不同类型的问题和不同类型的数据，根据他们希望完成的任务和他们处理的数据类型进行分类。机器学习算法的两个主要子类是**监督式学习算法**和**非监督式学习算法**。监督式学习算法推断输入数据和相应输出数据之间的关系，以便预测新的、以前看不见的输入数据的输出。另一方面，无监督学习算法的输入数据没有任何相应的输出数据，因此需要识别数据点之间或数据点内部的固有结构，并相应地进行分组和/或处理。在本课中，我们将讨论的算法仅限于监督式学习任务。

一旦你有了一个可以学习的数据集，机器学习过程通常包括将你的数据集分成三个不同的子集。第一个是**训练数据**，用于实际生成一个模型，将**输入映射到输出**。然后，**验证数据**(也称为坚持或开发数据)用于通过对输入进行预测并生成准确性得分来衡量模型的性能。如果你的模型没有你想要的那么好，那么你可以回去再训练一次，或者通过调整特定模型的超参数值，或者使用不同的学习算法，直到你对你的结果满意为止。最后，使用模型对数据的第三个也是最后一个子集——**测试集**进行预测。测试集是代理直到开发的最后才看到的数据部分，它相当于衡量现实数据性能的“期末考试”。

![](.\image\ML1.jpg)

### Naive Bayes

**分类问题**——给定各种数据点(在本例中，每封电子邮件都是一个数据点)  ，我们的目标是将它们分组为两个或两个以上的类中的一个。对于分类问题，我们给出了一组数据点及其相应标签的训练集，这些标签通常是少数几个离散值之一。描述如何构造一种特定类型的模型来解决分类问题，这种模型被称为**朴素贝叶斯分类器**。

#### Digits

**判断手写的是0-9中的哪个数字 根据像素点来判断**

![](.\image\ML2.jpg)

![](.\image\ML3.jpg)

根据已经学习并计算出的条件概率来判断筛选出最可能的数字

#### Spam Filtering

让我们考虑一下**构建垃圾邮件过滤器**的常见问题，该过滤器将邮件分类为垃圾邮件(不需要的邮件)或垃圾邮件(需要的邮件)。

![](.\image\ML5.jpg)

最后两列是取对数的结果

![](.\image\ML4.jpg)

#### Overfitting

![](.\image\ML6.jpg)

![](.\image\ML7.jpg)

遇到一些训练时没遇到的词(P=0) 从而出现inf的绝对是啥的现象

![](.\image\ML8.jpg)

#### Parameter Estimation

假设您有一组 N 个样本点或观测值，x1，... ，xN，并且您认为这些数据是从一个由未知值 θ 参数化的分布中得到的。换句话说，你相信每个观测值的概率 Pθ (xi)是 θ 的函数。例如，我们可以抛一枚硬币，它的正面朝上的概率 θ。

你如何“学习”最有可能的值 θ 给你的样本？例如，如果我们有10次抛硬币，看到其中7次是正面，我们应该为 θ  选择什么值？这个问题的一个答案是推断 θ 等于从假定的概率分布中最大化选择样本 x 1，... ，xN  的概率的值。最大似然估计(MLE)是机器学习中经常使用的一种基本方法。最大似然估计通常作出以下简化假设:

- 每个样本来自同一分布。换句话说，每个 xi 是相同分布的。在我们抛硬币的例子中，每次抛硬币都有相同的正面朝上的机会  θ。
- 根据我们的分布参数，每个样本 xi  有条件地独立于其他样本。这是一个很有说服力的假设，但是我们将会看到，它极大地帮助简化了最大似然估计的问题，并且通常在实践中工作得很好。在抛硬币的例子中，一次抛硬币的结果不会影响其他任何一次。
- θ 的所有可能值在我们看到任何数据之前都是一样的(这被称为统一先验)。

The first two assumptions above are often referred to as **independent, identically distributed (i.i.d.)**.

##### Maximum Likelihood Estimation

![](.\image\ML9.jpg)

![](.\image\ML10.jpg)

![](.\image\ML11.jpg)

##### Smoothing

![](.\image\ML12.jpg)

![](.\image\ML13.jpg)

##### tuning

**调常数参数（Hyperparameters 超参数）比如Smoothing中的k来更好的符合测试**

![](.\image\ML14.jpg)

##### Classifier confidences

比如根据特征推测出是spam的概率是0.6和0.9，那么推测为0.9的应该更准确

![](.\image\ML15.jpg)

#### Summary

![](.\image\ML16.jpg)

### Perceptrons 感知器

#### Linear Classifiers

可以用它来进行二进制分类，即标签有正负两种可能性

线性分类器的基本思想是使用特征的线性组合进行分类——我们称之为激活。具体来说，激活函数接收一个数据点，将数据点 fi (x)的每个特征乘以相应的权值 wi，然后输出所有结果值的总和。在向量形式中，我们也可以把它写成权重的点乘，w，特征数据点写成

![](.\image\ML18.jpg)

![](.\image\ML17.jpg)

对于二进制分类，当数据点激活为正时，我们用正标签对数据点进行分类，如果数据点激活为负，我们用负标签对数据点进行分类。

![](.\image\ML19.jpg)

我们称这条蓝线为**决策边界**，因为它是我们将数据点分类为正数和负数的区域之间的分界线。在高维空间中，线性决策边界通常被称为**超平面**。超平面是一个比潜在空间低一维的线性表面，因此将表面一分为二。对于一般的分类器(非线性分类器)  ，决策边界可能不是线性的，而是简单地定义为分离类的特征向量空间中的曲面。

#### Binary Perceptron

**update weight**

![](.\image\ML20.jpg)

![](.\image\ML21.jpg)

#### Multiclass Perceptron

对于多类情况，每个类有一个权重向量，所以在3类情况下，我们有3个权重向量。为了对样本进行分类，我们通过取特征向量与每个权重向量的点乘来计算每个类的得分。我们选择得分最高的类作为我们的预测。

![](.\image\ML22.jpg)

![](.\image\ML23.jpg)

![](.\image\ML24.jpg)

`BIAS` 例如f(x)=a*x+b中的b 如果没有这个w就全过原点

### Logistic Regression 逻辑回归

#### **problem**

![](.\image\ML25.jpg)

#### **probabilistic decisions**

**两类**: 对每种情况的得分可以转化成概率 比如利用Sigmoidfunction 

![](.\image\ML26.jpg)

**大于两类**

![](.\image\ML28.jpg)

![](.\image\ML27.jpg)

可以求出每条分割线上的点求出的概率值，从而选择0.5 0.5的那条

![](.\image\ML29.jpg)

![](.\image\ML30.jpg)

#### Optimization

可以使用梯度下降法方法来优化参数，以获得高精度的训练数据。例如，假设我们已经设计了一些分类网络来输出数据点 x 的类 y 的概率，并且有 m  个不同的训练数据点。设 w 是网络的所有参数。我们希望找到参数 w  的值，使得我们的数据的真实class概率的可能性最大化.

![](.\image\ML31.jpg)

![](.\image\ML32.jpg)

**一维**

![](.\image\ML33.jpg)

##### Gradient Ascent 梯度上升

在每次迭代梯度下降法时，使用所有的数据点 x (1) ，... ，x (m)来计算参数 w 的梯度，更新参数，然后重复，直到参数收敛(此时我们已经达到了函数的局部最大值)。

![](.\image\ML34.jpg)

![](.\image\ML35.jpg)

![](.\image\ML36.jpg)

###### Batch批量 Gradient Ascent 

![](.\image\ML37.jpg)

###### Stochastic随机 Gradient Ascen 

![](.\image\ML38.jpg)

###### Mini-Batch Gradient Ascent 

![](.\image\ML39.jpg)

### Neural Networks

![](.\image\ML40.jpg)

![](.\image\ML41.jpg)

#### [example](http://playground.tensorflow.org/)

#### Backpropagation Algorithm

为了有效地计算神经网络中每个参数的梯度，我们将使用一种称为反向传播的算法。

![](.\image\ML42.jpg)

#### 

#### Residual Neural Networks 残差神经网络

![](.\image\ML44.jpg)

![](.\image\ML43.jpg)

![](.\image\ML45.jpg)

![](.\image\ML46.jpg)

### Kernels

**黑盒打包了相似度的计算**

#### Case-Based Learning

![](.\image\ML47.jpg)

![](.\image\ML48.jpg)

##### Similarity Functions

**basic 向量点积**

![](.\image\ML49.jpg)

**本该不变的相似度**

![](.\image\ML50.jpg)

**例如旋转相似度 把图片旋转, 选择相似度最大的作为两个图片的相似度**

![](.\image\ML51.jpg)

**可变形**

![](.\image\ML52.jpg)

#### Kernelization

![](.\image\ML53.jpg)

![](.\image\ML55.jpg)

##### Dual Perceptron 双重感知器

![](.\image\ML54.jpg)

![](.\image\ML56.jpg)

![](.\image\ML57.jpg)

#### Non-Linear Separators 非线性分离器

![](.\image\ML58.jpg)

![](.\image\ML59.jpg)

![](.\image\ML60.jpg)

### Clustering 聚类

集群系统: 非监督式学习、检测未标记数据的模式，当不知道你在寻找什么的时候有用，需要数据，但没有标签，经常会出现乱码。

#### K-Means Clustering k均值聚类

先随机选取K个对象作为初始的聚类中心。

然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。聚类中心以及分配给它们的对象就代表一个聚类。

一旦全部对象都被分配了，每个聚类的聚类中心会根据聚类中现有的对象被重新计算。

这个过程将不断重复直到满足某个终止条件。

![](.\image\ML61.jpg)

#### Agglomerative Clustering 层次聚类

把相似的合并到一起

![](.\image\ML62.jpg)

### Decision Trees

#### Inductive Learning 归纳学习

**分类和回归**

![](.\image\ML63.jpg)

![](.\image\ML64.jpg)

#### Decision Tree

**data**

![](.\image\ML71.jpg)

![](.\image\ML65.jpg)

**感知器和决策树的比较**

![](.\image\ML66.jpg)

**算法**

![](.\image\ML67.jpg)

![](.\image\ML68.jpg)

**entropy**

![](.\image\ML69.jpg)

**用熵来衡量分割的标准(Information Gain)，例如下图左边的熵比右边的熵小，更好**

![](.\image\ML70.jpg)

![](.\image\ML72.jpg)

**使用卡方检验避免过拟合**

![](.\image\ML73.jpg)

![](.\image\ML74.jpg)

**构建好几颗决策树，然后分别预测，然后选更多决策树的结果**

![](.\image\ML75.jpg)