# Interesting Graphs (Explanation)
The color at each point represents which action MCTS decides to take given state at that point
### With Turn Penalty:
[2 action decision bound (10 lookahead)](11-5_2action_db_10la.png)
[4 action decision bound (100 lookahead)](11-5_4action_db_100la.png)
- Using a penalty on decision bound (value - (1 for each non-terminal decision leading to the state)) 
- Using a final reward (equal to norm of root state)

### Without Turn Penalty:
[2 action decision bound (100 lookahead)](10-11_2action_decisionbound_100la.png)
[4 action decision bound (100 lookahead)](10-24_4action_db_100la.png)
- Raw MCTS with UCT value function
