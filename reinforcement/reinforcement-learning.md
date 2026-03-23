# Reinforcement Learning

**Type:** Learning Paradigm (Sequential Decision Making)  
**Family:** Reward-Based Learning  
**Core Idea:** Learn optimal actions through interaction with an environment to maximize cumulative reward  

---

## 📌 Definition
Reinforcement Learning (RL) is a learning paradigm where an agent interacts with an environment and learns a policy to take actions that maximize cumulative reward over time.

![](<../Images/rl-01.jpg>)
---

## 🧠 Intuition
Think of training a dog:

- Good action → reward  
- Bad action → penalty  

Over time, the agent learns which actions lead to better long-term outcomes.

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Agent observes current state  
- Step 2: Agent takes an action  
- Step 3: Environment returns reward and next state  
- Step 4: Agent updates its policy based on feedback  
- Step 5: Repeat over many episodes  

---

## 🧮 Mathematics

### Markov Decision Process (MDP)

An RL problem is modeled as:

$$
(S, A, P, R, \gamma)
$$

Where:
- $S$ = set of states  
- $A$ = set of actions  
- $P$ = transition probability  
- $R$ = reward function  
- $\gamma$ = discount factor  

---

### Return (Cumulative Reward)

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots
$$

---

### Value Function

$$
V(s) = \mathbb{E}[G_t \mid s_t = s]
$$

---

### Q-Function (Action-Value)

$$
Q(s, a) = \mathbb{E}[G_t \mid s_t = s, a_t = a]
$$

---

### Bellman Equation

$$
V(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right]
$$

---

## 🔢 Vector / Matrix Form
In tabular RL, value functions can be represented as vectors:

$$
V = R + \gamma P V
$$

---

## 🎯 Objective
Learn a policy $\pi(a \mid s)$ that maximizes expected cumulative reward:

$$
\max_\pi \; \mathbb{E}[G_t]
$$

---

## 📈 When to Use
- Sequential decision-making problems  
- When actions affect future outcomes  
- Robotics, games, control systems  
- Situations with delayed rewards  

---

## ⚠️ Limitations
- Requires large number of interactions  
- Exploration vs exploitation trade-off  
- Can be unstable and hard to train  
- Sensitive to reward design  

---

## ⚖️ Bias-Variance Behavior
- Depends on algorithm (Q-learning, policy gradient, etc.)  
- High variance in policy gradient methods  
- Can overfit to environment if not generalized  

---

## 🔧 Key Hyperparameters
- learning_rate (α): Step size for updates  
- discount_factor (γ): Importance of future rewards  
- epsilon: Exploration rate (ε-greedy)  

---

## 🔄 Variants / Extensions
- Q-Learning  
- SARSA  
- Deep Q Networks (DQN)  
- Policy Gradient Methods  
- Actor-Critic  

---

## 🔗 Related Algorithms
- Dynamic Programming (Bellman equations)  
- Markov Chains  
- Neural Networks (Deep RL)  

---

## 💻 Implementation (Minimal)
```python
import numpy as np

# simple Q-learning update
Q = np.zeros((5, 2))  # 5 states, 2 actions

alpha = 0.1
gamma = 0.9

state = 0
action = 1
reward = 1
next_state = 2

Q[state, action] = Q[state, action] + alpha * (
    reward + gamma * np.max(Q[next_state]) - Q[state, action]
)

print(Q)
```