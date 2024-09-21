import numpy as np
import itertools

def generate_all_combinations(d):
    """
    Generate all possible combinations of values for an arbitrary dictionary.
    
    Args:
        d (dict): Dictionary with lists as values.
    
    Returns:
        List of dictionaries containing all possible combinations.
    """
    # Extract lists of values from the dictionary
    value_lists = [d[key] for key in d]
    
    # Generate all possible combinations using itertools.product
    all_combinations = itertools.product(*value_lists)
    
    # Convert combinations to a list of dictionaries
    list_of_dicts = [{key: combination[i] for i, key in enumerate(d)} for combination in all_combinations]
    
    return list_of_dicts

# Helper function: Bernoulli sample for P(μ)
def sample_bernoulli(p):
    """_summary_

    Args:
        p (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return np.random.binomial(1, p)

def initialize_theta(num_features, num_values_per_feature):
    """
    Initializes theta: the feature selection probabilities (p)
    and the feature value selection probabilities (q).
    
    Args:
    - num_features: Number of features.
    - num_values_per_feature: Number of possible values for each feature.
    
    Returns:
    - p: Probability distribution over features.
    - q: List of probability distributions for each feature's possible values.
    """
    p = np.random.uniform(0, 1, size=num_features)  # Initialize p randomly in [0, 1]
    q = [np.random.randn(num_values_per_feature) for _ in range(num_features)]  # Random values for q
    
    return p, q

# Helper function: softmax for Q_c(ν)
def compute_softmax(q_c):
    """_summary_

    Args:
        q_c (_type_): _description_

    Returns:
        _type_: _description_
    """    
    exp_q = np.exp(q_c - np.max(q_c))  # For numerical stability
    return exp_q / np.sum(exp_q)

# Apply action to the instance x to get modified instance x'
def apply_action(x, action):
    """_summary_

    Args:
        x (_type_): _description_
        action (_type_): _description_
        C (_type_): _description_

    Returns:
        _type_: _description_
    """    
    x_prime = x.copy().flatten()
    for feature, value in action.items():
        x_prime[feature] = value
    return x_prime

# Optimize the policy using REINFORCE algorithm
def reinforce_algorithm(
        env,
        V_C: dict,  # V_C: Dictionary of features and possible values
        gamma, 
        num_episodes, 
        alpha, 
        lambda1, 
        lambda2, 
        w
):
    """
    Solves the counterfactual feature selection problem using the REINFORCE algorithm.
    
    Args:
    - env: The environment (classifier f, etc.)
    - V_C: Dictionary where keys are features and values are possible feature values.
    - gamma: Discount factor for future rewards.
    - num_episodes: Number of episodes to run.
    - alpha: Learning rate.
    - lambda1: Regularization coefficient for l1 penalty.
    - lambda2: Regularization coefficient for entropy.
    - w: Sparsity constraint on the number of non-zero elements in p.
    
    Returns:
    - p: Optimized feature selection probabilities.
    - q: Optimized feature value selection probabilities.
    """
    
    # Determine the number of features and number of values per feature based on V_C
    s = len(list(V_C.keys()))  # Number of features
    m = len(list(V_C.values())[0]) # Number of possible values per feature
    p = np.random.rand(s)
    q = [np.random.rand(m) for _ in range(s)]  
    print(V_C)
    list_of_actions = generate_all_combinations(V_C)
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        rewards = []
        
        while not done:
            # Sample feature selection (mu) and feature values (nu)
            μ = sample_bernoulli(p)  # Sample from Bernoulli(p)
            ν = [np.random.choice(len(q_c), p=compute_softmax(q_c)) for q_c in q]
            
            for action in list_of_actions:
                # Apply action to get the modified state (x')
                x_prime = apply_action(state, action)
                
                # Interact with the environment
                next_state, reward, done, _ = env.step(x_prime)
                trajectory.append((state, (μ, ν), reward))
                rewards.append(reward)
                state = next_state
        
        #TODO add logging
        
        # Compute rewards-to-go (G)
        G = 0
        returns = []
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G  # Discount future rewards
            returns.insert(0, G)
        
        # Update policy parameters (p and q) based on trajectory
        for t, (state, (μ, ν), reward) in enumerate(trajectory):
            grad_log_pi_p = μ - p
            grad_log_pi_q = np.zeros_like(q)
            
            for c, q_c in enumerate(q):
                softmax_probs = compute_softmax(q_c)
                grad_log_pi_q[c] = np.eye(len(q_c))[ν[c]] - softmax_probs
            
            # Apply regularization terms
            l1_regularization = lambda1 * np.sign(p)
            entropy_regularization = lambda2 * (p * np.log(p + 1e-8))
            
            # Update p with the gradients and regularization
            p += alpha * grad_log_pi_p * returns[t] - alpha * (l1_regularization + entropy_regularization)
            p = np.clip(p, 0, 1)  # Ensure p stays within [0, 1]
            
            # Update q (feature value probabilities)
            for c, q_c in enumerate(q):
                q[c] += alpha * grad_log_pi_q[c] * returns[t]
        
        # Ensure that the number of non-zero elements in p doesn't exceed w
        if np.sum(p != 0) > w:
            indices = np.argsort(p)[::-1]  # Sort p in descending order
            top_indices = indices[:w]  # Get the top w features
            new_p = np.zeros_like(p)
            new_p[top_indices] = p[top_indices]  # Keep only the top w features
            p = new_p
    
    return p, q


# RL-based counterfactual feature optimization
def rl_based_counterfactual_optimization(
        env, 
        x,
        V_C, 
        w, 
        gamma, 
        num_episodes, 
        alpha, 
        lambda1, 
        lambda2
):
    """_summary_

    Args:
        env (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        C (_type_): _description_
        V_C (_type_): _description_
        w (_type_): _description_
        B (_type_): _description_
        gamma (_type_): _description_
        num_episodes (_type_): _description_
        alpha (_type_): _description_
        lambda1 (_type_): _description_
        lambda2 (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Initialize policy parameters
    p_star, q_star = reinforce_algorithm(env, V_C, gamma, num_episodes, alpha, lambda1, lambda2, w)
    list_of_actions = generate_all_combinations(V_C)
    c_star = np.argsort(p_star)[::-1][:w]
    f_star = []
    for c in c_star:
        v_star_c = np.argmax(q_star[c])
        f_star.append((c, v_star_c))
    
    x_prime_0 = x.copy().flatten()
    
    # Step 4: Update instance based on feature values
    f_star_sorted = sorted(f_star, key=lambda x: p_star[x[0]], reverse=True)
    
    for c, v in f_star_sorted:
        x_prime_0[c] = v
        if env.predict(x_prime_0) > 0.5:
            break
    
    # Step 5: Sample B actions and construct examples
    examples = [x_prime_0]#
    for action in list_of_actions:
        x_prime = apply_action(x, action)
        if env.predict(x_prime) > 0.5:
            examples.append(x_prime)
    
    return examples

# # Example Usage
# if __name__ == "__main__":
#     class DummyEnv:
#         def __init__(self):
#             self.C = list(range(5))
        
#         def reset(self):
#             return np.random.rand(5)
        
#         def step(self, x_prime):
#             reward = np.random.rand()
#             next_state = x_prime
#             done = np.random.rand() > 0.95
#             return next_state, reward, done, {}
        
#         def predict(self, x):
#             return np.random.rand()
    
#     env = DummyEnv()
#     x = np.random.rand(5)
#     y = np.random.choice([0, 1])
#     C = list(range(5))
#     V_C =  {i: list(range(3)) for i in range(5)}
    
#     gamma = 0.99
#     num_episodes = 2
#     alpha = 0.01
#     lambda1 = 0.1
#     lambda2 = 0.01
#     w = 3
    
#     examples = rl_based_counterfactual_optimization(env, x, V_C, w,  gamma, num_episodes, alpha, lambda1, lambda2)
    
#     print("Counterfactual Examples:")
#     for example in examples:
#         print(example)
