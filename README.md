## Project Overview

This project involves training a Proximal Policy Optimization (PPO) model to solve the Lunar Lander environment from OpenAI Gym. The goal is to optimize the model's hyperparameters to achieve stable and efficient landings.

---

## Repository Structure

Here is an overview of the files in this repository:

1. **`Lunar_lander_py.ipynb`**: Jupyter notebook containing the main code for setting up the environment, training the PPO model, and evaluating its performance.
2. **`best_hyperparameters.json`**: JSON file containing the best hyperparameters found using Optuna.
3. **`system_info.txt`**: Text file with system information used during the experiment, including Python version, library versions, and hardware details.
4. **`policy.pth`**: The final trained model's policy parameters.
5. **`policy.optimizer.pth`**: The optimizer state for the trained model.
6. **`pytorch_variables.pth`**: Additional PyTorch variables used during training.

---

## Getting Started

### Prerequisites

To run this project, you will need:

- Python 3.10 or higher
- `gymnasium` library
- `stable-baselines3` library
- `optuna` library
- `pytorch` library

### Installation

1. **Clone the repository**:

    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

    Ensure `requirements.txt` contains all necessary libraries:
    ```plaintext
    gymnasium
    stable-baselines3
    optuna
    torch
    ```

---

## Running the Project

### Training the Model

To train the model using the provided notebook:

1. Open the Jupyter notebook `Lunar_lander_py.ipynb`.

2. Follow the steps to train the PPO model. The notebook is structured to guide you through:
   - Setting up the Lunar Lander environment.
   - Using Optuna to find the best hyperparameters.
   - Training the final model using the best-found hyperparameters.
   - Saving and evaluating the trained model.

### Using the Trained Model

If you want to use the pre-trained model:

1. Load the model using the provided weights (`policy.pth`, `policy.optimizer.pth`, and `pytorch_variables.pth`):

    ```python
    import torch
    from stable_baselines3 import PPO

    model = PPO.load("ppo_lunarlander_v2_best")
    ```

2. Evaluate the model:

    ```python
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym
    from stable_baselines3.common.evaluation import evaluate_policy

    env = DummyVecEnv([lambda: gym.make("LunarLander-v2")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    ```

### Detailed Steps for Hyperparameter Optimization

1. **Define the Objective Function**:
    - Create a function to train the model with suggested hyperparameters and return the mean reward.

2. **Optimize with Optuna**:
    - Use Optuna to run multiple trials, each with different hyperparameter values, to find the optimal set.

3. **Save Best Hyperparameters**:
    - Save the best hyperparameters found into a JSON file.

4. **Train Final Model**:
    - Use the best hyperparameters to train the final model and save the trained model's parameters.

### Example Workflow

Here’s a simplified workflow outline:
1. **Set up environment**:

    ```python
    import gymnasium as gym
    env = gym.make("LunarLander-v2")
    ```

2. **Train model with chosen hyperparameters**:

    ```python
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.00025,
        n_steps=1024,
        batch_size=128,
        n_epochs=4,
        gamma=0.999,
        verbose=1
    )
    model.learn(total_timesteps=1000000)
    model.save("ppo_lunarlander_v2_best")
    ```

3. **Evaluate the model**:

    ```python
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    ```

---

## Conclusion

This project demonstrates the process of optimizing a PPO model for the Lunar Lander environment. By following the steps outlined, you can reproduce the results, tweak the hyperparameters further, and improve the model's performance.

For any questions or contributions, feel free to open an issue or submit a pull request. Happy coding!
