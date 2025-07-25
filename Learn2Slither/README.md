<!--HEADER-->
<h1 align="center"> 42 Outer | 
 <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/42/white">
  <img alt="42" width=40 align="top" src="https://cdn.simpleicons.org/42/Black">
 </picture>
 Cursus 
  <img alt="Complete" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/complete.svg">
</h1>

<!--FINISH HEADER-->

## Learn2Slither Reinforcement Learning

### Descripion
- With this exercise we lean how to create an environment and how to use it for a reinforcement learning.

 <img alt="Warning" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/warning.svg"> Folder MyNN is an experimental work using my own NN which is not yet finished. Ongoing work
</h1>

##### File configuration
- **__snake.py__**: Script for the game itself;
- **__snake_class.py__**: Class that executtes the cmplete hame ith it's options;
- **__motor_snake.py__**: Class where the logic of the game is programmed;
- **__dqn_agent_keras.py__**: Class where the agent calculates the Q_Learnng function;
- **__dl_q_model_keras.py__**: Neurak network where the model is learnt.
- **__env_snake.py__**: Class where the environment is obtained for the agent to learn;
- **__learn_snake.py__**: Script to execute and save a learning model outside the main program snake. done to be faster for calculating prouposes.;
- **__view_lengths.py__**: Script to visualize the history lengths generated by learn_snake script;
    ##### ENUM CLASES
- **__actions.py__**: Script for the game itself;
- **__rewards.py      
.py__**: Script for the game itself;
- **__collisions.py__**: Script for the game itself;
- **__rewards.py__**: Script for the game itself;
- **__directions.py__**: Script for the game itself;
    ##### FOR SUPPORT
- **__count_rewards.py__**: Script to count events during learn and tdisplay them;

##### Description
- **Execution**: 
    
    ##### To play
    pyhton snake.py -f models/model_Q_1K.pt -c 1- -t 10

    ##### To learn
    
    python learn_snake.py -f <model file --history_lengths <file to save hitory>.csv -e 1000

### Pictures

<p>
  <img src="./images/Screenshot from 2025-07-05 15-16-29.png">
  <img src="./images/Screenshot from 2025-07-05 15-17-15.png">
  <img src="./images/Screenshot from 2025-07-05 15-17-56.png">
  <img src="./images/Screenshot from 2025-07-05 15-18-08.png">
</p> 

Screenshot from 2025-07-05 15-13-26.png
### Resources

* **[Gymnasium lybrary to create the environment](https://gymnasium.farama.org/)**

* **[Hugging face course to learn how to Reinforcement learning works](https://huggingface.co/learn/deep-rl-course/)**