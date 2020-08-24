import gym
import gym_pid
from DDPG import Agent
import numpy as np
import utils
import matplotlib.pyplot as plt

env = gym.make('pid-v0')
agent = Agent(alpha=0.0001, beta=0.001, input_dims=[5], tau=0.0001, env=env,
              batch_size=64,  layer1_size=256, layer2_size=128, n_actions=3)

#agent.load_models()
np.random.seed(0)

score_history=[]
average_25_score =[]
num_episodes = 200
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    score = 0
    rewards = []
    yhistory = [0]
    for step in range(100) :
        
        act = agent.choose_action(obs)
        # print(act)
        new_state, reward, done = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
        # if step % 10 == 0:
        #     env.render()
        yhistory.append(new_state[0])
        rewards.append(reward)
        # if step == 99:
        #     plt.plot(yhistory)
        #     plt.show()
        if score < -7000:
            print("Junk Episode")
            break
            
    score_history.append(score)

    # if episode % 25 == 0:
    #     agent.save_models()
    #     plt.plot(yhistory)
    #     plt.show()
    #     env.render()

    print('episode ', episode, 'score %.2f' % score,
          'trailing 25 games avg %.3f' % np.mean(score_history[-25:]))

    average_25_score.append(np.mean(score_history[-25:]))

figure, axes = plt.subplots(2)

plt.figure(0)
plt.plot(list(range(num_episodes)),score_history)

plt.figure(1)
plt.plot(list(range(num_episodes)),average_25_score)

plt.show()
