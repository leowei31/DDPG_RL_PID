import gym
import matplotlib.pyplot as plt
import numpy as np

class PidEnv(gym.Env):
    def __init__(self, sample_rate=1, setpoint=50):
        self.sample_rate = sample_rate
        self.setpoint = setpoint
        self.error = self.setpoint
        self.proportional = 0
        self.integral = 0
        self.last_error = self.error
        self.currpoint = [0, 0]
        self.kp = 0.5
        self.ki = 0.5
        self.n = 200 # Simulation points


    def step(self, action):
        self.currpoint = [0, 0]
        self.xhistory = [0]
        self.yhistory = [0]
        self.kp = action[0] # Increasing p term reduces rise time
        self.ki = action[1]
        done = False

        while(self.currpoint[0]< self.n and done == False ):
            # max x axis of n points 
            self.proportional = self.kp * self.error
            self.integral += self.ki * self.error * self.sample_rate

            curr_input = self.proportional + self.integral

            self.last_error = self.error
            self.currpoint[1] += curr_input
            self.currpoint[0] += 1 
            self.error = self.setpoint - float(self.currpoint[1])
            self.xhistory.append(self.currpoint[0])
            self.yhistory.append(self.currpoint[1])
            done_error = abs(self.currpoint[1]) - abs(self.yhistory[self.currpoint[0]-1])
            if(done_error< 0.0005 and done_error > -0.0005):
                done = True

        self.state = [self.kp, self.ki]
        reward = -abs(self.error) - 0.05*self.currpoint[0]
        if reward > -5:
            reward = 10
        return np.array(self.state, dtype =np.float32), reward, done, {}

    def reset(self):
        self.error = self.setpoint
        self.proportional = 0
        self.integral = 0
        self.last_error = self.error
        self.currpoint = [0,0]
        self.kp = 0.5
        self.ki = 0.5
        self.continous = False
        return self.step(np.array([0,0,0]))[0]

    def render(self):
        print("Error: "+str(self.error))
        print("Proportional Term: "+str(self.proportional))
        print("Integral Term: "+str(self.integral))
        plt.plot(self.xhistory, self.yhistory)
        plt.show()
