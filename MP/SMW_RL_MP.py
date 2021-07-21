import gym
import retro
import numpy as np
import random
import copy
import tensorflow as tf
import cv2
import time
import sys

from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import ModelIntervalCheckpoint
from gym.spaces import Box

dateTimeNow = datetime.now()
episodeReward = 0
xPos = 0
xPosMax = 0
frameCounter = 1
numIteracion = 0
showImageTransformation = False

if len(sys.argv) == 2 and (sys.argv[1] == '-v' or sys.argv[1] == '-V'):
    showImageTransformation = True
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image4', cv2.WINDOW_NORMAL)

f_real = open("training_real_stats_"+str(dateTimeNow)+".csv","w+")
f_real.write("Iteracion" + ";" + "Reward" + "\n")

def processFrame(frame, shape=(100,100)):
    frame = frame.astype(np.uint8)
    if showImageTransformation: cv2.imshow('image',frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if showImageTransformation: cv2.imshow('image2',frame)

    frame = frame[40:40+224, :224]
    if showImageTransformation: cv2.imshow('image3',frame)

    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    if showImageTransformation: cv2.imshow('image4',frame)

    frame = frame.reshape((*shape, 1))

    if showImageTransformation: cv2.waitKey(1)
    
    return frame

class ObservationDiscretizer(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, act):
        global episodeReward
        global xPosMax
        global xPos
        global frameCounter
        global numIteracion
        global f_real
        
        obs, rew, done, info = super().step(act)
        episodeReward += rew

        xPos = info['x']

        if (xPosMax < xPos):
            frameCounter = 1
            xPosMax = xPos
            episodeReward += 5
        else:
            frameCounter += 1
            episodeReward = float(int(episodeReward * 0.999)) #Restamos el 0.1%

        if frameCounter % 300 == 0:
            episodeReward = float(int(episodeReward * 0.95)) #Restamos el 5%

        if frameCounter == 1200:
            #print("RESETEAR")
            done = True

        if info['endOfLevel']:
            #print("FIN NIVEL")
            episodeReward += 30000
            done = True

        if info['dead'] == 0:
            #print("MUERTO")
            episodeReward = int(episodeReward * 0.70) #Restamos el 30%
            done = True

        rew = episodeReward

        if done:
            numIteracion += 1
            f_real.write(str(numIteracion) + ";" + str(rew) + "\n")

        #print("Total acumulado: " + str(rew) + " Valor frame counter: " + str(frameCounter))

        return obs, rew, done, info

    def observation(self, obs):
        obs = processFrame(obs)
        return obs

class MarioObservationDiscretizer(ObservationDiscretizer):
    def __init__(self, env):
        super().__init__(env)

class ActionDiscretizer(gym.ActionWrapper):
    def __init__(self, env, combos):
        super(ActionDiscretizer,self).__init__(env)
        self.env = env
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))
    
    def reset(self, **kwargs):
        global episodeReward
        global xPosMax 
        global xPos
        global frameCounter

        xPos = 0
        xPosMax = 0
        episodeReward = 0
        frameCounter = 1

        return self.env.reset(**kwargs)

    def action(self, act):
        return self._decode_discrete_action[act].copy()
    
class MarioActionDiscretizer(ActionDiscretizer):
    def __init__(self, env):
        obs= env.reset()
        super().__init__(env, combos=[['LEFT'], ['RIGHT'], ['RIGHT', 'B'], ['RIGHT', 'Y'], ['B'], ['A']])


def build_model(height, width, channels, actions):
    model = Sequential()

    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3, height, width, 1)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))

    return model

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)

    memory = SequentialMemory(limit=1000, window_length=3)

    dqnAgent = DQNAgent(model=model, memory=memory, policy=policy,
                        enable_dueling_network=True, dueling_type='avg', 
                        nb_actions=actions, nb_steps_warmup=100
                       )

    return dqnAgent

def main():
    global f_real
    global numIteracion
    
    env = MarioActionDiscretizer(retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2.state', record='./records/'))
    env = MarioObservationDiscretizer(env)
    
    height, width, channels = 100, 100, 1
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)
    
    agent = build_agent(model, actions)
    agent.compile(Adam(lr=1e-4))

    '''
        Creamos un callback para ir guardando checkpoints con los pesos
    '''
    filepath='./checkpoints/'
    checkpoint = ModelIntervalCheckpoint(filepath,interval=10)

    time.sleep(5)

    print("\n\n-> ENTRENAMIENTO DEL AGENTE")
    start_time = time.time()
    training_stats = agent.fit(env, nb_steps=1000000, visualize=False, verbose=2)
    finish_time = time.time()
    f = open("training_stats_"+str(dateTimeNow)+".txt","w+")
    f.write(str(training_stats.params))
    f.write("\n###########################################\n")
    f.write(str(training_stats.history.keys()))
    f.write("\n###########################################\n")
    f.write(str(training_stats.history))
    f.write("\n###########################################\n")
    f.write("Tiempo entrenando: " + str(finish_time - start_time))
    f.close()
    f_real.close()

    print("\n\n-> VALIDACION DEL AGENTE")
    numIteracion = 0
    f_real = open("validating_real_stats_"+str(dateTimeNow)+".csv","w+")
    f_real.write("Iteracion" + ";" + "Reward" + "\n")
    start_time = time.time()
    scores = agent.test(env, nb_episodes=10, visualize=False)
    finish_time = time.time()
    print("\n\n-> GUARDANDO LOS SCORES EN FICHERO")
    f = open("validating_stats_"+str(dateTimeNow)+".txt","w+")
    f.write(str(scores.params))
    f.write("\n###########################################\n")
    f.write(str(scores.history.keys()))
    f.write("\n###########################################\n")
    f.write(str(scores.history))
    f.write("\n###########################################\n")
    f.write("Tiempo validando: " + str(finish_time - start_time))
    f.close()
    f_real.close()
    
    env.close()

if __name__ == '__main__':
    main()
