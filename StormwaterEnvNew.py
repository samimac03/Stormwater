import numpy as np
import random
from reward_functions import reward_new as reward_function
from pyswmm import Simulation, Nodes, Links
import matplotlib.pyplot as plt
import gym
import math
import csv


class StormwaterEnv(gym.Env):

    def __init__(self, swmm_inp=""):
        self.floodings = -1
        self.eps_flooding = []
        self.eps_depths = [[],[],[]]
        self.eps_action = [[],[],[]]

        super(StormwaterEnv, self).__init__()
        self.total_rewards = []
        self.eps_reward = 0

        self.num_steps = 0

        self.timestep = 0
        self.time = 0
        self.swmm_inp = swmm_inp        
        self.temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
        self.temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting
        self.i = 0
        self.flood = []
        
        self.log_dumps = []
        
        self.links = ['R1','R2','R3']
        self.nodes = ['St1', 'St2', 'St3']

        self.rl, self.baseline = [],[]

    def reset(self,date=False,test=False):
        self.test = test
        self.settings, self.depths, self.flooding = [],[],[]
        self.i += 1
        self.swmm_inp="data/simple1.inp"
        dTest = date
        if(not dTest):
            with open("data/dates.txt","r") as f:
                date = list(csv.reader(f, delimiter=' '))
                random.Random(10).shuffle(date)
                if not test:
                    time = (random.randint(0,int(len(date)*0.9)))
                    self.test = False
                else:
                    if not self.test:
                        time = int(len(date)*0.85)
                        self.test = True

                        time = 0
                    else:
                        self.time += 1
                        time = self.time

                self.time = time
                
                year = int(date[time][0])
                month = int(date[time][1])
                day = int(date[time][2])

                self.t = (str(year) +" "+ str(month) +" "+ str(day))

            self.date = [month,day,year]
            with open("data/simple1.inp", "r") as File:
                lines = File.readlines()
            with open("data/simple1.inp", "w") as File:   
                for line in lines:
                    if "REPORT_START_DATE" in line:
                        File.write(line)
                    elif "START_DATE" in line:
                            File.write("START_DATE  " + str(month)+"/"+str(day)+"/"+str(year)+ "\n")    
                    elif "END_DATE" in line:
                            File.write("END_DATE  " + str((month))+"/"+str(day+1)+"/"+str(year) + "\n") 
                    else:
                        File.write(line)
        if not dTest:
            self.eps_flooding.append(self.floodings)
            self.total_rewards.append(self.eps_reward)
        self.timestep += 1

        self.eps_reward = 0
        self.floodings = 0
        self.flood = []
        self.eps_depths = [[],[],[]]
        self.eps_action = [[],[],[]]

        self.num_steps = 0
        
        self.done = False

        self.current_step = 0


        self.sim = Simulation(self.swmm_inp)

        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)
        print(self.i)
        self.sim.start()

        self.sim_len = self.sim.end_time - self.sim.start_time

        self.num_steps = int(self.sim_len.total_seconds()/self.control_time_step)
        
        x = 0
        for i in self.links:
            y = random.randrange(1)
            self.link_object[i].target_setting = y
            self.eps_action[x].append(y)
            x += 1

        for i in self.links:
            self.settings.append(self.link_object[i].current_setting)

        for i in self.nodes:
            self.depths.append(self.node_object[i].depth)
            self.flooding.append(self.node_object[i].flooding)
        self.reward = reward_function(self.depths, self.flooding)
        
        self.eps_reward += self.reward
        
        state = []
        
        state = self.settings.copy()
        state.extend(self.depths)
        state.extend(self.flooding)
        
        self.day = []
        time =list(open("forcast.txt", "r"))
        time = list(map(lambda s: s.strip(), time))
        for i in time:
            if(self.t in i):
                self.day.append(i)


        time = str(self.sim.current_time).split(" ")
        time = time[1].split(":")

        cur = self.day[int(time[0])].split(" ")[3]
        nxt = self.day[int(time[0])+1].split(" ")[3]

        #cur = int(float(cur) * random.uniform(.95,1.05))
        #nxt = int(float(nxt) * random.uniform(.85,1.15))

        state.extend([cur,nxt])
        xx = []
        for x in state:
            xx.append(float(x) * random.uniform(0.9,1.1))

        #state = [float(x) * random.uniform(0.9,1.1) for x in state]
        return state


    def step(self, action):
    
        self.current_step += 1
        self.settings, self.depths, self.flooding = [],[],[]
        
        
        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)
        
        x = 0
        for i in self.links:
            self.link_object[i].target_setting = action[x]
            self.eps_action[x].append(action[x])
            x += 1

        self.sim.__next__()
        
        for i in self.links:
            self.settings.append(self.link_object[i].current_setting)
        ii = 0

        for i in self.nodes:
            self.depths.append(self.node_object[i].depth)
            self.flooding.append(self.node_object[i].flooding)
            self.eps_depths[ii].append(self.node_object[i].depth)
            ii += 1
        self.floodings += sum(self.flooding)
        self.flood.append(sum(self.flooding))
        self.reward = reward_function(self.depths, self.flooding)

        self.eps_reward += self.reward

        state = []
        state = self.settings.copy()
        state.extend(self.depths)
        state.extend(self.flooding)
        
        time = str(self.sim.current_time).split(" ")
        time = time[1].split(":")

        cur = self.day[int(time[0])].split(" ")[3]
        if(int(time[0]) < 23):
            nxt = self.day[int(time[0])+1].split(" ")[3]
        else:
            nxt = 0

        cur = int(float(cur) * random.uniform(.95,1.05))
        nxt = int(float(nxt) * random.uniform(.85,1.15))

        state.extend([cur,nxt])

        xx = []
        for x in state:
            xx.append(float(x) * random.uniform(0.9,1.1))
        return state, self.reward, self.done #, {} # {} is debugging information
    
    def close(self):
        self.sim.report()
        self.sim.close()

    def graph(self, location, end=False):
        with open("plots/data.csv", "a") as f:
                writer = csv.writer(f,delimiter=',')
                writer.writerow([self.i])
        self.eps_flooding.append(self.floodings)
        self.total_rewards.append(self.eps_reward)
        self.eps_depths.append(self.eps_reward)

        self.old_flooding = self.floodings
        self.old_rewards = self.eps_reward
        self.floodrl = self.flood
        self.depthsRL = self.eps_depths
        self.actionRL = self.eps_action

      
        self.reset(date=True)
        for i in range(self.num_steps-1):
            self.step([1,1,1])

        self.flooding1 = self.floodings
        self.eps_reward1 = self.eps_reward
        self.flood1 = self.flood
        self.depths1 = self.eps_depths
        self.action1 = self.eps_action
       
        
        #"Flooding_Sum_P, Reward_Sum_P, Flooding_CFS_P, J1_Depths_P,J2_Depths_P,J3_Depths_P,R1_P,R2_P, Flooding_Sum_RL, Reward_Sum_RL, Flooding_CFS_RL, J1_Depths_RL,J2_Depths_RL,J3_Depths_RL,R1_RL,R2_RL"
        rows =([[self.flooding1], [self.eps_reward1], self.flood1, self.depths1[0],self.depths1[1],self.depths1[2], self.action1[0], self.action1[1], \
            [self.old_flooding], [self.old_rewards], self.floodrl, self.depthsRL[0],self.depthsRL[1],self.depthsRL[2], self.actionRL[0], self.actionRL[1]])
            
        columns= ["Flooding_Sum_P", "Reward_Sum_P", "Flooding_CFS_P", "J1_Depths_P","J2_Depths_P","J3_Depths_P","R1_P","R2_P", "Flooding_Sum_RL", "Reward_Sum_RL", "Flooding_CFS_RL", "J1_Depths_RL","J2_Depths_RL","J3_Depths_RL","R1_RL","R2_RL"]

        with open("plots/data.csv", "a") as f:
            writer = csv.writer(f,delimiter=',')
            x = 0
            for row in rows:
                row = list(row)
                row.insert(0,columns[x])
                writer.writerow(row)
                x += 1