from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm           # import colormap stuff!
import numpy as np


np.random.seed(19680801)

num_time_steps = 50
num_rewards = 4

x = np.array(range(0, num_rewards))
y = np.array(range(0, num_time_steps))

# 50*4
# 1 agent 2 POIs (1 of each type), temporal coupling of 1
'''
z = [[85, 4, 5, 6],
    [84, 5, 5, 6],
    [80, 4, 8, 7],
    [86, 6, 4, 4],
    [85, 4, 5, 6],
    [82, 6, 4, 8],
    [90, 2, 6, 2],
    [85, 5, 5, 5],
    [85, 4, 5, 6],
    [84, 7, 5, 4],
    [84, 6, 5, 5],
    [85, 4, 5, 6],
    [88, 2, 5, 5],
    [85, 4, 5, 6],
    [88, 2, 5, 5],
    [2, 10, 85, 3],
    [4, 10, 83, 3],
    [2, 5, 90, 3],
    [5, 7, 83, 2],
    [2, 10, 85, 3],
    [2, 10, 85, 3],
    [9, 2, 87, 2],
    [2, 10, 85, 3],
    [4, 6, 86, 3],
    [6, 6, 85, 3],
    [2, 10, 85, 3],
    [4, 4, 90, 2],
    [2, 10, 85, 3],
    [2, 4, 92, 2],
    [2, 10, 90, 3],
    [2, 10, 85, 1],
    [2, 10, 85, 1],
    [2, 10, 94, 3],
    [2, 10, 85, 0],
    [2, 10, 91, 2],
    [2, 10, 85, 1],
    [2, 10, 93, 2],
    [2, 10, 85, 1],
    [2, 10, 92, 1],
    [3, 10, 85, 2],
    [3, 3, 94, 0],
    [2, 5, 95, 0],
    [2, 5, 95, 0],
    [2, 2, 96, 0],
    [2, 8, 90, 0],
    [2, 5, 93, 1],
    [2, 4, 94, 1],
    [2, 6, 92, 1],
    [2, 2, 95, 1],
    [2, 1, 97, 0]]

z = np.array(z)
z.reshape(num_rewards, num_time_steps)


xpos, ypos = np.meshgrid(x, y)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = 0.5*np.ones_like(zpos)
dy = dx.copy()
dz = z.flatten()

colors = ["b", "r", "g", "black"] * num_time_steps

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

labels = ["go to POI-A", "go away from POI-A", "go to POI-B", "go away from POI-B"]
ax.set_title("Distribution of selected rewards", fontsize = 30)

ax.set_xticklabels(["go to POI-A", "go away from POI-A", "go to POI-B", "go away from POI-B"], fontsize = 20)
ax.set_xticks(range(num_rewards)) # how many ticks
ax.set_xlabel("Reward types", labelpad = 60, fontsize = 20)

ax.set_yticklabels(np.arange(0, num_time_steps+1, 5), fontsize = 20)
ax.set_yticks(np.arange(0, num_time_steps+1, 5))
#ax.set_yticks(range(num_time_steps))
ax.set_ylabel("Time step", labelpad = 30, fontsize = 20)

ax.set_zticklabels(np.arange(0, 101, 10), fontsize = 20)
ax.set_zticks(np.arange(0, 101, 10))
ax.set_zlabel("Percentage", labelpad = 30, fontsize = 20)

ax.bar3d(xpos,
          ypos,
          zpos,
          dx,
          dy,
          dz,
          color= colors)

plt.show()
'''

##### 2 agents

num_time_steps = 50
num_rewards = 6

x = np.array(range(0, num_rewards))
y = np.array(range(0, num_time_steps))

# Agent # 1
z1 = [[89, 2, 4, 0, 4, 1],
    [88, 1, 6, 1, 3, 2],
    [85, 2, 4, 0, 4, 1],
    [89, 3, 5, 1, 3, 2],
    [90, 2, 4, 1, 3, 3],
    [87, 4, 6, 2, 3, 1],
    [90, 3, 2, 1, 4, 1],
  [5, 1, 1, 1, 87, 1],
  [5, 1, 1, 1, 90, 1],
  [5, 1, 1, 1, 91, 1],
  [5, 1, 1, 1, 90, 1],
  [5, 1, 1, 1, 92, 1],
  [5, 1, 1, 1, 91, 1],
  [5, 1, 1, 1, 90, 1],
  [5, 1, 1, 1, 90, 1],
    [89, 1, 4, 1, 4, 2],
    [90, 1, 2, 1, 5, 3],
    [91, 1, 4, 1, 5, 3],
    [88, 1, 4, 2, 4, 2],
    [87, 1, 3, 0, 5, 1],
      [5, 1, 1, 1, 91, 1],
      [5, 1, 1, 1, 90, 1],
      [5, 1, 1, 1, 92, 1],
      [5, 1, 1, 1, 91, 1],
      [5, 1, 1, 1, 92, 1],
      [5, 1, 1, 1, 91, 1],
      [5, 1, 1, 1, 90, 1],
      [5, 1, 1, 1, 91, 1],
      [5, 1, 1, 1, 90, 1],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2],
    [89, 1, 4, 0, 6, 2]]

# Agent # 2
z2 = [[5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [5, 1, 87, 1, 1, 1],
    [4, 1, 89, 1, 4, 2],
    [3, 1, 88, 1, 3, 1],
    [4, 1, 89, 1, 3, 2],
    [5, 2, 89, 2, 2, 0],
    [3, 2, 85, 2, 3, 0],
    [2, 2, 85, 1, 4, 1],
    [4, 2, 91, 1, 2, 0],
    [6, 2, 92, 2, 0, 1],
    [2, 5, 86, 1, 1, 1],
    [2, 5, 88, 3, 1, 0],
    [3, 4, 86, 1, 2, 1],
    [4, 5, 88, 2, 2, 1],
    [5, 4, 88, 1, 3, 1],
    [4, 4, 85, 2, 4, 0],
    [3, 5, 87, 1, 1, 0],
    [4, 4, 88, 1, 2, 0],
    [5, 5, 90, 2, 2, 0],
    [2, 3, 83, 0, 2, 0],
    [1, 5, 85, 0, 2, 1],
    [1, 6, 89, 0, 1, 1],
    [2, 6, 88, 0, 1, 1],
    [1, 4, 89, 0, 2, 1],
    [2, 5, 86, 1, 2, 0],
    [1, 4, 88, 1, 1, 1],
    [1, 6, 87, 1, 0, 1],
    [1, 2, 89, 1, 1, 0],
    [1, 4, 90, 0, 1, 1]]


z1 = np.array(z1)
z2 = np.array(z2)

z1.reshape(num_rewards, num_time_steps)
z2.reshape(num_rewards, num_time_steps)


xpos, ypos = np.meshgrid(x, y)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = 0.5*np.ones_like(zpos)
dy = dx.copy()
dz1 = z1.flatten()
dz2 = z2.flatten()

colors1 = ["b", "r", "g", "black", "m", "y"] * num_time_steps
colors2 = ["r", "b", "y", "g", "m", "black" ] * num_time_steps

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

labels = ["go to POI-A", "go away from POI-A", "go to POI-B", "go away from POI-B", "go to agent", "go away from agent"]
ax.set_title("Distribution of selected rewards", fontsize = 30)

ax.set_xticklabels(["go to POI-A", "go away from POI-A", "go to POI-B", "go away from POI-B", "go to agent", "go away from agent"], fontsize = 20, rotation = 'vertical')
ax.set_xticks(range(num_rewards)) # how many ticks
#ax.set_xlabel("Reward types", labelpad = 60, fontsize = 20)

ax.set_yticklabels(np.arange(0, num_time_steps+1, 5), fontsize = 20)
ax.set_yticks(np.arange(0, num_time_steps+1, 5))
#ax.set_yticks(range(num_time_steps))
ax.set_ylabel("Time step", labelpad = 30, fontsize = 20)

ax.set_zticklabels(np.arange(0, 101, 10), fontsize = 20)
ax.set_zticks(np.arange(0, 101, 10))
ax.set_zlabel("Percentage", labelpad = 30, fontsize = 20)

ax.bar3d(xpos,
          ypos,
          zpos,
          dx,
          dy,
          dz1,
          color= colors1, alpha = 1)


plt.show()