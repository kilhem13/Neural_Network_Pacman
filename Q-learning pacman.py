import numpy as np
from random import randint
import random
import os
import matplotlib.pyplot as plt

class EnvGrid(object):

    def __init__(self):
        super(EnvGrid, self).__init__()
        self.grid = [
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ]

        self.y = 2
        self.x = 0

        self.actions = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ]

    def reset(self):
        self.y = 2
        self.x = 0
        return self.y*2+self.x+1

    def step(self, action):

        self.y = max(0, min(self.y + self.actions[action][0], 2))
        self.x = max(0, min(self.x + self.actions[action][1], 2))
        return (self.y*3+self.x+1), self.grid[self.y][self.x]

    def show(self):
        print("------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                if abs(pt) != 1:
                    print("%s\t" % (' |' if y != self.y or x != self.x else "X|"), end="", flush=True)
                else:
                    print("%s|\t" % pt, end="", flush=True)
                x += 1
            y += 1
            print("")

    def is_finished(self):
        return self.grid[self.y][self.x] == 1

    def loss(self):
        return self.grid[self.y][self.x] == -1


def take_actions(st, Q, eps):
    if random.uniform(0,1) < eps:
        action = randint(0, 3)
    else:
        action = np.argmax(Q[st])
    return action

env = EnvGrid()
Q = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
]
memory = []
memory2 = []
it = 0
prev_it = 0
win = 0
for i in range(100):
    st = env.reset()
    while not env.is_finished() and not env.loss():
        # env.show()
        it += 1
        memory2.append(0)
        for i in range(1000):
            a = 0
        # at = int(input("$>"))
        at = take_actions(st, Q, 0.1)
        stp1, r = env.step(at)
        # st = stp1

        atp1 = take_actions(stp1, Q, 0.0)
        Q[st][at] = Q[st][at] + 0.1*(r + 0.9*Q[stp1][atp1] - Q[st][at])
        # os.system('clear')
        st = stp1
    if env.is_finished():
        win += 1
        print("WIN %i" % win)
        print("Nombre de déplacements: %i" % (it - prev_it))
        # memory2.
        prev_it = it
        memory.insert(10, win)
        memory2.pop()
        memory2.append(win)
np.trim_zeros(memory2)
plt.plot(memory)
plt.plot(memory2, 'r.')
plt.show()
for s in range(1, 10):
    print(s, Q[s])