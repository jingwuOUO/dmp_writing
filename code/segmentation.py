import math
import numpy as np
import matplotlib.pyplot as plt
import torch

class Segmentation:

    def __init__(self, traj, dtraj, ddtraj):

        self.traj = traj
        self.dtraj = dtraj
        self.ddtraj = ddtraj
        self.strokes = []
        self.Len = len(traj)


    def segmentate_three(self):
        stroke0 = [[], [], []]
        stroke1 = [[], [], []]
        stroke2 = [[], [], []]
        stroke = 0
        i = 0

        while i < self.Len:
            if self.dtraj[1][i] < 0.1:
                stroke0[0].append(self.traj[0][i])
                stroke0[1].append(self.traj[1][i])
                stroke0[2].append(self.traj[2][i])
            else:
                break
            i += 1

        while i < self.Len:
            if stroke == 0 and self.dtraj[1][i] > 0.1:
                i += 1
                continue
            elif self.dtraj[1][i] < 0.1:
                stroke1[0].append(self.traj[0][i])
                stroke1[1].append(self.traj[1][i])
                stroke1[2].append(self.traj[2][i])
                stroke = 1
            elif stroke == 1 and self.dtraj[1][i] > 0.1:
                break
            i += 1

        while i < self.Len:
            if stroke == 1 and self.dtraj[1][i] > 0.1:
                i += 1
                continue    
            elif self.dtraj[1][i] < 0.1:
                stroke2[0].append(self.traj[0][i])
                stroke2[1].append(self.traj[1][i])
                stroke2[2].append(self.traj[2][i])
                stroke = 2
            elif stroke == 2 and self.dtraj[1][i] > 0.1:
                break
            i += 1

        return stroke0, stroke1, stroke2  
        # sequence0 = range(0, len(traj2[0]))
        # plt.subplot(311)
        # plt.plot(sequence0, traj2[0], color = 'r', label = 'x')
        # plt.legend(loc='upper left', frameon=False)
        # plt.subplot(312)
        # plt.plot(sequence0, traj2[1], color = 'g', label = 'y')
        # plt.legend(loc='upper left', frameon=False)
        # plt.subplot(313)
        # plt.plot(sequence0, traj2[2], color = 'b', label = 'z')
        # plt.legend(loc='upper left', frameon=False)
        # plt.show()

    def segmentate_two(self):
        stroke0 = [[], [], []]
        stroke1 = [[], [], []]
        stroke = 0
        i = 0

        while i < self.Len:
            if self.traj[1][i] < 0.1:
                stroke0[0].append(self.traj[0][i])
                stroke0[1].append(self.traj[1][i])
                stroke0[2].append(self.traj[2][i])
            else:
                break
            i += 1

        while i < self.Len:
            if stroke == 0 and self.traj[1][i] > 0.1:
                i += 1
                continue
            elif self.traj[1][i] < 0.1:
                stroke1[0].append(self.traj[0][i])
                stroke1[1].append(self.traj[1][i])
                stroke1[2].append(self.traj[2][i])
                stroke = 1
            elif stroke == 1 and self.traj[1][i] > 0.1:
                break
            i += 1
        
        return stroke0, stroke1