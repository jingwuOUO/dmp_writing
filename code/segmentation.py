import math
import numpy as np
import matplotlib.pyplot as plt

class Segmentation:

    def __init__(self, traj, dtraj, ddtraj):

        self.traj = traj
        self.dtraj = dtraj
        self.ddtraj = ddtraj
        self.strokes = []
        self.Len = len(traj[0])


    def segmentate_three(self):
        i = 100
        seg_order = []
        while i < self.Len:
            if abs(self.traj[1][i] - 0.03) > 0.006 and abs(self.dtraj[1][i]) > 0.1:
                if len(seg_order) == 0 or len(seg_order) == 1 or len(seg_order) == 3:
                    seg_order.append(i)
                elif (i - seg_order[-1]) < 20:
                    seg_order[-1] = i
                else:
                    seg_order.append(i)
            i += 1
        # print("seg order", seg_order)
        stroke0 = np.array(self.traj)[:, :seg_order[0]]
        # print("stroke0 size", stroke0)
        stroke1 = np.array(self.traj)[:, seg_order[1]:seg_order[2]]
        stroke3 = np.array(self.traj)[:, seg_order[3]:]
        return [list(stroke0), list(stroke1), list(stroke3)]

    def segmentate_two(self):
        i = 100
        seg_order = []
        while i < self.Len:
            if abs(self.traj[1][i] - 0.03) > 0.006 and abs(self.dtraj[1][i]) > 0.1:
                if len(seg_order) == 0 or len(seg_order) == 1:
                    seg_order.append(i)
                elif (i - seg_order[-1]) < 20:
                    seg_order[-1] = i
                else:
                    seg_order.append(i)
            i += 1
        # print("seg order", seg_order)
        stroke0 = np.array(self.traj)[:, :seg_order[0]]
        # print("stroke0 size", stroke0)
        stroke1 = np.array(self.traj)[:, seg_order[1]:]
        # print("stroke1 size", stroke1)
        # while i < self.Len:
        #     if self.traj[1][i] > 0.03 and self.traj[1][i] < 0.04 :
        #         stroke0[0].append(self.traj[0][i])
        #         stroke0[1].append(self.traj[1][i])
        #         stroke0[2].append(self.traj[2][i])
        #     else:
        #         break
        #     i += 1

        # while i < self.Len:
        #     if stroke == 0 and self.traj[1][i] > 0.035:
        #         i += 1
        #         continue
        #     elif self.traj[1][i] < 0.035:
        #         stroke1[0].append(self.traj[0][i])
        #         stroke1[1].append(self.traj[1][i])
        #         stroke1[2].append(self.traj[2][i])
        #     i += 1
        
        return [list(stroke0), list(stroke1)]