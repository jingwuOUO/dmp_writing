from segmentation import Segmentation
from dmp_qlwj import DynamicMP

import numpy as np
import math
import matplotlib.pyplot as plt

def get_d_xyz(x, len, freq):
    dx = np.zeros(len)
    dx[0] = (x[1] - x[0])*freq
    dx[len-1] = (x[len-1] - x[len-2])*freq
    for i in range(len - 2):
        dx[i+1] = (x[i+2] - x[i])*freq/2                  
    dx = dx.tolist()
    return dx

# input the traj, get the velocity and acceleration of the traj
def get_vel_and_pos(x, freq):
    dx = get_d_xyz(x[0], len(x[0]), freq)
    ddx = get_d_xyz(dx, len(dx), freq)
    dy = get_d_xyz(x[1], len(x[1]), freq)
    ddy = get_d_xyz(dy, len(dy), freq)
    dz = get_d_xyz(x[2], len(x[2]), freq)
    ddz = get_d_xyz(dz, len(dz), freq)
    vel = [dx, dy, dz]
    acc = [ddx, ddy, ddz]
    return vel, acc

# imput the traj, the velocity of this traj and the acceleration , the output is the traj processed by dmp algorithm
def dmp_process(m, traj, dtraj, ddtraj, shape=0, linear=0):
    process_x = DynamicMP(m, traj[0], dtraj[0], ddtraj[0], len(traj[0]), shape, linear)
    xx = process_x.getddy1()
    process_y = DynamicMP(m, traj[1], dtraj[1], ddtraj[1], len(traj[1]), shape, linear)
    yy = process_y.getddy1()
    process_z = DynamicMP(m, traj[2], dtraj[2], ddtraj[2], len(traj[2]), shape, linear)
    zz = process_z.getddy1()
    dmp_traj = [xx, yy, zz]
    return dmp_traj    

class DataPlot:

    def __init__(self, name, PaintRaw, PaintDMP, Segment, PaintPaper, full_traj, dfull_traj, ddfull_traj):
        
        self.name = name
        self.full_traj = full_traj
        self.dfull_traj = dfull_traj
        self.ddfull_traj = ddfull_traj
        self.freq = 120

        self.PaintRaw = PaintRaw
        self.PaintDMP = PaintDMP
        self.Segment = Segment
        self.PaintPaper = PaintPaper

    def start_plot(self):
        if self.PaintRaw == True:
            ax = plt.subplot(projection='3d')
            ax.scatter(self.full_traj[0], self.full_traj[1], self.full_traj[2], c='r')
            ax.set_zlabel('Z')
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            # ax.axis('equal')
            plt.show()
            # plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/%s_raw.png' % self.name)
        if self.PaintDMP == True:
            ax = plt.subplot(projection='3d')
            dmp_traj = dmp_process(100, self.full_traj, self.dfull_traj, self.ddfull_traj)
            ax.scatter(self.full_traj[0], self.full_traj[1], self.full_traj[2], c='r')
            ax.scatter(dmp_traj[0], dmp_traj[1], dmp_traj[2], c='g')
            plt.show()
            # plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/%s_dmp.png' % self.name)
        if self.Segment == True:
            stroke0, stroke1, stroke2 = Segmentation(self.full_traj, self.dfull_traj, self.ddfull_traj).segmentate_three()
            print(stroke2)
            dstroke0, ddstroke0 = get_vel_and_pos(stroke0, self.freq)
            dstroke1, ddstroke1 = get_vel_and_pos(stroke1, self.freq)
            dstroke2, ddstroke2 = get_vel_and_pos(stroke2, self.freq)
            dmp_stroke0 = dmp_process(100, stroke0, dstroke0, ddstroke0)
            dmp_stroke1 = dmp_process(100, stroke1, dstroke1, ddstroke1)
            dmp_stroke2 = dmp_process(100, stroke2, dstroke2, ddstroke2)
            plt.plot(stroke0[0], stroke0[2], 'r--', stroke1[0], stroke1[2], "r--", stroke2[0], stroke2[2], 'r--')
            plt.show()
        if self.PaintPaper == True:
            number = np.arange(0, len(self.full_traj[0])).tolist()
            dmp100 = dmp_process(20, self.full_traj, self.dfull_traj, self.ddfull_traj)
            dmp300 = dmp_process(50, self.full_traj, self.dfull_traj, self.ddfull_traj)
            dmp600 = dmp_process(100, self.full_traj, self.dfull_traj, self.ddfull_traj)
            # dmp1000 = dmp_process(200, full_traj, dfull_traj, ddfull_traj)
            error1 = [math.sqrt((dmp100[0][i] - self.full_traj[0][i])**2 + (dmp100[2][i] - self.full_traj[2][i])**2) for i in range(len(self.full_traj[0]))]
            error2 = [math.sqrt((dmp300[0][i] - self.full_traj[0][i])**2 + (dmp300[2][i] - self.full_traj[2][i])**2) for i in range(len(self.full_traj[0]))]
            error3 = [math.sqrt((dmp600[0][i] - self.full_traj[0][i])**2 + (dmp600[2][i] - self.full_traj[2][i])**2) for i in range(len(self.full_traj[0]))]
            
            plt.figure(1)
            plt.plot([ -i for i in self.full_traj[0]], self.full_traj[2], 'r', label='original trajectory', linewidth=5, alpha=0.6)
            plt.plot([ -i for i in dmp100[0]], dmp100[2], 'g', label='DMP kernel = 20')
            plt.plot([ -i for i in dmp300[0]], dmp300[2], 'chocolate', label='DMP kernel = 50' )
            plt.plot([ -i for i in dmp600[0]], dmp600[2], 'b', label='DMP kernel = 100' )
            plt.xlabel('X/m')
            plt.ylabel('Y/m')
            plt.title('Comparision between trajectory computed by different kernel number')
            plt.legend(loc='upper right', frameon=False)            
            plt.interactive(False)
            plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/%s_dmp_compare.png' % self.name)            
            plt.show()

            
            plt.figure(2)
            plt.plot(number, error1, 'g', label='DMP kernel = 20')
            plt.plot(number, error2, 'chocolate', label='DMP kernel = 50' )
            plt.plot(number, error3, 'b', label='DMP kernel = 100' )
            plt.xlabel('N')
            plt.ylabel('Euclidean distance/m')
            plt.title('Euclidean distance of original trajectory and trajectory computed by DMP')
            plt.legend(loc='upper right', frameon=False)
            plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/%s_euclidean_dmp_compare.png' % self.name)
            plt.show()