from segmentation import Segmentation
from dmp_qlwj import DynamicMP
from dmp_qlwj import Bad_DynamicMP

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_d_xyz(x, len, freq):
    dx = np.zeros(len)
    dx[0] = (x[1] - x[0])*freq
    dx[len-1] = (x[len-1] - x[len-2])*freq
    for i in range(len - 2):
        dx[i+1] = (x[i+2] - x[i])*freq/2                  
    dx = dx.tolist()
    return dx

# input the traj, get the velocity and acceleration of the traj
def get_vel_and_acc(x, freq):
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
def dmp_process(m, traj, dtraj, ddtraj, start_error=0, goal_error=0, dmp_choose=True, shape=0, linear=0):
    '''
        imput the traj, the velocity of this traj and the acceleration , the output is the traj processed by dmp algorithm
    '''

    if dmp_choose == True:
        process_x = DynamicMP(m, traj[0], dtraj[0], ddtraj[0], len(traj[0]), start_error, goal_error, shape, linear)
        xx = process_x.getddy1()
        process_y = DynamicMP(m, traj[1], dtraj[1], ddtraj[1], len(traj[1]), start_error, goal_error, shape, linear)
        yy = process_y.getddy1()
        process_z = DynamicMP(m, traj[2], dtraj[2], ddtraj[2], len(traj[2]), start_error, goal_error, shape, linear)
        zz = process_z.getddy1()
    else:
        process_x = Bad_DynamicMP(m, traj[0], dtraj[0], ddtraj[0], len(traj[0]), start_error, goal_error, shape, linear)
        xx = process_x.getddy1()
        process_y = Bad_DynamicMP(m, traj[1], dtraj[1], ddtraj[1], len(traj[1]), start_error, goal_error, shape, linear)
        yy = process_y.getddy1()
        process_z = Bad_DynamicMP(m, traj[2], dtraj[2], ddtraj[2], len(traj[2]), start_error, goal_error, shape, linear)
        zz = process_z.getddy1()
    dmp_traj = [xx, yy, zz]
    return dmp_traj    

class DataPlot:

    def __init__(self, name, full_traj, dfull_traj=None, ddfull_traj=None):
        
        self.name = name
        self.full_traj = full_traj
        self.freq = 120
        if dfull_traj == None or ddfull_traj == None:
            self.dfull_traj, self.ddfull_traj = get_vel_and_acc(self.full_traj, self.freq)
        else:
            self.dfull_traj = dfull_traj
            self.ddfull_traj = ddfull_traj
        

    def paint_xyz(self):
        n = range(0, len(self.full_traj[0]))
        time = [1/ self.freq * i for i in n]

        # x, y, z
        plt.figure(1)
        plt.subplot(311)
        plt.title(" The X Z Y coordinate change according to Time")
        plt.plot(time, self.full_traj[0])
        plt.ylabel("X / m")

        plt.subplot(312)
        plt.plot(time, self.full_traj[2])
        plt.ylabel("Z / m")

        plt.subplot(313)
        plt.plot(time, self.full_traj[1])
        plt.ylim(0, 0.2)
        plt.ylabel("Y / m")

        plt.xlabel(" Time (s)")
        # plt.interactive(False)
        plt.show()

        # dx, dy, dz
        plt.figure(2)
        plt.subplot(311)
        plt.title(" The dX dZ dY coordinate change according to Time")
        plt.plot(time, self.dfull_traj[0])
        plt.ylabel("dX / m")

        plt.subplot(312)
        plt.plot(time, self.dfull_traj[2])
        plt.ylabel("dZ / m")

        plt.subplot(313)
        plt.plot(time, self.dfull_traj[1])
        # plt.ylim(0, 0.2)
        plt.ylabel("dY / m")

        plt.xlabel(" Time (s)")
        # plt.interactive(False)
        plt.show()


        # ddx, ddy, ddz
        plt.figure(3)
        plt.subplot(311)
        plt.title(" The dX dZ dY coordinate change according to Time")
        plt.plot(time, self.ddfull_traj[0])
        plt.ylabel("ddX / m")

        plt.subplot(312)
        plt.plot(time, self.ddfull_traj[2])
        plt.ylabel("ddZ / m")

        plt.subplot(313)
        plt.plot(time, self.ddfull_traj[1])
        # plt.ylim(0, 0.2)
        plt.ylabel("ddY / m")

        plt.xlabel(" Time (s)")
        plt.show()

    def paint_raw(self):
        ax = plt.subplot(projection='3d')
        ax.scatter(self.full_traj[0], self.full_traj[1], self.full_traj[2], c='r')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        # ax.axis('equal')
        plt.show()
        # plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/%s_raw.png' % self.name)

    def paint_dmp(self):
        ax = plt.subplot(projection='3d')
        dmp_traj = dmp_process(100, self.full_traj, self.dfull_traj, self.ddfull_traj)
        ax.scatter(self.full_traj[0], self.full_traj[1], self.full_traj[2], c='r')
        ax.scatter(dmp_traj[0], dmp_traj[1], dmp_traj[2], c='g')
        plt.show()
        # plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/%s_dmp.png' % self.name)

    def paint_paper(self):
        number = np.arange(0, len(self.full_traj[0])).tolist()
        goal_error = 0.0
        dmp_kind = False
        # dmp100 = dmp_process(20, self.full_traj, self.dfull_traj, self.ddfull_traj)
        # dmp300 = dmp_process(50, self.full_traj, self.dfull_traj, self.ddfull_traj)
        # dmp600 = dmp_process(100, self.full_traj, self.dfull_traj, self.ddfull_traj)
        dmp600 = dmp_process(200, self.full_traj, self.dfull_traj, self.ddfull_traj, 0.0, goal_error, dmp_kind)
        # error1 = [math.sqrt((dmp100[0][i] - self.full_traj[0][i])**2 + (dmp100[2][i] - self.full_traj[2][i])**2) for i in range(len(self.full_traj[0]))]
        # error2 = [math.sqrt((dmp300[0][i] - self.full_traj[0][i])**2 + (dmp300[2][i] - self.full_traj[2][i])**2) for i in range(len(self.full_traj[0]))]
        error3 = [math.sqrt((dmp600[0][i] - self.full_traj[0][i])**2 + (dmp600[2][i] - self.full_traj[2][i])**2) for i in range(len(self.full_traj[0]))]

        if dmp_kind == False:
            name1 =  self.name + str(goal_error) + 'a.png'
            name2 =  self.name + str(goal_error) + 'a_Euclidean_distance.png'
        else:
            name1 = 'good_' + self.name + str(goal_error) + 'a.png'
            name2 = 'good_' + self.name + str(goal_error) + 'a_Euclidean_distance.png'
        
        plt.figure(1)
        plt.plot([ -i for i in self.full_traj[0]], self.full_traj[2], 'r', label='original trajectory', linewidth=5, alpha=0.6)
        # plt.plot([ -i for i in dmp100[0]], dmp100[2], 'g', label='DMP kernel = 20')
        # plt.plot([ -i for i in dmp300[0]], dmp300[2], 'chocolate', label='DMP kernel = 50' )
        plt.plot([ -i for i in dmp600[0]], dmp600[2], 'b', label='DMP kernel = 200' )
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        plt.title('Comparision between trajectory computed by different kernel number')
        plt.legend(loc='upper right', frameon=False)            
        plt.interactive(False)
        plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/dmp_compare/%s' % name1)            
        plt.show()
        
        plt.figure(2)
        # plt.plot(number, error1, 'g', label='DMP kernel = 20')
        # plt.plot(number, error2, 'chocolate', label='DMP kernel = 50' )
        plt.plot(number, error3, 'b', label='DMP kernel = 200' )
        plt.xlabel('N')
        plt.ylabel('Euclidean distance/m')
        plt.title('Euclidean distance of original trajectory and trajectory computed by DMP')
        plt.legend(loc='upper right', frameon=False)
        plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/dmp_compare/%s' % name2)
        plt.show()

    def paint_segment(self):
        seg = Segmentation(self.full_traj, self.dfull_traj, self.ddfull_traj)
        strokes = seg.segmentate_two()
        for stroke in strokes:
            vel, acc = get_vel_and_acc(stroke, self.freq)
            dmp_stroke = dmp_process(100, stroke, vel, acc)
            plt.plot([ -i for i in stroke[0]], stroke[2], label='stroke %d' % strokes.index(stroke), linewidth=5, alpha=0.6)
            plt.plot([ -i for i in dmp_stroke[0]], stroke[2], label='DMP_stroke %d' % strokes.index(stroke))
        plt.xticks(np.arange(0.45, 0.65, step=0.05))
        plt.yticks(np.arange(0.25, 0.40, step=0.05))
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        plt.title('Segmentated trajectory and trajectory computed by DMP of writing belongs to %s' % self.name[2:])
        plt.legend(loc='upper right', frameon=False)
        plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/segment/%s_segment.png' % self.name)
        plt.show()

    def paint_seg_xz(self):
        seg = Segmentation(self.full_traj, self.dfull_traj, self.ddfull_traj)
        stroke2 = seg.segmentate_two()[1]
        vel, acc = get_vel_and_acc(stroke2, self.freq)

        goal_error = 0.1
        dmp_kind = True

        dmp_stroke = dmp_process(200, stroke2, vel, acc, 0, goal_error, dmp_kind)
        if dmp_kind == False:
            name1 =  str(goal_error) + '_D_xy_segment.png'
            name2 =  str(goal_error) + '_D_xy_absolute_distance.png'
        else:
            name1 = 'good_' + str(goal_error) + '_D_xy_segment.png'
            name2 = 'good_' + str(goal_error) + '_D_xy_absolute_distance.png'
        
        plt.figure(1)
        plt.plot((1/self.freq)*np.arange(len(stroke2[0])), [-i for i in stroke2[0]], label='original x coordinate', linewidth=5, alpha=0.6)
        plt.plot((1/self.freq)*np.arange(len(stroke2[0])), [-i for i in dmp_stroke[0]], label='dmp x coordinate')
        plt.xlabel('t/sec')
        plt.ylabel('x/m')
        plt.title('x/m coordinate changing with time/sec')
        plt.legend(loc='upper right', frameon=False)
        plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/dmp_compare/%s'% name1)
        plt.interactive(False)
        plt.show()

        plt.figure(2)
        error = [abs(dmp_stroke[0][i] - stroke2[0][i]) for i in range(len(stroke2[0]))]
        plt.plot(range(len(stroke2[0])), error, 'b')
        plt.xlabel('N')
        plt.ylabel('Absolute distance/m')
        plt.title('Absolute distance of original trajectory and trajectory computed by DMP')
        plt.legend(loc='upper right', frameon=False)
        plt.savefig('/home/jingwu/Desktop/CS8803/Project/Dec/dmp_compare/%s' % name2)
        plt.show()