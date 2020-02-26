import os
import numpy as np

def generate_data(dmp_traj, name):
    '''
    Use DMP to generate the trajectory file that can be used by the Fetch Robot
    @dmp_traj a 3xN list contains the x,y,z position of the trajectory
    @name the name of the trajectory
    '''
    dmp_traj = np.array(dmp_traj).reshape(3, -1)
    # np.savetxt('%s.txt' % name, (dmp_traj[0, :], dmp_traj[1, :], dmp_traj[2, :]), delimiter='\'')
    with open('%s.txt' % name, 'w') as f:
        for i in range(dmp_traj.shape[1]):
            x, y, z = 10 * (dmp_traj[:, i] +5.0)
            line = '[{0}, {1}, {2}]'.format(x, y, z)
            f.write(line+"\'"+"\n")

    print("----------------------Finished saving trajectory file---------------------------------------")