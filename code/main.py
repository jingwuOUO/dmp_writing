import csv
import argparse
import os
import numpy as np
import torch
from plot import DataPlot
from gmm import GaussianMixture
from segmentation import Segmentation
import matplotlib.pyplot as plt

def read_from_csv(filename, name):
    with open(filename) as csv_file:
        # Skip first two rows
        next(csv_file)
        next(csv_file)
        # Create a CSV reader
        reader = csv.reader(csv_file, delimiter=',')
        # Get info
        types = np.array(next(reader))
        names = np.array(next(reader))
        hex_id = np.array(next(reader))  # hex IDs
        quantities = np.array(next(reader))
        coords = np.array(next(reader))
        # Extract names and find columns that need to be extracted
        indices = names == name
        if not np.any(indices):
            raise RuntimeError("Object not found in csv file")
        info = np.array(list(zip(types[indices], names[indices], hex_id[indices],
                                 quantities[indices], coords[indices])))
        def extract_floats(row):
           return [(float(s) if s else 0) for s in row[indices]]
        values = np.array([extract_floats(np.array(row)) for row in reader])
        return info, values

def get_d_xyz(x, len, freq):
    dx = np.zeros(len)
    dx[0] = (x[1] - x[0])*freq
    dx[len-1] = (x[len-1] - x[len-2])*freq
    for i in range(len - 2):
        dx[i+1] = (x[i+2] - x[i])*freq/2                  
    dx = dx.tolist()
    return dx


if __name__ == "__main__":

    # parse all datasets in the path
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    file_path = []
    domain = os.path.abspath(args.path)
    for info in os.listdir(args.path):
        file_path.append(os.path.join(domain, info))
    file_path.sort()
    print(file_path)

    # all name of the plots
    names = {}
    # names[0] = 'aa'
    # names[1] = 'A'
    # names[2] = 'bb'
    # names[3] = 'B'
    # names[4] = 'ee'
    # names[5] = 'E'
    # names[6] = 'mm'
    # names[7] = 'M'
    # names[8] = 'D1'
    # names[9] = 'D2'
    # names[10] = 'D3'
    # names[11] = 'D4'
    # names[12] = 'D5'
    names[13] = 'D_JW'
    names[14] = 'aa_JW'
    names[15] = 'D_QL'
    names[16] = 'aa_QL'
    names[17] = 'D_AK'
    names[18] = 'aa_AK'
    names[19] = 'D_FR'
    names[20] = 'aa_FR'
    names[21] = 'D_GC'
    names[22] = 'aa_GC'
    names[23] = 'D_JCL'
    names[24] = 'aa_JCL'
    names[25] = 'D_LAURA'
    names[26] = 'aa_LAURA'
    names[27] = 'D_Leng'
    names[28] = 'aa_Leng'
    names[29] = 'D_Mandy'
    names[30] = 'aa_Mandy'
    names[31] = 'D_MEC'
    names[32] = 'aa_MEC' 
    names[33] = 'D_SCM'
    names[34] = 'aa_SCM'
    names[35] = 'D_XHP'
    names[36] = 'aa_XHP'
    names[37] = 'D_YFC'
    names[38] = 'aa_YFC'
    names[39] = 'D_YTZ'
    names[40] = 'aa_YTZ'
    names[41] = 'D_YY'
    names[42] = 'aa_YY'


    # Parse CSV file
    for key, name in names.items():
        print(key, name)
        # find  according file to process
        filename = None
        for path in file_path:
            if name in path:
                filename = path
        if filename == None:
            print("The file %s is not in the path" % name)
            continue

        tracker_name = "Unlabeled:5000"
        info, values = read_from_csv(filename, tracker_name)
        Len = len(values[:,0])

        for i in range(Len-1):
            for j in range(3):
                if values[i+1, j]==0 and values[i, j]!=0 :
                   values[i+1, j]=values[i, j]
        freq = 120

        # put the trajectory read from csv into fulltraj, fulltraj[0][1][2] means fulltraj[x][y][z]
        degree = 0
        full_traj = [[], [], []]
        dfull_traj = [[], [], []]
        ddfull_traj = [[], [], []]
        while degree < 3:
            temp = values[:, degree]
            temp = temp.tolist()
            full_traj[degree] = temp
            dfull_traj[degree] = get_d_xyz(full_traj[degree], len(full_traj[degree]), freq)
            ddfull_traj[degree] = get_d_xyz(dfull_traj[degree], len(dfull_traj[degree]), freq)
            degree += 1


        # segment using heuristic functiion H = dx^2 + ddx^2
        # from mpl_toolkits import mplot3d
        # order = range(len(full_traj[0]))
        # points = list(map(list, zip(*full_traj)))
        # v_points = list(map(list, zip(*dfull_traj)))
        # a_points = list(map(list, zip(*ddfull_traj)))
        # seg_order = np.array([points[i] if points[i][1] < 0.035 else [0, 0, 0] for i in order])
        # ax = plt.subplot(projection='3d')
        # ax.scatter(seg_order[:, 0], seg_order[:, 1], seg_order[:, 2], c='r')
        # ax.scatter(full_traj[0], full_traj[1], full_traj[2], c='b')
        # plt.show()

        # segmentation test
        # segment_number = 2
        # model = GaussianMixture(segment_number, 3)
        # # tmp = list(map(list, zip(*full_traj)))
        # # tmp = [[i] for i in full_traj[1]]
        # # tmp  = [[1,1,1], [1,1,1], [0.999, 0.999, 0.999], [0.005, 0.007, 0.009], [0,0,0], [0.5, 0.5, 0.5]]
        # tmp = seg_order
        # data = torch.FloatTensor(tmp)
        # model.fit(data)
        # result = model.predict(data)
        # print(result)
        # print(model.mu)
        # #test mu
        # mu = np.array(model.mu)[0]
        # print("mu", mu)
        # plt.plot([ -i for i in full_traj[0] ], full_traj[2], "r")
        # plt.plot( - mu[0][0], mu[0][2], "g*")
        # plt.plot( - mu[1][0], mu[1][2], "b*")
        # # plt.plot( - mu[2][0], mu[2][2], "y*")
        # # plt.plot( - mu[3][0], mu[3][2], "c*")
        # plt.show()

        
        # choose what kind of data to paint
        PaintXYZ = False   # paint the xyz change according to time
        PaintRaw = False  # paint 3D raw data as scatter points
        PaintDMP = False   # paint 3D raw data and data processed by DMP as scatter points 
        PaintPaper = False  # plot DMP processed points on 2D
        PaintSegment = True  # plot strokes on 2D

        plot = DataPlot(name, full_traj, dfull_traj, ddfull_traj)
        if PaintXYZ == True:
            plot.paint_xyz()
        if PaintRaw == True:
            plot.paint_raw()
        if PaintDMP == True:
            plot.paint_dmp()
        if PaintPaper == True:
            plot.paint_paper()
        if PaintSegment == True:
            plot.paint_segment()
