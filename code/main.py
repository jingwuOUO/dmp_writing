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
    # print(file_path)

    # all name of the plots
    # names = ['aa', 'A', 'bb', 'B', 'ee', 'E', 'mm', 'M', 'D1', 'D2', 'D3', 'D4', 'D5', 'D_JW', 'aa_JW', 'D_QL', 'aa_QL', 'D_AK', 'aa_AK', 'D_FR', 'aa_FR', 'D_GC', 'aa_GC',\
    #         'D_JCL', 'aa_JCL', 'D_LAURA', 'aa_LAURA', 'D_Leng', 'aa_Leng', 'D_Mandy', 'aa_Mandy', 'D_MEC', 'aa_MEC', 'D_SCM', 'aa_SCM', 'D_XHP', 'aa_XHP', 'D_YFC', 'aa_YFC',\
    #         'D_YTZ', 'aa_YTZ', 'D_YY', 'aa_YY']
    names = ['aa_JW', 'D_JW']

    # Parse CSV file
    for name in names:
        print(name)
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
        print("----------------------------------Finished reading data-------------------------------------------")

        # choose what kind of data to paint
        PaintXYZ = False   # paint the xyz change according to time
        PaintRaw = False  # paint 3D raw data as scatter points
        PaintDMP = True  # paint 3D raw data and data processed by DMP as scatter points 
        PaintPaper = False   # plot DMP processed points on 2D
        PaintSegment = False  # plot strokes on 2D
        PaintSeg_XZ = False  # segment D and plot stroke2 of D in x according to time
        Creat_letter = False  # segment D and sue it to create P/B/D

        plot = DataPlot(name, full_traj, dfull_traj, ddfull_traj)
        # if PaintXYZ == True:
        #     plot.paint_xyz()
        # if PaintRaw == True:
        #     plot.paint_raw()
        # if PaintDMP == True:
        #     plot.paint_dmp()
        # if PaintPaper == True:
        #     plot.paint_paper()
        # if PaintSegment == True:
        #     plot.paint_segment()
        # if PaintSeg_XZ == True:
        #     plot.paint_seg_xz()
        # if Creat_letter == True:
        #     plot.create_letter()


        # plot.paint_paper_kernel_number(10, 310, 10)
        # plot.paint_paper()
        plot.paint_seg_xz()