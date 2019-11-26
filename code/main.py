import csv
import argparse
import os
import numpy as np
import torch
from plot import DataPlot
from gmm import GaussianMixture
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
    names[0] = 'aa'
    names[1] = 'A'
    names[2] = 'bb'
    names[3] = 'B'
    names[4] = 'ee'
    names[5] = 'E'
    names[6] = 'mm'
    names[7] = 'M'
    names[8] = 'D1'
    names[9] = 'D2'
    names[10] = 'D3'
    names[11] = 'D4'
    names[12] = 'D5'

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


        # segmentation test
        segment_number = 2
        model = GaussianMixture(segment_number, len(full_traj))
        tmp = list(map(list, zip(*full_traj)))
        data = torch.FloatTensor(tmp)
        model.fit(data)
        result = model.predict(data)
        # print(result)
        # print(model.mu)

        #test mu
        mu = np.array(model.mu)[0]
        # print("mu", mu)
        # plt.plot([ -i for i in full_traj[0] ], full_traj[2], "r--")
        # plt.plot( - mu[0][0], mu[0][2], "go")
        # plt.plot( - mu[1][0], mu[1][2], "b*")
        #plt.show()

        # choose what kind of data to paint
        PaintXYZ = True   # paint the xyz change according to time
        PaintRaw = False  # paint 3D raw data as scatter points
        PaintDMP = False   # paint 3D raw data and data processed by DMP as scatter points 
        PaintPaper = False  # plot DMP processed points on 2D
        PaintCenter = False  # plot trajectory and its point center calculated by gmm

        plot = DataPlot(name, full_traj, dfull_traj, ddfull_traj)
        if PaintXYZ == True:
            plot.paint_xyz()
        if PaintRaw == True:
            plot.paint_raw()
        if PaintDMP == True:
            plot.paint_dmp()
        if PaintPaper == True:
            plot.paint_paper()
