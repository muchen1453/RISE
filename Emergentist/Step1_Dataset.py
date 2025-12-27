import numpy as np
import math
import time
import random

def shiftDim(x, x0, L):
    if x - x0 >= L:
        return x - x0 - L
    elif x - x0 < 0:
        return x - x0 + L
    else:
        return x - x0

def readData(filename, N):
    atom = np.zeros((N, 4))
    with open(filename, 'r') as file:
        for line in file:
            if  'atom types' in line:
                break
        file.readline()
        lx = file.readline().strip().split(' ')
        x0 = float(lx[0])
        x1 = float(lx[1])
        Lx = x1 - x0
        ly = file.readline().strip().split(' ')
        y0 = float(ly[0])
        y1 = float(ly[1])
        Ly = y1 - y0
        lz = file.readline().strip().split(' ')
        z0 = float(lz[0])
        z1 = float(lz[1])
        Lz = z1 - z0
        
        for line in file:
            if line.strip() == 'Atoms # atomic':
                break
        file.readline()

        for i in range(N):
            l = file.readline().strip().split(' ')
            id = int(l[0]) - 1
            atom[id][0] = int(l[1]) - 1          # Atom Type
            x = float(l[2])          # x
            y = float(l[3])          # y
            z = float(l[4])          # z
            atom[id][1] = shiftDim(x, x0, Lx)          # x
            atom[id][2] = shiftDim(y, y0, Ly)          # y
            atom[id][3] = shiftDim(z, z0, Lz)          # z

    return (Lx, Ly, Lz, atom)

def readLabel(filename):
    label = []
    with open(filename, 'r') as file:
        for line in file:
            l = line.strip().split(' ')
            label.append(float(l[0]))
    return np.array(label)


N = 2000
Data = np.empty((0, 3, 32, 32, 32))
Labels = np.empty(0)
start_time = time.time()
with open('output.txt','w') as ofile:

    for coolrate in ['10_1a','10_1b','10_1c','11_1a','11_1b','12_1a','12_1b','12_1c','13_1a','13_1b']:
        for temp in range(500, 750, 50):
            ofile.write(f'processing {coolrate} {temp}:\n')
            for fileID in range(1, 201):
                ofile.write(f'{fileID}\n')
                ofile.flush()
                Lx, Ly, Lz, atom = readData(f'../zr44cu56/2000/Cooling{coolrate}/{temp}/Sort/ML{coolrate}_{str(temp)}_{str(fileID)}.txt', N)

                Density = np.zeros((1, 3, 32, 32, 32))
                dl = Lx/32
                for i in range(N):
                    t = int(atom[i][0])
                    cellx = math.floor(atom[i][1]/dl)
                    celly = math.floor(atom[i][2]/dl)
                    cellz = math.floor(atom[i][3]/dl)
                    Density[0][t][cellx][celly][cellz] += 1

                Data = np.concatenate((Data, Density), axis = 0)
            Labels = np.concatenate((Labels, readLabel(f'../zr44cu56/2000/Cooling{coolrate}/{temp}/Sort/ML{coolrate}_Energy.txt')), axis = 0)
    

np.save("Data.npy", Data)
np.save("Labels.npy", Labels)

# Load data and targets
data = np.load('Data.npy')
targets = np.load('Labels.npy')

# Initialize lists to hold the split data
data_train, data_valid, data_test = [], [], []
targets_train, targets_valid, targets_test = [], [], []

# Function to split indices into train, valid, test
def split_indices(indices):
    random.shuffle(indices)
    train_size = int(len(indices) * 0.6)
    valid_size = int(len(indices) * 0.2)
    
    return indices[:train_size], indices[train_size:train_size+valid_size], indices[train_size+valid_size:]

# Process each set of 10000 data points
for i in range(1):
    start_idx = i * 10000
    end_idx = start_idx + 10000
    indices = list(range(start_idx, end_idx))
    
    # Select randomly 10% of the data (100 points from each set of 1000)
    selected_indices = random.sample(indices, 10000)
    
    # Split the selected indices
    train_indices, valid_indices, test_indices = split_indices(selected_indices)
    
    # Append the splits to the respective lists
    data_train.append(data[train_indices])
    data_valid.append(data[valid_indices])
    data_test.append(data[test_indices])
    
    targets_train.append(targets[train_indices])
    targets_valid.append(targets[valid_indices])
    targets_test.append(targets[test_indices])

# Concatenate all the splits
data_train = np.concatenate(data_train, axis=0)
data_valid = np.concatenate(data_valid, axis=0)
data_test = np.concatenate(data_test, axis=0)
targets_train = np.concatenate(targets_train, axis=0)
targets_valid = np.concatenate(targets_valid, axis=0)
targets_test = np.concatenate(targets_test, axis=0)

# Save the splits into separate files
np.save('data_train.npy', data_train)
np.save('data_valid.npy', data_valid)
np.save('data_test.npy', data_test)
np.save('targets_train.npy', targets_train)
np.save('targets_valid.npy', targets_valid)
np.save('targets_test.npy', targets_test)


# Load the saved targets
targets_train = np.load('targets_train.npy')
targets_valid = np.load('targets_valid.npy')
targets_test = np.load('targets_test.npy')

# Combine all targets to calculate global min and max for normalization
all_targets = np.concatenate([targets_train, targets_valid, targets_test])
min_val = np.min(all_targets)
max_val = np.max(all_targets)

# Normalize using min-max scaling

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

targets_train_norm = normalize(targets_train, min_val, max_val)
targets_valid_norm = normalize(targets_valid, min_val, max_val)
targets_test_norm = normalize(targets_test, min_val, max_val)

# Save the normalized targets
np.save('targets_train_norm.npy', targets_train_norm)
np.save('targets_valid_norm.npy', targets_valid_norm)
np.save('targets_test_norm.npy', targets_test_norm)

print("Normalization complete. Normalized files saved.")
