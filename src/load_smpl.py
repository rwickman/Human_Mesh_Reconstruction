import os, pickle
import numpy as np
smpl_params = "../datasets/neutrMosh/neutrSMPL_CMU/"

poses = []
shapes = []
max_load = 2
cur_load = 0
# # Recursively go through all the files
for root, dirs, files in os.walk(smpl_params):
    cur_load += 1
    if cur_load > max_load:
        break
    # For each directory
    for datafile in files:
        if ".pkl" == datafile[-4:]:
            data_path = os.path.join(root, datafile)

            # Open datafile
            print(data_path)
            with open(data_path, "rb") as f:
                cur_data = pickle.load(f, encoding="latin-1")
                cur_poses = cur_data["poses"]
                cur_shapes = np.tile(cur_data["betas"], (cur_poses.shape[0], 1))
                print(cur_poses.shape)
                for i in range(cur_poses.shape[0]):
                    poses.append(cur_poses[i])
                    shapes.append(cur_shapes[i])

poses = np.array(poses)
shapes = np.array(shapes)
print(poses.shape)
print(shapes.shape)
print(np.stack([poses, shapes]))