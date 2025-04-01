import numpy as np
npimg = np.load("../data/dataset01/dataset01.npy", allow_pickle=True)
data_dict = npimg.item()
print(data_dict["features"].shape)
print(data_dict["timestamps"].shape)
