import pandas as pd
import numpy as np

np.random.seed(42)

arr = np.array(pd.read_csv("./H3_Multiclass_Hate_Speech_Detection_train_preprocessed.csv"))
np.random.shuffle(arr)
# print(arr)

hate = []
normal = []
offensive = []

for i in range(arr.shape[0]):
    if(arr[i,0] == 0):
        hate.append([arr[i,0], arr[i,1]])

print(len(hate))

for j in range(arr.shape[0]):
    if(arr[j,0] == 1 and len(offensive) < len(hate)):
        offensive.append([arr[j,0], arr[j,1]])
    elif (arr[j,0] == 2 and len(normal) < len(hate)):
        normal.append([arr[j,0], arr[j,1]])   

hate = np.array(hate)
normal = np.array(normal)
offensive = np.array(offensive)

final_arr = np.concatenate((hate, normal, offensive))

np.random.shuffle(final_arr)

print(final_arr)
print(final_arr.shape)

pd.DataFrame(final_arr).to_csv('downsampled_data_equal.csv')



