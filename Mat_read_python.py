import scipy.io as sio
import cv2
import numpy as np

# 5Hz

test = sio.loadmat('5Hz.mat')
print(test.items())
print(test["data_array"].shape)

test_image = test["data_array"]
B_scan = test_image[0:256,:]
cv2.imwrite("test_Bscan.png", B_scan)


# Save every B-scan
i = 0
d = 0
data_mat = np.zeros((200, 1000, 256)) 
while (i < 51200):
    ith_B_scan = test_image[i:i+256,:]
    filename = "./B_scans/img_%d.png"%d
    cv2.imwrite(filename, ith_B_scan)
    i += 256
    d = i/256
    data_mat[int(d)-1,:,:] = ith_B_scan.T
    print(d)

map_img = 255 - np.asarray(np.max(data_mat, axis=2))/255
print(map_img)
cv2.imwrite("test_map.png", map_img)