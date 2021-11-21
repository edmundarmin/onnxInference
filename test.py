import cv2
import onnxruntime
import numpy as np
import time
from utils import prep,non_max_suppression_fast,scale_coords

w = "kmodel/yolox_s.onnx"
session = onnxruntime.InferenceSession(w, providers=["CUDAExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

for i in session.get_inputs():
    print(i)


m = time.time()
g = cv2.imread("images/coba.jpg")

d  = g.copy()
d = prep(d)
data = []
for i in range(1):
    data.append(d)
data= np.array(data)
data = data.astype('float32')
data = data / 255.0 

result = session.run([output_name], {input_name: data})
#prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
print(1/(time.time()-m))