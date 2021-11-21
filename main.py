import cv2
import onnxruntime
import numpy as np
import time
from utils import prep,non_max_suppression_fast,scale_coords

cap =  cv2.VideoCapture(0)
w = "model/yolov5s.onnx"
session = onnxruntime.InferenceSession(w,   providers=["CUDAExecutionProvider"])

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

while 1:
    wl = time.time()
    _,g = cap.read()

    if _:
        g1 = g.copy()
        g1 = prep(g1)
        g2 = []
        for i in range(1):
            g2.append(g1)
        g2 = np.array(g2)
        g2 = g2.astype('float32')
        g2 = g2 / 255.0 

        pred = np.array(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: g2}))

        nc = pred.shape[2] - 5
        
        xc = pred[..., 4] > 0.25
       
        
        for xi, x in enumerate(pred):
            x = x[xc[xi]] 
            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4]).astype(int)
            bbox = non_max_suppression_fast(box,0.5)
            bbox = scale_coords(g2.shape[2:], bbox, g.shape).round()
            conf = x[..., 4]
            
            for box in bbox:
                c1,c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                g = cv2.rectangle(g,c1,c2,(255,0,255),1)
        fps = round(1 / (time.time()-wl),2)
        g = cv2.putText(g,f"FPS = {fps}",(20,30),1,2,(120,50,255),2)

        cv2.imshow("a",g)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
cap.release()