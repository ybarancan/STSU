'''
Using onnxruntime to run converted onnx model's inference.

'''

import numpy as np  
import onnxruntime as rt 
import os
import cv2
from copy import deepcopy

print(rt.__version__)

#set parameters.
class param:
    def __init__(self):
        self.color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),
                    (100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]

        self.x_size = 512
        self.y_size = 256
        self.resize_ratio = 8
        self.grid_x = self.x_size//self.resize_ratio #64
        self.grid_y = self.y_size//self.resize_ratio #32

        self.threshold_point = 0.81
        self.threshold_instance = 0.08

        self.grid_location = np.zeros((self.grid_y, self.grid_x, 2))
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                self.grid_location[y][x][0] = x
                self.grid_location[y][x][1] = y

p = param() 

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

# draw_points on original img.
def draw_points(x, y, image, w_ratio, h_ratio):
    color_index = 0
    for id in range(len(x)):
        color_index += 1
        if color_index > 12:
            color_index = 12  # 最多显示12种不同颜色，代表实例
        x_l = x[id]
        x_list = [int(x / w_ratio) for x in x_l]
        y_l = y[id]
        y_list = [int(y / h_ratio) for y in y_l]
        for pts in zip(x_list, y_list):
            image = cv2.circle(image, (int(pts[0]), int(pts[1])), 8, p.color[color_index], -1)  # 5
    return image

def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]

    offset = offsets[mask]

    feature = instance[mask]
   
    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y


def test(model_output, test_image, w_ratio, h_ratio ,thresh=p.threshold_point):

    confidence, offset, instance = model_output[0],model_output[1],model_output[2]
    
    out_x = []
    out_y = []
    out_image = []
    
    confidence = np.squeeze(confidence)
  
    offset = np.squeeze(offset)
    offset = np.rollaxis(offset, axis=2, start=0)
    offset = np.rollaxis(offset, axis=2, start=0)
    
    instance = np.squeeze(instance)
    instance = np.rollaxis(instance, axis=2, start=0)
    instance = np.rollaxis(instance, axis=2, start=0)
    
    # generate point and cluster
    raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

    # eliminate fewer points
    in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
            
    # sort points along y 
    in_x, in_y = sort_along_y(in_x, in_y) 

    result_image = draw_points(in_x, in_y,test_image, w_ratio, h_ratio)

    out_x.append(in_x)
    out_y.append(in_y)
        
    return out_x, out_y,  result_image

def to_np(test_image):
    test_image = np.rollaxis(test_image, axis=2, start=0)
    inputs = test_image.astype(np.float32)
    inputs = inputs[np.newaxis,:,:,:] 
    return inputs


def onnx_inference(model_path,test_images,save_test_dir):
    
    sess = rt.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    # output->9 10 11
    confidences, offsets, instances = sess.get_outputs()[9].name, sess.get_outputs()[10].name, sess.get_outputs()[11].name
    output_name = [confidences, offsets, instances]

    img_list = os.listdir(test_images)
    img_list = [img for img in img_list if '.jpg' in img]
    use_ori = True

    for img in img_list:
        print("Now Dealing With:",img)
        ori_image = cv2.imread(test_images + '/' + img) #hw, cv2.IMREAD_UNCHANGED
        test_image = cv2.resize(ori_image, (p.x_size, p.y_size)) / 255.0
        
        test_image = to_np(test_image)

        pred_onnx = sess.run(output_name, {input_name:test_image})

        w_ratio = p.x_size * 1.0 / ori_image.shape[1]
        h_ratio = p.y_size* 1.0 / ori_image.shape[0]

        _, _, ti = test(pred_onnx, ori_image, w_ratio, h_ratio ,thresh=p.threshold_point)
        cv2.imwrite(save_test_dir + '/' + "{}_tested.jpg".format(img.split('.jpg')[0]), ti)

if __name__ == '__main__':

    model_dir = './onnx_models/'
    model_path = model_dir + 'pinet_v2.onnx'

    test_images = './test_curves'
    save_test_dir = './test_onnx_result'
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    print("======= MODEL INFERENCE =======")

    onnx_inference(model_path, test_images, save_test_dir)
    print("finished~~")


