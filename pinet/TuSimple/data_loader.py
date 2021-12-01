#########################################################################
##
##  Data loader source code for TuSimple dataset
##
#########################################################################


import math
import numpy as np
import cv2
import json
import random
from copy import deepcopy
from parameters import Parameters


#########################################################################
## some iamge transform utils
#########################################################################
def Translate_Points(point,translation): 
    point = point + translation 
    
    return point

def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


#########################################################################
## Data loader class
#########################################################################
class Generator(object):
    ################################################################################
    ## initialize (load data set from url)
    ################################################################################
    def __init__(self):
        self.p = Parameters()

        # load training set
        self.train_data_five = []
        self.train_data_four = []
        self.train_data_three = []
        self.train_data_two = []

        with open("/home/kym/research/autonomous_car_vision/lane_detection/code/Tusimple/fitting/dataset/five.json") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_five.append(jsonString)

        with open("/home/kym/research/autonomous_car_vision/lane_detection/code/Tusimple/fitting/dataset/four.json") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_four.append(jsonString)

        with open("/home/kym/research/autonomous_car_vision/lane_detection/code/Tusimple/fitting/dataset/three.json") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_three.append(jsonString)

        with open("/home/kym/research/autonomous_car_vision/lane_detection/code/Tusimple/fitting/dataset/two.json") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_two.append(jsonString)

        self.size_train = len(self.train_data_two) + len(self.train_data_three) + len(self.train_data_four) + len(self.train_data_five)
        print(self.size_train)

        # load test set
        self.test_data = []
        with open(self.p.test_root_url+'test_tasks_0627.json') as f:
            #with open(self.p.test_root_url+'test_label.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.test_data.append(jsonString)

        self.size_test = len(self.test_data)

    #################################################################################################################
    ## Generate data as much as batchsize and augment data (filp, translation, rotation, gaussian noise, scaling)
    #################################################################################################################
    def Generate(self, sampling_list = None): 
        cuts = [(b, min(b + self.p.batch_size, self.size_train)) for b in range(0, self.size_train, self.p.batch_size)]
        for start, end in cuts:
            # resize original image to 512*256
            self.inputs, self.target_lanes, self.target_h, self.test_image, self.data_list = self.Resize_data(start, end, sampling_list)
            
            self.actual_batchsize = self.inputs.shape[0]
            self.Flip()
            self.Translation()
            self.Rotate()
            self.Gaussian()
            self.Change_intensity()
            self.Shadow()

            yield self.inputs/255.0, self.target_lanes, self.target_h, self.test_image/255.0, self.data_list  # generate normalized image

    #################################################################################################################
    ## Generate test data
    #################################################################################################################
    def Generate_Test(self): 
        cuts = [(b, min(b + self.p.batch_size, self.size_test)) for b in range(0, self.size_test, self.p.batch_size)]
        for start, end in cuts:
            test_image, path, ratio_w, ratio_h, target_h, gt = self.Resize_data_test(start, end)

            yield test_image/255.0, target_h, ratio_w, ratio_h, path, gt

    #################################################################################################################
    ## resize original image to 512*256 and matching correspond points
    #################################################################################################################
    def Resize_data_test(self, start, end):
        inputs = []
        path = []
        target_h = []
        gt = []
        for i in range(start, end):
            data = self.test_data[i]
            temp_image = cv2.imread(self.p.test_root_url+data['raw_file'])
            ratio_w = self.p.x_size*1.0/temp_image.shape[1]
            ratio_h = self.p.y_size*1.0/temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.p.x_size,self.p.y_size))
            inputs.append( np.rollaxis(temp_image, axis=2, start=0) )
            path.append(i)
            gt.append( np.array(data['lanes']) )
            target_h.append( np.array(data['h_samples']) )

        return np.array(inputs), path, ratio_w, ratio_h, target_h, gt

    def Resize_data(self, start, end, sampling_list):
        inputs = []
        target_lanes = []
        target_h = []
        data_list = []

        # choose data from each number of lanes
        for i in range(start, end):

            choose = random.random()
            if sampling_list == None or len(sampling_list) < 10:
                if 0.75 <= choose:
                    data = random.sample(self.train_data_five, 1)[0]
                    data_list.append(data)
                elif 0.3 <= choose < 0.75:
                    data = random.sample(self.train_data_four, 1)[0]
                    data_list.append(data)
                elif 0.05 <= choose < 0.3:
                    data = random.sample(self.train_data_three, 1)[0]
                    data_list.append(data)
                elif 0.0 <= choose < 0.05:
                    data = random.sample(self.train_data_two, 1)[0]
                    data_list.append(data)
            else:
                if 0.75 <= choose:
                    data = random.sample(self.train_data_five, 1)[0]
                    data_list.append(data)
                elif 0.35 <= choose < 0.75:
                    data = random.sample(self.train_data_four, 1)[0]
                    data_list.append(data)
                elif 0.2 <= choose < 0.35:
                    data = random.sample(self.train_data_three, 1)[0]
                    data_list.append(data)
                elif 0.15 <= choose < 0.2:
                    data = random.sample(self.train_data_two, 1)[0]
                    data_list.append(data)
                else:
                    data = random.sample(sampling_list, 1)[0]
                    data_list.append(data)
                
            '''
            if sampling_list == None:
                data = random.sample(self.train_data_five, 1)[0]
                data_list.append(data)
            elif len(sampling_list) < 10:
                data = random.sample(self.train_data_five, 1)[0]
                data_list.append(data)
            else:            
                choose = random.random()
                if choose > -0.2:#0.25:
                    data = random.sample(self.train_data_five, 1)[0]
                    data_list.append(data)
                else:
                    data = random.sample(sampling_list, 1)[0]
                    data_list.append(data)
            '''
            #data = random.sample(self.train_data_five, 1)[0]
            #data_list.append(data)

            # train set image
            temp_image = cv2.imread(self.p.train_root_url+data['raw_file'])
            ratio_w = self.p.x_size*1.0/temp_image.shape[1]
            ratio_h = self.p.y_size*1.0/temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.p.x_size,self.p.y_size))
            inputs.append( np.rollaxis(temp_image, axis=2, start=0) )

            temp_lanes = []
            temp_h = []
            
            for j in data['lanes']:
                l = np.array(j)
                h = np.array(data['h_samples'])
                l, h = self.make_dense_x(l, h)
                temp_h.append( h*ratio_h )
                temp_lanes.append( l*ratio_w )
            target_lanes.append(np.array(temp_lanes))
            target_h.append(np.array(temp_h))

        #test set image
        test_index = random.randrange(0, self.size_test-1)
        test_image = cv2.imread(self.p.test_root_url+self.test_data[test_index]['raw_file'])
        test_image = cv2.resize(test_image, (self.p.x_size,self.p.y_size))
        
        return np.array(inputs), target_lanes, target_h, np.rollaxis(test_image, axis=2, start=0), data_list

    def make_dense_x(self, l, h):
        out_x = []
        out_y = []

        p_x = -1
        p_y = -1
        for x, y in zip(l, h):
            if x > 0:
                if p_x < 0:
                    p_x = x
                    p_y = y
                else:
                    out_x.append(x)
                    out_y.append(y)
                    for dense_x in range(min(p_x, x), max(p_x, x), 10):
                        out_x.append(dense_x)
                        if p_x<x:
                            out_y.append( p_y + abs(p_x - dense_x) * abs(p_y-y)/float(abs(p_x - x)) )
                        else:
                            out_y.append( p_y + abs(p_x - dense_x) * abs(p_y-y)/float(abs(p_x - x)) )
                    p_x = x
                    p_y = y

        return np.array(out_x), np.array(out_y)

    #################################################################################################################
    ## Generate random unique indices according to ratio
    #################################################################################################################
    def Random_indices(self, ratio):
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    #################################################################################################################
    ## Add Gaussian noise
    #################################################################################################################
    def Gaussian(self):
        indices = self.Random_indices(self.p.noise_ratio)
        img = np.zeros((256,512,3), np.uint8)
        m = (0,0,0) 
        s = (20,20,20)
        
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            cv2.randn(img,m,s)
            test_image = test_image + img
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            self.inputs[i] = test_image

    #################################################################################################################
    ## Change intensity
    #################################################################################################################
    def Change_intensity(self):
        indices = self.Random_indices(self.p.intensity_ratio)
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            test_image =  np.rollaxis(test_image, axis=2, start=0)

            hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(random.uniform(-60.0, 60.0))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1*value
                v[v < lim] = 0
                v[v >= lim] -= lim                
            final_hsv = cv2.merge((h, s, v))
            test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            self.inputs[i] = test_image

    #################################################################################################################
    ## Generate random shadow in random region
    #################################################################################################################
    def Shadow(self, min_alpha=0.5, max_alpha = 0.75):
        indices = self.Random_indices(self.p.shadow_ratio)
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            test_image =  np.rollaxis(test_image, axis=2, start=0)

            top_x, bottom_x = np.random.randint(0, 512, 2)
            coin = 0
            rows, cols, _ = test_image.shape
            shadow_img = test_image.copy()
            if coin == 0:
                rand = np.random.randint(2)
                vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
                if rand == 0:
                    vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
                elif rand == 1:
                    vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
                mask = test_image.copy()
                channel_count = test_image.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (0,) * channel_count
                cv2.fillPoly(mask, [vertices], ignore_mask_color)
                rand_alpha = np.random.uniform(min_alpha, max_alpha)
                cv2.addWeighted(mask, rand_alpha, test_image, 1 - rand_alpha, 0., shadow_img)
                shadow_img =  np.rollaxis(shadow_img, axis=2, start=0)
                self.inputs[i] = shadow_img

    #################################################################################################################
    ## Flip
    #################################################################################################################
    def Flip(self):
        indices = self.Random_indices(self.p.flip_ratio)
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)

            temp_image = cv2.flip(temp_image, 1)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image

            x = self.target_lanes[i]
            for j in range(len(x)):
                x[j][x[j]>0]  = self.p.x_size - x[j][x[j]>0]
                x[j][x[j]<0] = -2
                x[j][x[j]>=self.p.x_size] = -2

            self.target_lanes[i] = x

    #################################################################################################################
    ## Translation
    #################################################################################################################
    def Translation(self):
        indices = self.Random_indices(self.p.translation_ratio)
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)       

            tx = np.random.randint(-50, 50)
            ty = np.random.randint(-30, 30)

            temp_image = cv2.warpAffine(temp_image, np.float32([[1,0,tx],[0,1,ty]]), (self.p.x_size, self.p.y_size))
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image

            x = self.target_lanes[i]
            for j in range(len(x)):
                x[j][x[j]>0]  = x[j][x[j]>0] + tx
                x[j][x[j]<0] = -2
                x[j][x[j]>=self.p.x_size] = -2

            y = self.target_h[i]
            for j in range(len(y)):
                y[j][y[j]>0]  = y[j][y[j]>0] + ty
                x[j][y[j]<0] = -2
                x[j][y[j]>=self.p.y_size] = -2

            self.target_lanes[i] = x
            self.target_h[i] = y

    #################################################################################################################
    ## Rotate
    #################################################################################################################
    def Rotate(self):
        indices = self.Random_indices(self.p.rotate_ratio)
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)  

            angle = np.random.randint(-10, 10)

            M = cv2.getRotationMatrix2D((self.p.x_size//2,self.p.y_size//2),angle,1)

            temp_image = cv2.warpAffine(temp_image, M, (self.p.x_size, self.p.y_size))
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image

            x = self.target_lanes[i]
            y = self.target_h[i]

            for j in range(len(x)):
                index_mask = deepcopy(x[j]>0)
                x[j][index_mask], y[j][index_mask] = Rotate_Points((self.p.x_size//2,self.p.y_size//2),(x[j][index_mask], y[j][index_mask]),(-angle * 2 * np.pi)/360)
                x[j][x[j]<0] = -2
                x[j][x[j]>=self.p.x_size] = -2
                x[j][y[j]<0] = -2
                x[j][y[j]>=self.p.y_size] = -2

            self.target_lanes[i] = x
            self.target_h[i] = y
