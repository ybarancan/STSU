## Update
2020-11-06: New trained model for TuSimple ("0_tensor(0.5242)_lane_detection_network") (Acc: 96.81%, FP: 0.0387, FN: 0.0245, threshold = 0.36)

# key points estimation and point instance segmentation approach for lane detection

- Yeongmin Ko, Younkwan Lee, Shoaib Azam, Farzeen Munir, Moongu Jeon, Witold Pedrycz
- Abstract: Perception techniques for autonomous driving should be adaptive to various environments. In the case of traffic line detection, an essential perception module, many condition should be considered, such as number of traffic lines and computing power of the target system. To address these problems, in this paper, we propose a traffic line detection method called Point Instance Network (PINet); the method is based on the key points estimation and instance segmentation approach. The PINet includes several stacked hourglass networks that are trained simultaneously. Therefore the size of the trained models can be chosen according to the computing power of the target environment. We cast a clustering problem of the predicted key points as an instance segmentation problem; the PINet can be trained regardless of the number of the traffic lines. The PINet achieves competitive accuracy and false positive on the TuSimple and Culane datasets, popular public datasets for lane detection.
- link: https://arxiv.org/abs/2002.06604

## Dependency
- python ( We tested on python3, python2 is also work, but spline fitting is supported only on python3 )
- pytorch ( We tested on pytorch 1.1.0 with GPU(RTX2080ti))
- opencv
- numpy
- visdom (for visualization)
- sklearn (for evaluation)
- ujon (for evaluation)
- csaps (for spline fitting)

## Dataset (TuSimple)
You can download the dataset from https://github.com/TuSimple/tusimple-benchmark/issues/3. We recommand to make below structure.

    dataset
      |
      |----train_set/               # training root 
      |------|
      |------|----clips/            # video clips, 3626 clips
      |------|------|
      |------|------|----some_clip/
      |------|------|----...
      |
      |------|----label_data_0313.json      # Label data for lanes
      |------|----label_data_0531.json      # Label data for lanes
      |------|----label_data_0601.json      # Label data for lanes
      |
      |----test_set/               # testing root 
      |------|
      |------|----clips/
      |------|------|
      |------|------|----some_clip/
      |------|------|----...
      |
      |------|----test_label.json           # Test Submission Template
      |------|----test_tasks_0627.json      # Test Submission Template
            
Next, you need to change "train_root_url" and "test_root_url" to your "train_set" and "test_set" directory path in "parameters.py". For example,

```
# In "parameters.py"
line 54 : train_root_url="<tuSimple_dataset_path>/train_set/"
line 55 : test_root_url="<tuSimple_dataset_path>/test_set/"
```

Finally, you can run "fix_dataset.py", and it will generate dataset according to the number of lanes and save dataset in "dataset" directory. (We have uploaded dataset. You can use them.)

## Dataset (CULane)
You can download the dataset from https://xingangpan.github.io/projects/CULane.html.

If you download the dataset from the link, you can find some files and we recommand to make below structure.

    dataset
      |
      |----train_set/               # training root 
      |------|
      |------|----driver_23_30frame/
      |------|----driver_161_90frame/
      |------|----driver_182_30frame/
      |
      |----test_set/               # testing root 
      |------|
      |------|----driver_37_30frame/
      |------|----driver_100_30frame/
      |------|----driver_193_90frame/
      |
      |----list/               # testing root 
      |------|
      |------|----test_split/
      |------|----test.txt
      |------|----train.txt
      |------|----train_gt.txt
      |------|----val.txt
      |------|----val_gt.txt


## Test
We provide trained model, and it is saved in "savefile" directory. You can run "test.py" for testing, and it has some mode like following functions 
- mode 0 : Visualize results on test set
- mode 1 : Run the model on the given video. If you want to use this mode, enter your video path at line 63 in "test.py"
- mode 2 : Run the model on the given image. If you want to use this mode, enter your image path at line 82 in "test.py"
- mode 3 : Test the model on whole test set, and save result as json file.

You can change mode at line 22 in "parameters.py".

If you want to use other trained model, just change following 2 lines.
```
# In "parameters.py"
line 13 : model_path = "<your model path>/"
# In "test.py"
line 42 : lane_agent.load_weights(<>, "tensor(<>)")
```

- TuSimple
If you run "test.py" by mode 3, it generates "test_result.json" file. You can evaluate it by running just "evaluation.py".

- CULane
The evaluation code is forked from https://github.com/harryhan618/SCNN_Pytorch. The repository ported official evaluation code and provide the extra CMakeLists.txt.

```
cd evaluation_code/
mkdir build && cd build
(remove default build directory, it is my mistake)
cmake ..
make
```
If you run "test.py" by mode 3, it generates result files in the defined path (the path is defined by test.py). The generated file can be evaluated by the following:

```
./evaluation_code/Run.sh <file_name>
```
Before running it, you should ckech path in Run.sh


## Train
If you want to train from scratch, make line 13 blank in "parameters.py", and run "train.py"
```
# In "parameters.py"
line 13 : model_path = ""
```
"train.py" will save sample result images(in "test_result/"), trained model(in "savefile/").

If you want to train from a trained model, just change following 2 lines.
```
# In "parameters.py"
line 13 : model_path = "<your model path>/"
# In "train.py"
line 54 : lane_agent.load_weights(<>, "tensor(<>)")
```

## Network Clipping 
PINet is made of several hourglass modules; these hourglass modules are train by the same loss function.

We can make ligher model without addtional training by clipping some hourglass modules.

```
# In "hourglass_network.py"
self.layer1 = hourglass_block(128, 128)
self.layer2 = hourglass_block(128, 128)
#self.layer3 = hourglass_block(128, 128)
#self.layer4 = hourglass_block(128, 128) some layers can be commentted 
```


## Result
You can find more detail results at our paper.

4 horglass modules is run about 25fps on RTX2080ti

- TuSimple (4 hourglass modules)

| Accuracy | FP   | FN   |
| -------- | ---- | ---- |
| 96.75%   |0.0310|0.0250|

- CULane (4 hourglass modules)

| Category  | F1-measure          |
| --------- | ------------------- |
| Normal    | 90.3               |
| Crowded   | 72.3               |
| HLight    | 66.3                |
| Shadow    | 68.4               |
| No line   | 49.8               |
| Arrow     | 83.7               |
| Curve     | 65.6               |
| Crossroad | 1427 （FP measure） |
| Night     | 67.7               |
| Total     | 74.4               |
