Official code for "Structured Bird’s-Eye-View Traffic Scene Understanding from Onboard Images"

[Link to paper](https://arxiv.org/pdf/2110.01997.pdf)

We provide support Nuscenes and Argoverse datasets. 

## Steps
0. Make sure you have installed Nuscenes and/or Argoverse devkits and datasets installed
1. In configs/deafults.yml file, set the paths
2. Run the make_labels.py file for the dataset you want to use
3. You can use train_tr.py for training the transformer based model or train_prnn.py to train the Polygon-RNN based model
4. Validator files can be used for testing

