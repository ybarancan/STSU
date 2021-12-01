Official code for "Structured Bird’s-Eye-View Traffic Scene Understanding from Onboard Images"


[Link to paper](https://arxiv.org/pdf/2110.01997.pdf)

We provide support for Nuscenes and Argoverse datasets. 

## Steps
0. Make sure you have installed Nuscenes and/or Argoverse devkits and datasets installed
1. In configs/deafults.yml file, set the paths
2. Run the make_labels.py file for the dataset you want to use
3. If you want to use zoom augmentation (only for Nuscenes currently), run src/data/nuscenes/sampling_grid_maker.py (Set the path to save the .npy file in the sampling_grid_maker.py)
4. You can use train_tr.py for training the transformer based model or train_prnn.py to train the Polygon-RNN based model
5. Validator files can be used for testing

