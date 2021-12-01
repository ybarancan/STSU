# lanefinder

Transformer with Cityscapes pretrained Deeplabv3 backbone.

There are currently 3 losses: 
1) Binary cross entropy for accepting or rejecting a lane;
2) L1 loss on control points
3) Focal loss on the joint line topography estimation

All the parameters can be changed from the args in the train.py

TODO: How to deal with occluded/non-visible parts of the road? Currently I am cropping roads to visible region and calculate the control points. But I am not sure if it is optimal. The other option is deleteing lines that are more than a threshold non-visible.




