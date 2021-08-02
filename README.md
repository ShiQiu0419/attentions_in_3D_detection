# Investigating Attention Mechanism in 3D Point Cloud Object Detection (arXiv 2021)

This repository is for the following paper:
"Investigating Attention Mechanism in 3D Point Cloud Object Detection"  
[Shi Qiu](https://shiqiu0419.github.io/)\*, Yunfan Wu\*, [Saeed Anwar](https://saeed-anwar.github.io/), [Chongyi Li](https://li-chongyi.github.io/)

## Abstract
This project investigates the effects of five classical 2D attention modules (**Non-local, Criss-cross, SE, CBAM, Dual-attention**) and five novel 3D attention modules (**A-SCN, Point-Attention, CAA, Offset-Attention, Point-Transformer**) in 3D point cloud object detection, based on VoteNet pipeline.
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig3.png">
</p>  

## Our Attentional Backbone
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig4.png">
</p>  

## Settings
Set up the [VoteNet](https://github.com/facebookresearch/votenet) project, and replace the ```models/backbone_module.py``` file with ours.

## Results
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig1.png">
</p>  

