# Investigating Attention Mechanism in 3D Point Cloud Object Detection (arXiv 2021)

This repository is for the following paper:
"Investigating Attention Mechanism in 3D Point Cloud Object Detection"  
[Shi Qiu](https://shiqiu0419.github.io/)\*, Yunfan Wu\*, [Saeed Anwar](https://saeed-anwar.github.io/), [Chongyi Li](https://li-chongyi.github.io/)

## Abstract
This project investigates the effects of five classical 2D attention modules (**Non-local, Criss-cross, Squeeze-Excitation, CBAM, Dual-attention**) and five novel 3D attention modules (**Attentional-ShapeContextNet, Point-Attention, Channle Affinity Attention, Offset-Attention, Point-Transformer**) in 3D point cloud object detection, based on VoteNet pipeline.
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig3.png">
</p>  

## Our Attentional Backbone
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig4.png">
</p>  

## Results
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig1.png">
</p> 

## Visualization
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/attentions_in_3D_detection/blob/main/fig2.png">
</p> 

## Settings
Set up the [VoteNet](https://github.com/facebookresearch/votenet) project, and replace the ```models/backbone_module.py``` file with ours.

## Trained-models
The trained models can be downloaded from [google drive](https://drive.google.com/file/d/1ZMrZqQEp1or-ZlUU3I_kQ61bETI3pQXr/view?usp=sharing).  
The detailed evaluation logs reported in the paper can be found at [google drive](https://drive.google.com/file/d/1MrhJt6CwoD7clJIlAze_eaB2AeFiCKxK/view?usp=sharing).

## Paper and citation
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/2103.07074).  
If you find our paper/codes/results are useful, please cite:

    @article{qiu2021investigating,
      title={Investigating Attention Mechanism in 3D Point Cloud Object Detection},
      author={Qiu, Shi and Wu, Yunfan and Anwar, Saeed and Li, Chongyi},
      journal={arXiv preprint arXiv:2103.07074},
      year={2021}
    }

## Acknowledgment
The code is built on [VoteNet](https://github.com/facebookresearch/votenet). We thank the authors for sharing the codes.
