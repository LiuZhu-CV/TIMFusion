
#  "A Task-guided, Implicitly-searched and Meta- initialized Deep Model for Image Fusion"




- [arXiv](https://arxiv.org/abs/2305.15862)

Risheng Liu, Zhu Liu, Jinyuan Liu, Xin Fan, Zhongxuan Luo

## Fusion Reuslts and Chinese Version

The source images and fused results on TNO, RoadScene and M3FD (4200 images) are
provided in [link](https://drive.google.com/drive/folders/1D_pAOXA6edDYc2_gDcfBTB0iJiTR7XvM)

中文版介绍提供在此链接 [link]()

We will provide more details of searching and training soon.
If you have any questions, please sending an email to "liuzhu_ssdut@foxmail.com"

## Abstract
Image fusion plays a key role in a variety of multi-sensor-based vision systems, especially for enhancing visual quality and/or extracting aggregated 
features for perception. However, most existing methods just consider image fusion as an individual task, 
thus ignoring its underlying relationship with these downstream vision problems. Furthermore, 
designing proper fusion architectures often requires huge engineering labor. It also lacks mechanisms to improve 
the flexibility and generalization ability of current fusion approaches.
 To mitigate these issues, we establish a Task-guided, Implicit-searched and Meta-initialized (TIM) deep model to address the image fusion problem 
 in a challenging real-world scenario. Specifically, we first propose a constrained strategy to incorporate information from downstream tasks to 
 guide the unsupervised learning process of image fusion. Within this framework, we then design an implicit search scheme to automatically discover
  compact architectures for our fusion model with high efficiency. In addition, a pretext meta initialization technique is introduced to leverage
   divergence fusion data to support fast adaptation for different kinds of image fusion tasks. Qualitative and quantitative experimental results
    on different categories of image fusion problems and related downstream tasks (e.g., visual enhancement and semantic understanding) 
substantiate the flexibility and effectiveness of our TIM. 

### Data and checkpoints
All the required files are in the *[links](https://drive.google.com/drive/folders/1X91RfVWWuI7hYTWY34pmE4y16VAMtAPT?usp=drive_link)*.

## Searching
 Please refer the IAS folder to find the details
## Training

## Testing
The fusion versions for visual enhancement have been updated in IVIF and MmIF.

## Experimental results

# Citation
If you use this code for your research, please cite our paper.


```
@article{liu2023task,
  title={A Task-guided, Implicitly-searched and Meta-initialized Deep Model for Image Fusion},
  author={Liu, Risheng and Liu, Zhu and Liu, Jinyuan and Fan, Xin and Luo, Zhongxuan},
  journal={arXiv preprint arXiv:2305.15862},
  year={2023}
}

```

We also have some image fusion works to address the adversarial atack, efficient architecture design,
and joint learning with  sematic perception tasks.

```
@inproceedings{liu2021searching,
  title={Searching a hierarchically aggregated fusion architecture for fast multi-modality image fusion},
  author={Liu, Risheng and Liu, Zhu and Liu, Jinyuan and Fan, Xin},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1600--1608},
  year={2021}
}

@article{liu2023paif,
  title={PAIF:  Perception-Aware   Infrared-Visible Image Fusion for Attack-Tolerant  Semantic Segmentation},
  author={Zhu Liu and Jinyuan Liu and Benzhuang Zhang and Long Ma and Xin Fan and Risheng Liu},
  journal={ACM MM},
  year={2023},
}
@article{liu2023embracing,
  title={Searching a Compact Architecture for Robust Multi-Exposure Image Fusion},
  author={Liu, Zhu and Liu, Jinyuan and Wu, Guanyao and Chen, Zihang and Fan, Xin and Liu, Risheng},
  journal={IEEE TCSVT},
  year={2024}
}
@article{liu2023bilevel,
  title={Bi-level Dynamic Learning  for Jointly Multi-modality Image Fusion and Beyond},
  author={Zhu Liu and Jinyuan Liu and Guanyao Wu and Long Ma and Xin Fan and Risheng Liu},
  journal={IJCAI},
  year={2023},
}
```

