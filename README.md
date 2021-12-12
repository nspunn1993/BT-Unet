# BT-Unet: A self-supervised learning framework for biomedical image segmentation using Barlow Twins
Deep learning has brought most profound contribution towards biomedical image segmentation to automate the process of delineation in the medical imaging. To accomplish such task, the models are required to be trained using huge amount of annotated or labelled data that highlights the region of interest with binary mask. However, efficient generation of the annotations for such huge data require expert biomedical analyst and extensive manual effort. It is a tedious and expensive task, while also being vulnerable to human error. To address this problem, a self-supervised learning framework, BT-Unet is proposed that uses Barlow Twins approach to pre-train the encoder of a U-Net model via redundancy reduction using unlabeled data to learn data representation, later complete network is fine-tuned to perform actual segmentation. BT-Unet framework can be trained with limited number of annotated samples while having high number of unannotated samples, which is mostly the case in real-world problems.

This repository includes the implementation of BT-Unet framework which can easily be integrated with any U-Net model.

## Citation
```
@article{punn2021btunet,
  title={BT-Unet: A self-supervised learning framework for biomedical image segmentation using Barlow Twins with U-Net models},
  author={Punn, Narinder Singh and Agarwal, Sonali},
  journal={arXiv preprint arXiv:2112.03916},
  year={2021}
}
```
