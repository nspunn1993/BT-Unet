# BT-Unet: A self-supervised learning framework for biomedical image segmentation using Barlow Twins
Deep learning has brought most profound contribution towards biomedical image segmentation to automate the process of delineation in the medical imaging. To accomplish such task, the models are required to be trained using huge amount of annotated or labelled data that highlights the region of interest with binary mask. However, efficient generation of the annotations for such huge data require expert biomedical analyst and extensive manual effort. It is a tedious and expensive task, while also being vulnerable to human error. To address this problem, a self-supervised learning framework, BT-Unet is proposed that uses Barlow Twins approach to pre-train the encoder of a U-Net model via redundancy reduction using unlabeled data to learn data representation, later complete network is fine-tuned to perform actual segmentation. BT-Unet framework can be trained with limited number of annotated samples while having high number of unannotated samples, which is mostly the case in real-world problems.

This repository includes the implementation of BT-Unet framework which can easily be integrated with any U-Net model using Tensorflow v2.6.

The code for BT-Unet offers following functionalties:
- Developing a U-Net model.
- Extracting the encoder part and converting it into Siamese-Net for pre-training using Barlow Twins.
- Integrating the pre-trained encoder with U-Net model for fine-tuning.
- Evaluation of the BT-UNet framework with standard metrics.
- Output visualization.

A jupyternotebook is also included highlighting the execution of the code with and without the pre-training with Barlow Twins.

## Citation
```
@article{punn2022bt,
  title={BT-Unet: A self-supervised learning framework for biomedical image segmentation using Barlow Twins with U-Net models},
  author={Punn, Narinder Singh and Agarwal, Sonali},
  journal={Machine Learning},
  pages={1--16},
  year={2022},
  publisher={Springer}
}
```
