# DETiN
The increasing realism of generative inpainting techniques has raised serious concerns in digital forensics and media authenticity. In this work, we introduce DETiN (DEtection of Inpainted areas), a novel framework for the pixel-level detection of manipulated regions in images. Our approach leverages a multimodal input representation combining RGB, frequency, and noise residual information to highlight artifacts left by generative models. We construct a custom dataset based on MS-COCO, enriched with synthetic inpainted images generated using two state-of-the-art diffusion models and diverse masking strategies. Additionally, we propose a CNN-based solution to generate reliable ground truth tampering masks. DETiN employs a multi-branch ResNet-50 backbone to independently process each modality, followed by a dedicated fusion and prediction block for final segmentation. Experimental results show that our model significantly outperforms a DeepLabv3 baseline, demonstrating capability in detecting subtle generative manipulations. However, compared to the state-of-the-art methods in the literature, our performance remains limited—primarily due to the relatively small size of the training dataset and the inherent noise present in the ground-truth masks used during validation and testing. Future work will focus on reducing model complexity, enlarging and diversifying the dataset, and improving mask quality to increase both robustness and generalization.

## Project Organization
```
├── README.md                                <- README for developers using this project.
├── LICENSE                                  <- The license file for this project.
├── requirements.txt                         <- The requirements file listing project dependencies.
├── setup.py                                 <- Setup script for installing the dependencies in venv.
├── data                                     <- Data directory containing raw, processed and utility data.
│   ├── processed                            <- The final, canonical datasets for modeling.
│   ├── raw                                  <- The original, immutable data dump.
│   │   ├── CASIA2                           <- CASIA2 dataset.
│   │   │   ├── au_list.txt
│   │   │   ├── list_acronyms.json
│   │   │   ├── README.md
│   │   │   └── tp_list.txt
│   │   └── COCO                             <- COCO dataset.
│   │       └── README.md
│   └── util                                 <- The data used in interim.
│       ├── annotations                      <- Annotations and captions for COCO dataset.
│       │   ├── captions_val2017.json
│       │   └── instances_val2017.json
│       ├── inpaint_log.csv                  
│       └── prompts.json                     
├── docs                                     <- Project documentation.
├── models                                   <- Trained and serialized models, model predictions, or model summaries.
│   ├── CNN_masks                            <- CNN_mask model checkpoints and configuration files.
│   ├── DETIN                                <- DETiN model checkpoints and configuration files.
│   └── inpaint                              <- Inpainting models downloaded.
└── src                                      <- Source code for use in this project.
    ├── data                                 <- Scripts to generate and use datasets.
    │   ├── CNN_masks                        <- Dataset loaders for CNN mask prediction.
    │   │   ├── cnn_inference_dataset.py
    │   │   └── cnn_training_dataset.py
    │   ├── DETiN_dataset.py                 <- Custom dataset class used by the DETiN model.
    │   ├── __init__.py
    │   ├── inpaint                          <- Scripts for generating COCO-Inpaint.
    │   │   ├── __init__.py
    │   │   ├── inpaint_models.py
    │   │   ├── mask_generator.py
    │   │   └── prompt_generator.py
    │   └── make_dataset.py                  <- Main dataset builder script for preparing COCO-Inpaint.
    ├── DeepLv3(old)_pipeline.py             <- Experimental pipeline using DeepLabv3.
    ├── DETiN_pipeline.py                    <- Main training and evaluation pipeline for DETiN model.
    ├── features                             <- Scripts to turn raw data into features for modeling.
    │   ├── build_features.py  
    │   ├── create_pairs.py                
    │   ├── create_triplets.py               
    │   ├── __init__.py
    │   └── visualization.py                 
    ├── generate_GTMasks.py                  <- Script to generate ground-truth masks.
    ├── __init__.py
    └── models                               <- Scripts to train and run models.
        ├── CNN_masks                        <- CNN architecture and training code for mask generation.
        │   ├── cnn_eval.py
        │   ├── cnn_infer.py
        │   ├── cnn_mask.py
        │   ├── cnn_training.py
        │   └── __init__.py
        ├── DETiN                            <- DETiN architecture and training code for inpaint detection.
        │   ├── detin_inference.py
        │   ├── detin_metrics.py
        │   ├── detin_model.py
        │   ├── detin_training.py
        │   └── __init__.py
        └── __init__.py
```