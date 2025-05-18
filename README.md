AI-Inpainting-Detection
==============================

This project focuses on the detection and precise localization of image forgeries produced through AI-based inpainting.

Project Organization
------------

    ├── README.md                       <- README for developers using this project.
    ├── data
    │   ├── processed                   <- The final, canonical datasets for modeling.
    │   └── raw                         <- The original, immutable data dump.
    ├── docs                            <- Project docs
    ├── models                          <- Trained and serialized models, model predictions, or model summaries
    ├── eval                            <- Models evaluation files
    │   └── figures                     <- Generated graphics and figures to be used in evalutaion.
    ├── requirements.txt                <- The requirements file
    ├── src                             <- Source code for use in this project.
    │   ├── data                        <- Scripts to download or generate data
    |   |   ├── inpaint                 <- Scripts for inpainting models.
    │   │   └── make_dataset.py
    │   ├── features                    <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   └── models                      <- Scripts to train models and then use trained models to make predictions
    │       ├── predict_model.py
    │       └── train_model.py
    └── LICENSE
--------