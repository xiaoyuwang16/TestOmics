# TestOmics
A deep learning framework for integrating and analyzing multi-omics data from plant-microbe interactions. This repository contains the implementation of a feedforward neural network model designed to dissect complex biological mechanisms through nonlinear regression analysis.  

The article is available at:   
Wang, X., Zhang, H., Zhan, X., Li, J., Huang, J., & Qin, Z. (2024). Dissecting the Herbicidal Mechanism of Microbial Natural Product Lydicamycins Using a Deep Learning-Based Nonlinear Regression Model. ACS omega, 9(44), 44778–44784. https://doi.org/10.1021/acsomega.4c07971



## Description
Deep Neural Network analysis for multi-omics data integration.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Data and Results Directory Structure
The example data of three omics is in `/data`

Model artifacts:

Automatically save the best model in `/models`

Analysis outputs:

After model training, will calculate the importance of each omics feature in the whole dataset
The feature importance results will be saved in `/data/processed`

Visualizations:

Feature importance plot and prediction results scatter plot will be saved in `/results/visualizations`

![feature_importance](https://github.com/user-attachments/assets/b25a9358-5d7b-4515-86d4-bf5520c6870f)


## Features
- Multi-omics data integration  
- Deep Neural Network model  
- Feature importance analysis  
- Visualization tools  

## Network construction

![Abstract使用的-g7](https://github.com/user-attachments/assets/7ad0664f-db59-4150-9a04-674121ffbdb8)


![G7paper流程图](https://github.com/user-attachments/assets/d13d313a-2ec5-41f1-a63e-a727ee2c100e)
