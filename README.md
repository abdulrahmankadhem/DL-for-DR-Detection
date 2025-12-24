# DL-for-DR-Detection

## Problem Overview
Diabetic Retinopathy is a complication of Diabetes which causes damage to the retina. If left untreated the disease results in significant vision loss and even blindness. 

With globally increasing rates, the early detection of the disease has become a crucial part of vision loss prevention.

This is where deep learning comes into play. Modern DL algorithms can be finetuned to detect the disease from retinal images and grade its severity, aiding ophthalmologists in the diagnosis process.

## Project Scope

During the course of the fyp, 4 algorithms were finetuned for DR detection and grading.

This repo includes code used to finetune and thoroughly evaluate ConvNeXt Tiny, which was the best performing algorithm. Codes used for binary as well as multiclass classification are included.

In addition the repo also includes the full fyp report for those interested in further reading.

## Dataset

The dataset used for finetuning and testing in all cases was the APTOS 2019 dataset. 

It includes about 4000 color fundus images annotated by a team of medical professionals on a severity scale of 0-4.

The dataset is available at: "https://www.kaggle.com/datasets/mariaherrerot/aptos2019"

## Methodology

A pretrained model is loaded from Pyrtorch and the head is redefined to contain a select number of classes. 

After specifying hyperparameters, training begins. Images are preprocessed using resizing, augmentation, green channel enhancement, and normalization.

The model's accuracy is assessed after each epoch and is then used to update LR or stop training completely. The best version is saved each epoch.

Finally, after training concludes, the best model is assessed using an extensive list of qualitative and quantitative metrics.

## Results

Scores achieved are agreeable with relevant studies of the same scope.


