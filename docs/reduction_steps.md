SPARK-LRS was created to reduce raw science files from the Low Resolution Spectrograph on the 2.4 m Thai National Telescope. It was designed to meet SPEARNET’s need for a dual-slit data reduction pipeline and was also adapted for single-target long-slit observations.

This documentation describes the functionality of the pipeline and explains each user-accessible function in detail. If you are looking for a quick tutorial or a step-by-step guide to using the pipeline, please see [INSERT].

## Introduction

The pipeline is structured into distinct stages, each with a specific goal. First, it sets up the working environment and classifies files based on their headers. The second stage performs the core data reduction, where standard processes are applied. Finally, the data are calibrated and exported. At this stage, it is also possible to create and normalize a light curve of the target.

Accordingly, this tutorial is divided into four sections:

1. File management 
2. Data reduction
3. Calibration and output
4. Light curve creation (optional)

## 1. File management

