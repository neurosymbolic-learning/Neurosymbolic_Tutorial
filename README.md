# Neurosymbolic Summer School Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurosymbolic-learning/Neurosymbolic_Summer_School)

This is the material used in the three morning tutorials of the [Neurosymbolic Summer School 2022](http://www.neurosymbolic.org/summerschool.html). The goal of the notebooks is to provide a hands-on component (30 min) to complement the lecture in each session. We provide code as an initial walk-through of baseline methods grounded in [behavior analysis applications](https://arxiv.org/pdf/2104.02710.pdf), with optional exercises for open-ended exploration. The dataset consists of thousands of frames of trajectory data annotated with behavior labels by domain experts. The notebooks demonstrate neurosymbolic programming for behavior quantification, where the task is to learn the relationship between pose and behavior. 

## Setup

You will need Google Colab and a directory to access `code_and_data` from Colab. The data can be downloaded from this public Google drive link: https://drive.google.com/drive/folders/1TS9DPhtpe4oSjA3TJ65niHU7ICNCPXXj?usp=sharing to be placed inside `code_and_data`. For example, the current code assumes the training data is in the path `code_and_data/data/calms21_task1/train_data.npy`.

## Session 1 

The goal of this notebook is to provide a walk-through of the data with example code for training neural networks and programs. 

* Data Visualization
     *  Plot trajectory samples
* Neural Network
     * Train a 1D Conv Net
* Program 
     * Train program given structure
* Visualize Model Weights
* Open-Ended Exploration

## Session 2 

This notebook walks through top-down type-guided enumeration, one approach for learning neurosymbolic programs.

* Running Enumeration
     *  Base DSL
     *  [Morlet Filter DSL](https://arxiv.org/pdf/2106.06114.pdf)
     *  Neurosymbolic DSL     
* Visualize Runtime vs. Classification Performance
* Implement Temporal Filter 
* Open-Ended Exploration

## Session 3

This notebook walks through informed search via [admissible neural heuristics](https://arxiv.org/pdf/2007.12101.pdf) (NEAR), another approach for learning neurosymbolic programs.

* Running NEAR
     *  Base DSL
     *  [Morlet Filter DSL](https://arxiv.org/pdf/2106.06114.pdf)
* Visualize Runtime vs. Classification Performance
* Choose your path:
     *  Open-ended Exploration
     *  Modifying Architecture of Neural Heuristic 
     *  IDDFS Search
     *  Test on Other Behaviors
