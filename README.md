# Neurosymbolic Programming Tutorial (POPL23)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurosymbolic-learning/Neurosymbolic_Tutorial/blob/popl23)

This is the material used in the [Neurosymbolic Summer School 2022](http://www.neurosymbolic.org/summerschool.html) and for the [POPL 2022 tutorial on Neurosymbolic Programming](https://sites.google.com/view/nsptutorial). The goal of the notebooks is to provide a hands-on component (30 min each) to complement the tutorials. We provide code as an initial walk-through of baseline methods grounded in [behavior analysis applications](https://arxiv.org/pdf/2104.02710.pdf), with optional exercises for open-ended exploration. The dataset consists of thousands of frames of trajectory data annotated with behavior labels by domain experts. The notebooks demonstrate neurosymbolic programming for behavior quantification, where the task is to learn the relationship between pose and behavior. 

## Setup

This tutorial is divided into three Jupyter notebooks that we will run on Google Colab. **If you do not wish to save progress, no setup is required. Head over to [Notebook 1](neurosymbolic_notebook1.ipynb) to get started!**

### (Optional) Setup on Google Drive

If you wish to save your work, we recommend manually copying the provided code to your google drive as shown here:
![setup.gif](https://github.com/neurosymbolic-learning/Neurosymbolic_Tutorial/blob/popl23/imgs/setup.gif?raw=true)

Specifically:
1. Visit the [code repository](https://github.com/neurosymbolic-learning/Neurosymbolic_Tutorial/tree/popl23).
2. Click on `Code (Green Button)` > `Download ZIP`. A file `Neurosymbolic_Tutorial-popl23.zip` should be downloaded.
3. Extract the zip file and rename the folder to from `Neurosymbolic_Tutorial-popl23` to `Neurosymbolic_Tutorial`.
4. Upload the folder to the **root folder** in your Google Drive.
5. Navigate to `Neurosymbolic_Tutorial/neurosymbolic_notebook{1/2/3}.ipynb`.
6. Set `WITHIN_GDRIVE` to `True` in the notebook.


## [Notebook 1](neurosymbolic_notebook1.ipynb)

The goal of this notebook is to provide a walk-through of the data with example code for training neural networks and programs. 

* Data Visualization
     *  Plot trajectory samples
* Neural Network
     * Train a 1D Conv Net
* Program 
     * Train program given structure
* Visualize Model Weights
* Open-Ended Exploration

## [Notebook 2](neurosymbolic_notebook2.ipynb)

This notebook walks through top-down type-guided enumeration, one approach for learning neurosymbolic programs.

* Running Enumeration
     *  Base DSL
     *  [Morlet Filter DSL](https://arxiv.org/pdf/2106.06114.pdf)
     *  Neurosymbolic DSL     
* Visualize Runtime vs. Classification Performance
* Implement Temporal Filter 
* Open-Ended Exploration

## [Notebook 3](neurosymbolic_notebook3.ipynb)

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
