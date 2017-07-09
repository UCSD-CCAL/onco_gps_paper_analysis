# Onco-GPS Paper Analysis

NbPackage for [Onco-GPS paper]() analysis

## Onco-GPS Notebook Package (NbPackage)
This repository contains the Onco-GPS Notebook Package (NbPackage), which is a complete bundle of computational notebooks, tools, data (or the code to download data not hosted on this GitHub repository), and results from the Onco-GPS paper. A NbPackage is a self-contained, comprehensive bundle of stuff needed to reproduce a computational analysis. The NbPackage model was created to make every Jupyter-based computational analysis fully reproducible. And this NbPackage contains everything you need to reproduce the computational analyses seen in the Onco-GPS paper. 

[Watch](https://www.youtube.com/watch?v=Tph5BVYcbUA) how we're making the Onco-GPS Analysis easily accessbile.

### Book of Analysis
There are 10 notebooks in the notebooks/ directory. They are numbered from 0 to 9. These notebooks compose the "analysis book" for the Onco-GPS paper, where notebook "0 Introduction and Overview.ipynb" (as the name already suggests) is the overview of notebooks 1 to 9, which are the chapters of the analysis book. Just as you read a book moving from chapter 1, to chapter 2, and so on, you run the notebooks moving from notebook 1, to notebook 2, and so forth. For instance, notebook "1 Downloading Data.ipynb" contains code to download data that is too large to put on this GitHub repository. So, simply running notebook 1 would populate the data/ directory with some data hosted elsewhere, leaving you with a complete data/ directory needed to reproduce the analysis.

### Tools for Analysis
You only need 3 things to be able to run all of the notebooks: 1) python>=3.6, 2) R, and 3) Jupyter Notebook (R is required because, although we don't code in R, some python functions internally use R libraries.) Fortunately, Jupyter Notebook comes with Anaconda3, and installing rpy2 (which in turn installs R) with Anaconda is easy. So you only need to install Anaconda, then rpy2 (if you dont already have them)! 

Watch [this](https://youtu.be/xKGaGXmy8j4) to see how to install Anaconda and [this]() to see how to install rpy2 using Anaconda.

### Feedback & Questions
If something is not working or if you have any questions, comments, or concerns, please submit XXX here.
