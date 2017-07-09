# Onco-GPS Paper Analysis
NbPackage for Onco-GPS paper analysis

## Onco-GPS Notebook Package (NbPackage)
This is the overview of the Onco-GPS Notebook Package (NbPackage), which is a complete bundle of computational data (or the code to download the data not hosted on this GitHub repository), tools, and results from the Onco-GPS paper. A NbPackage is a self-contained, comprehensive bundle of stuff needed to reproduce a computational analysis. NbPackage model was created to make every Jupyter-based computational analysis fully reproducable. And this NbPackage contains everything you need to reproduce the computational analyses seen in the Onco-GPS paper.

#### Onco-GPS NbPackage overview
(Show NbPackage introductory video)

### Book of Analysis
There are 10 notebooks in the notebooks/ directory. They are numbered from 0 to 9. There notebooks compose the "analysis book" for the Onco-GPS paper, where the notebook "0 Introduction and Overview.ipynb" (as the name already suggests) is the overview of the notebooks 1 to 9, which are the chapters of the analysis book. Just as you read a book from chapter 1, moving to chapter 2, and so on, you run the notebooks from notebook 1, moving to notebook 2, and so forth. For instance, the notebook "1 Downloading Data.ipynb" contains code to download data that was too large to put on this GitHub repository. So, simply running the notebook 1 would populate the data/ directory with some data hosted else where, leaving you with a complete data/ directory needed to reproduce the analysis.

### Tools for Analysis
You only need 3 things to be able to run all of the notebooks: 1) python>=3.6, 2) R, and 3) Jupyter Notebook (R is required because, although we don't code in R, some python functions internally use R libraries.) Fortunately, Jupyter Notebook comes with Anaconda3, and installing rpy2 (which will in turn install )with Anaconda is easy. So you only need to install Anaconda, then rpy2 (if you dont already have them).

#### How to install Anaconda3

Watch [this quick video](https://youtu.be/xKGaGXmy8j4)

#### How to install rpy2

### Feedback & Questions
If something is not working or if you have any questions, comments, or concerns, please submit XXX here.
