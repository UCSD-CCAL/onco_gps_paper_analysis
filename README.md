# Onco-GPS Paper

## Onco-GPS Notebook Package
This is the overview of the Onco-GPS Notebook Package (NbPackage), which is a complete bundle of computational data (or the code to download the data that was too big to host on GitHub), tools, and results from the Onco-GPS paper. A NbPackage is self contained and you can reproduce the computational analyses seen in the paper.

(Show NbPackage introductory video)

### Book of Analysis
There are 10 notebooks in the notebooks/ directory. They are numbered from 0 to 9. There notebooks compose the "analysis book" for the Onco-GPS paper, where the notebook "0 Introduction and Overview.ipynb" (as the name already suggests) is the overview of the notebooks 1 to 9, which are the chapters of the analysis book. Just as you read a book from chapter 1, moving to chapter 2, and so on, you run the notebooks from notebook 1, moving to notebook 2, and so forth. For instance, the notebook "1 Downloading Data.ipynb" contains code to download datasets that were too large to put on GitHub. So, simply running the notebook 1 would populate the data/ directory with the large datasets and leaving you with a complete data/ directory contaiing all of the data needed to reproduce the analysis.

### Tools for Analysis
You only need 3 things to be able to run all of the notebooks: 1) python>=3.6, 2) R, and 3) Jupyter Notebook (R is required because, although we don't code in R, some python functions internally use R libraries.) And fortunately, installing Anaconda3 would install all of these requriements! So you only need to install Anaconda3 following the following tutorial. (if you don't already have it).

(Show environment set-up video)

### Feedback & Questions
If something is not working or if you have any questions, comments, or concerns, please submit XXX here.
