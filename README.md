<h1 align="center">
  <br>
  <a href="https://github.com/UCSD-CCAL"><img src="media/ccal-logo-D3.png" width="400"></a>
  <br>
  Onco-GPS Paper Analysis
  <br>
</h1>
<h4 align="center">NbPackage for the <a href="http://www.cell.com/cell-systems/fulltext/S2405-4712(17)30335-6" target="_blank">Onco-GPS paper</a> analysis</h4>
<br><br>

## The Onco-GPS Notebook Package (NbPackage)
This repository contains the Onco-GPS [Notebook Package (NbPackage](https://github.com/UCSD-CCAL/nbpackage), which is everything you need to reproduce the Onco-GPS paper analysis.

[Watch](https://www.youtube.com/watch?v=Tph5BVYcbUA) how we're making the Onco-GPS Analysis easily accessible.

## The Onco-GPS Book of analysis
Inside the `notebooks/` directory you'll find 10 notebooks, numbered 0 to 9. You can think of these notebooks as chapters in the Onco-GPS book of analysis, where notebook 0 ("0 Introduction and Overview.ipynb") is an introduction to the chapters to come. Just as you read a book moving from chapter 1, to chapter 2, and so on, you run the notebooks moving from notebook 1, to notebook 2, and so forth. Notebook 1 ("1 Downloading Data.ipynb") for example, populates the `data/` directory and prepares the datasets in the `data/` directory for use in the analyses of following chapters.

<br><br>
## Reproduce the Onco-GPS paper analysis

### 1. Set up your computer's environment  
Your computer needs the following things to run the notebooks: 1) python>=3.6, 2) Jupyter Notebook, 3) R and two R packages (rpy2 and r-mass), and 4) biopython. In case you're missing these things, the good news is python 3.6 and Jupyter Notebook come with a software bundle called Anaconda. So all you need to do is install Anaconda and use Anaconda to install R, the R packages, and biopython. Here's how:

1. Install Anaconda
   * [Watch how](https://youtu.be/xKGaGXmy8j4) or [go here](https://www.continuum.io/downloads)
2. Install R, rpy2, r-mass, and biopython
   * [Watch how](https://www.youtube.com/watch?v=m8wWZEV4z2A&feature=youtu.be) or in Terminal enter:

  ```
  conda install rpy2
  conda install r-mass biopython
  ```

### 2. Get the Onco-GPS NbPackage

If you're not familiar with git:
1. Click the green "Clone or download" button, and "Download ZIP"
2. Unzip the Onco-GPS NbPackage and put it anywhere in your computer

If you're familiar with git, in Terminal enter:
```
git clone --recursive https://github.com/UCSD-CCAL/onco-gps-paper-analysis.git
```

### 3. Run the notebooks
In Terminal enter:
```sh
jupyter notebook
```
Navigate to the onco-gps-paper-analysis/notebooks directory and begin running the notebooks.
<br><br>
## Feedback
If something's not working or you have questions, comments, or concerns, please [create an issue](https://github.com/UCSD-CCAL/onco-gps-paper-analysis/issues/new).
