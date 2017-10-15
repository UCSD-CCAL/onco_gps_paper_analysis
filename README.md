<h1 align="center">
  <br>
  <a href="https://github.com/UCSD-CCAL"><img src="media/logo_ccal.png" width="400"></a>
  <br>
  Onco-GPS Paper Analysis
  <br>
</h1>
<h4 align="center">NbPackage for the <a href="http://www.cell.com/cell-systems/fulltext/S2405-4712(17)30335-6" target="_blank">Onco-GPS paper</a> analysis</h4>
<br>

<hr style="border: none; border-bottom: 3px solid #88BBEE;">

This repository contains the Onco-GPS [Notebook Package (NbPackage)](https://github.com/UCSD-CCAL/nbpackage), which is everything you need to reproduce the Onco-GPS paper analysis. [Watch](https://www.youtube.com/watch?v=Tph5BVYcbUA) how we're making the Onco-GPS Analysis easily accessible.

## Introduction and Overview

**Authors:** William Kim(1), Huwate (Kwat) Yeerna(2), Taylor Cavazos(2), Kate Medetgul-Ernar(2), Clarence Mah(3), Stephanie Ting(2), Jason Park(2), Jill P. Mesirov(2,3), and Pablo Tamayo(2,3).

1. Eli and Edythe Broad Institute
2. UCSD Moores Cancer Center
3. UCSD School of Medicine

**Date:** April 17, 2017

**Article:** [*Kim et al.* Decomposing Oncogenic Transcriptional Signatures to Generate Maps of Divergent Cellular States](https://drive.google.com/file/d/0B0MQqMWLrsA4b2RUTTAzNjFmVkk/view?usp=sharing)

### Analysis overview

In this series of notebook chapters, we introduce Onco-*GPS* (OncoGenic Positioning System), a data-driven analysis framework and associated experimental and computational methodology that makes use of an oncogenic activation signature to identify multiple cellular states associated with oncogene activation. In this chapter we will describe the overall method and then we will provide a guide to the remaining chapters. We also provide at the end a guide to download the input datasets.

The Onco-GPS methodology decomposes an oncogenic activation signature  into its constituent components in such way that the context dependencies and different modalities of oncogenic activation are made explicit and taken into account. Once characterized and annotated, these components are used to deconstruct and define cellular states, and to map individual samples onto a novel visual paradigm: a two-dimensional Onco-*GPS* “map.” This resulting model facilitates further molecular characterization and provides an effective analysis and summarization tool that can be applied to explore complex oncogenic states.


The Onco-*GPS* approach is executed in 3 major modular steps as shown in the Figure below. 

<div>
    <img src="media/method_chap0.png" width=2144 height=1041>
</div>

Step I involves the experimental generation of a representative gene expression signature reflecting the activation of an oncogene of interest. In step II, the resulting signature is decomposed into a set of coherent transcriptional components using a large reference dataset that represents multiple cellular states relevant to the oncogene of interest. These components are also biologically annotated and characterized through further analysis and experimental validation (see article). In step III, a representative subset of samples and components are selected to define cellular states using a clustering procedure. The selected components are also used as transcriptional coordinates to generate a two-dimensional map where the selected individual samples are projected relative to these transcriptional coordinates in analogy to a geographical *GPS* system as shown below.

<div>
    <img src="media/GPS.png" width=500 height=500>
</div> 

The Onco-*GPS* map can also be used to display the association of samples with various genomic features, such as genetic lesions, pathway activation, individual gene expression, genetic dependencies and drug sensitivities. We will use the Onco-*GPS* approach to explore the complex functional landscape of cancer cell lines with alterations in the RAS/MAPK pathway. 


### The Onco-GPS methodology is organized in a series of 9 chapters

Before executing these notebooks make sure you download the input datasets as described in the section at the end of this notebook.

Chapter 1: Set up data
  * This chapter downloads data that was too large to put on GitHub, populating data/ directory and leaving the directory with all input data needed for the following analyses.

Chapter 2: Generate oncogenic-activation signature 
  * This chapter shows how to generate the oncogenic signature (step 1 above). This is useful if one is interested in creating an Onco-GPS map for a given oncogene (for which one has a dataset or at least a gene set representing its activation).

Chapter 3: Decompose oncogenic-activation signature and define transcriptional components
  * This chapter shows how to take the oncogenic signature from chapter 1, or any other signature or gene set of interest, and decomposed it into transcriptional components using Non-Negative Matrix Factorization (NMF).

Chapter 4: Annotate transcriptional components
  * This chapter annotates, or characterizes, the transcriptional components found in chapter 2 by matching many types of genomic features to the component profiles (i.e. the rows of the "H" matrix generated in chapter 2). The full results sets produced by this analysis are also stored under the directory "../results" in subfolder: component_annotation.

Chapter 5: Define cellular states and make Onco-GPS map
  * This chapter defines the oncogenic states by clustering the KRAS mutant subset of  the "H" matrix obtained in chapter 2. It also defines a triangular or ternary Onco-GPS map using components C1, C7 and C2, and then projects the KRAS mutant samples on it.

Chapter 6: Annotate cellular states
  * This chapter is similar to chapter 3 but it annotates and characterizes the oncogenic states defined in chapter 4. The full results sets produced by this analysis are also stored under the directory "../results" in subfolder: state_annotation.

Chapter 7: Display genomic features on Onco-GPS map
  * This chapter displays selected genomic features of interest on the KRAS mutants Onco-GPS map including gene, protein and pathway expression, mutations, tissue types etc.

Chapter 8: Define global cellular states and make global Onco-GPS map
  * This chapter defines the global oncogenic states (S1-S15) and corresponding Onco-GPS map using all the KRAS components (C1-C9) defined in chapter 2.

Chapter 9: Display genomic features on global Onco-GPS map
  * This chapter displays selected genomic features of interest on the global Onco-GPS map including gene, protein and pathway expression, mutations, tissue types etc.

### Additional Notes on Using the Notebooks

*  To reproduce the entire analysis one runs the 9 chapters in sequence. If one is interested in applying the methodology to a different oncogene, one would start by generating the oncogenic signature (chapter 2) using an appropriate dataset e.g. one that you generate in your laboratory, one taken from the literature, or a relevant gene set.      

* If one is interested  in exploring the original KRAS mutant or the global Onco-GPS presented in the article, e.g. display your favorite gene mRNA or mutations status, you would go directly to chapters 7 or 9 and modify these chpaters to display the gene or feature of interest.      

* The chapters (notebooks) are organized as a *notebook package (NB),* a collection of subfolders that contains the following subfolders:

 1.  **notebooks:** contains the Jupyter notebooks, corresponding to each chapter (0-9), and **notebook_environment.py**, which is imported into every notebook to set up the notebook environment.

 2.  **data:** contains the input data for the notebooks.
 
 3. **results:** contains the intermediate and final results produced by the notebooks.

 3.  **tools:** the analysis libraries and source code that implements the Onco-GPS method.    

 4. **media:** the images, logos and other supplementary files used by the notebooks.
 

* The analysis in most chapters will run in under a couple of hours of computer execution time. However, because chapters 4 and 6 execute a full annotation sweep using all components and all states against many datasets of genomic features they could take a few days of computer time to execute.

## Reproduce the Onco-GPS paper analysis

### 1. Set up your computer's environment  
#### Requirements: 
  1. python>=3.6
  2. Jupyter Notebook
  3. R and two R packages (rpy2 and r-mass)
  4. biopython

#### To get the requirements:

1. Install Anaconda
   * [Watch how](https://youtu.be/xKGaGXmy8j4) or [go here](https://www.continuum.io/downloads)
2. Install R, rpy2, r-mass, and biopython
   * [Watch how](https://asciinema.org/a/142193)


### 2. Get the Onco-GPS NbPackage

In Terminal enter:
```
git clone --recursive https://github.com/UCSD-CCAL/onco-gps-paper-analysis.git
```

### 3. Run the notebooks
In Terminal enter:
```sh
jupyter notebook
```
Navigate to the onco-gps-paper-analysis/notebooks directory. Inside the `notebooks/` directory you'll find 10 notebooks, numbered 0 to 9. Just like chapters in a book, each notebook builds off the previous notebook. So each notebook should be run one after another startting with 0 and ending with 9.

## Feedback
If something's not working or you have questions, comments, or concerns, please [create an issue](https://github.com/UCSD-CCAL/onco-gps-paper-analysis/issues/new).
