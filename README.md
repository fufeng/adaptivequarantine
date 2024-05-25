# Computer Code

> custom computer code that allows readers to repeat the results.

## About

There are three code files: common.py and Main.ipynb.

The notations in the code are consistent with those in the manuscript and the supplementary information.

### common.py
> shared variables and functions for Main.ipynb

* symbols 
* functions

### Main.ipynb
> figures in the manuscript

* plotting and graphing

## Instructions

The code is organized into python files or ipython notebooks. 

The software, module and hardware list is given below.

* Software

> Python 3.10.9 and above

* Module

name | version 
------------ | ------------- 
numpy | 1.23.5 
matplotlib | 3.8.0 
networks | 3.1 

* OS

> Mac OS X, Windows, or Linux


We use Anaconda GUI to run our code (which comes with the packages automatically installed). Otherwise, we may run `pip install` with the name and version for every candidate item.


### Get Started

The figures are saved in the two folders below.
```python
'./figures/main/'
```
```python
'./figures/SI/'
```

We can change the corresponding 
```python
_Figure_PATH_
```
if needed.

### How to Obtain the Expressions and Figures

More detailed explanations are given in the comments (for python files and ipython notebooks). 

In particular, for every user-defined function, the corresponding docstring summarize its behavior and document its arguments.
