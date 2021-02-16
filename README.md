# IBM Recommender System

### Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Running](#running)
4. [Repository Structure](#repo)
5. [Final Considerations](#considerations)

## Overview <a name="overview"></a>

The goal of this project is to improve the IBM Watson articles recommendation.
At this project you will find a notebook showing the implementation of the
3 types of recommendation approaches most used, also there is a python module,
where you can find all the functions used at the notebook encapsulate and ready
for you to use.

This project is part of the Udacity Data Science Nanodegree program.

## Installation <a name="installation"></a>

Create a virtual environment named **ibm_venv**.

```
$ python3 -m venv ibm_venv -- for Linux and macOS
$ python -m venv ibm_venv -- for Windows
```

After that, activate the python virtual environment

```
$ source ibm_venv/bin/activate -- for Linux and macOS
$ ibm_venv\Scripts\activate -- for Windows
```

Install the requirements

```
$ pip install -r requirements.txt
```

## Running <a name="running"></a>

The two csvs needed to run the notebook are at the data folder, but because the
process to make a recommendation using NLP is expensive, the user could download
the resulting csv at the link below and put the file in the data folder.

 `https://drive.google.com/file/d/1jNjllAzcnM50nTHFJMx7Fv0QPVwm5d0R/view?usp=sharing`

This csv is needed also to run the tests, by the way to run the tests you can
 execute the command:

`python tests.py`

Finally you can create a file script and import the Recommender from the 
recommender_template.py.

## Repository Structure <a name="repo"></a>

- The `data` folder contains the disaster's data and the script to clean and store the data.
- The `requirements.txt` has the needed packages to run the code.
- `Recommendations_with_IBM` notebook with all the process
- `recommender_functions` python script with all the functions
- `recommender_template` python script with the Recommender class
- `project_tests` the project tests

## Final Considerations and acknowledgments <a name="considerations"></a>

Part of the code used is inspired from the Experimental Design & Recommendations
 module of the Udacity Nanodegree.
Go ahead and contribute to this repository.
the data was kindly provided by IBM Watson Studio Platform
