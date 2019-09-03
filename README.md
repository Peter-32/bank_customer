Bank Customer
=============

# What is the Problem?

## Informal

The classification goal is to predict if the client will make a particular cash investment with the bank.

## Formal

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E

- E - A list of clients with attributes about the clients
- T - Classify if a client will make a particular cash investment with the bank
- P - Maximize recall with 20% precision; aim for a 20% success rate per call and include as many sales as possible


## Assumptions
- Day of the week could be important, probably want to try different transforms
- Month may be useful, certain months might be better.  I don't have a guess for which ones.
- A housing loan will be a positively correlated with the target.  These people have relied on the bank before for loans, and they often have good income.  Most people probably have a house, so I'm interested in which people do or do not have house loans.
- A personal loan indicates that this person probably doesn't have a large amount saved, but they may be looking to gain passive income to pay off debt
- For those with credit in default, they may be less willing to invest because they have other commitments to pay off before investing.  They also might be looking to make cash quickly to remove the stress of a default, so earning passive income may interest them.  It could go either way.
- marital status should have a large affect because I think finances are often different for each marital status group.
- The job title will be a great attribute indicating disposable income.  Might want to estimate the salary of each job as well to have an additional attribute combined across jobs.
- Age should be a good attribute for disposable income to invest.
- Duration should be discarded
- The quarterly indicators will have little effect
- Contact communication type probably doesn't matter
- Previous marketing campaign will be rarely used but useful when there is history.  If successful in the past, it will be more likely to be successful and likewise, failures in the past will more likely be failures again.  Also a success in the past may also lead to a failure in the future if they already have an investment with them.
- Previous number of contacts performed could be useful, there might be a middling number that does best.
- Days since last contacted from a previous campaign could be useful, will have to consider dealing with missing values.  A middling number might do best.

## Why Does It Need to Be Solved?

### Motivation

We can only reach out to so many people, so we want to reach out to those who are most likely to say yes to our investment opportunity.

### Solution Benefits

This solution would be a large improvement over unprioritized calls to clients.

### How the Solution Will Be Used

The model will rank the predictions and we give these rankings to our sales people to reach out the the right people.

## Solving the Problem Manually

The problem would be solved manually by calling those with a favorable age, credit history, and education.  Also people with a high paying job may have a higher priority.

# Prepare Data

## Select Data

**What data do you have available?**

The work has been done for us, we should include all of the data from the bank-additional-full file.  This file includes all the attributes and rows available.  

**What data is not available that you wish was available?**

It would have been nice to have the salary of each client.  Any data that indicates that a user is interested in the investment would help.  Implicityly or explicitly collected data would be helpful to cluster clients together to find those similar to those interested in past campaigns.  Active use on the website or app shows interest and would help towards prioritizing leads.  If credit card data can be used ethically it would help.

**What data should be excluded?**

The duration column is a data leak and should be excluded.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
