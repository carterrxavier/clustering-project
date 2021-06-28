# Predicting customer churn at Telco


## 1 Project Overview 
- The basis of this project is to find key drivers for logerror by utlizing clustering.

## Deliverables 
- The README file that gives context to the project.
  * This readme includes key findings, take aways, and hypothesis 
- A Final Jupyter notebook containing well organized commented thought process and analysis.

## Project Summary
 - We must aquire, clean, and visualize the data in order to help narrow down true drivers for logerror
 - Models used were  Decision Tree, Random Forest, K-Nearesr Neighbors in order to predict the log error cluster.

## Process

##### Plan -> **Acquire ->** Prepare -> Explore -> Model -> Deliver
> - Store functions that are needed to acquire dat
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module
> - Complete some initial data summarization 
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> **Prepare ->** Explore -> Model -> Deliver
> - Store functions needed to prepare the zillow data; make sure the module contains the necessary imports to run the code. The final function should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
> - Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.
___

##### Plan -> Acquire -> Prepare -> **Explore ->** Model -> Deliver
> - Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, churn 
> - Run at least 2 statistical tests in data exploration.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). 
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> **Model ->** Deliver
> - Establish a baseline accuracy to determine if having a model is better well.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_report.ipynb notebook

