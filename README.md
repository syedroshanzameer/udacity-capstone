# Capstone Project - Udacity Azure Machine Learning Engineer

This is the final project which is the Capstone in the Udacity Azure Machine Learning Engineer Nanodegree. This project requires the expertise in the Azure Cloud Machine learning technologies. This acts as the final step in practically implementing the knowledge that I have gathered from the nanodegree.

## Project Set Up and Installation
1. This project requires the creation on compute instance to run Jupyter Notebook & compute cluster to run the experiments.
2. Dataset needs to be manually selected. 
3. Two experiments were run using Auto-ML & HyperDrive
4. The best model that gave good metrics was deployed and consumed.


## Dataset
Name: heart_failure_clinical_records_dataset.csv

### Overview
I have downloaded the dataset from "UC Irvine Machine Learning Repository"

Heart failure (HF) occurs when the heart cannot pump enough blood to meet the needs of the body.Available electronic medical records of patients quantify symptoms, body features, and clinical laboratory test values, which can be used to perform biostatistics analysis aimed at highlighting patterns and correlations otherwise undetectable by medical doctors. Machine learning, in particular, can predict patients’ survival from their data and can individuate the most important features among those included in their medical records.

### Task
Task: This is a classification problem where in I'm trying to predict if the symptons used in the features will cause death in the patient.(Yes or No)
The target variable is "death event"

Thirteen (13) clinical features:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)

### Access

I'm accessing the data from the direct link to the UCI repository in the notebook where in I import the data using TabularDataset library in the  Azure. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
1. I created a Compute Instance with specification "STANDARD_D3_V2" to run Jupyter Notebook in Azure.
2. I have imported the dataset using TabularDataset library.
3. The setting that I used for Auto-ML were 
* "experiment_timeout_minutes", 
* "enable_early_stopping", 
* "n_cross_validations", 
* "max_concurrent_iterations"

automl_settings = {"primary_metric":"accuracy", "experiment_timeout_minutes":30, "enable_early_stopping":True, "n_cross_validations":3,"max_concurrent_iterations": 4}

4. automl_config = AutoMLConfig(compute_target = compute_target, task = 'classification', training_data = train, label_column_name = 'DEATH_EVENT',**automl_settings)

### Results

1. The best performing Algorithm was "VotingEnsemble" with an accuracy of 87.48%
![](screnshots/automl-bestModel.png)

2. Some of the other parameters as shown in the screenshot are the following:
* precision_score_micro 0.8748348348348348
* recall_score_macro 0.8428914238803412
* norm_macro_recall 0.6857828477606823
* AUC_micro 0.9212086031977925

3. Run Details 
![](screnshots/automl-runDetails.png)

4. The list of the Algorithms that ran are shown in the below screenshot
![](screnshots/automl-models.png)

5. I can still improve the model performance by increasing the runs, capturing more data, including more features.


## Hyperparameter Tuning

1. I have used LogisticRegression for this experiment since it is easily understandable and works well with Classification problems.
2. I have used RandomParameterSampling with 3 parameters for this model:
solver
max_iter
C

RandomParameterSampling({'C': choice(0.01, 0.1, 1, 10, 100),
                                        'max_iter' : choice(50,75,100,125,150,175,200),
                                        'solver' : choice('liblinear','sag','lbfgs', 'saga')})

3. I have used the primary metric as "Accuracy" for this problem and I have tried to maximize it.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
1. The best performing accuracy was 92% 
2. The parameters of the model are:
['--C', '0.1', '--max_iter', '50', '--solver', 'liblinear']
3. I could increase the number of parameter ranges that I have used.
I can even change the method of sampling used for the execution to run faster or slower and find good accurate results.

Best Model Screenshot:
![](screnshots/hyper-bestModel.png)

Run Details Screenshot:
![](/screnshots/hyper-runDetails.png)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Since the HyperDrive experiment gave me best metrics i.e Accuracy of 92%, I went ahead and deployed this model. 

Deployed Service Screenshot:
![](/screnshots/hyper-endpoint1.png)

Sample input that I provided for the endpoint to get the response:
![](/screnshots/hyper-endpoint.png)
The endpoint needs to receive the sample in the form of JSON, I have displayed the code and sample in the above screenshot for clarity.

## Screen Recording
Link: https://youtu.be/sNhuO7utmf0 

The sample request to the endpoint is shown at the end of the video.

 Remember that the screencast should demonstrate:
- A working model 
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
