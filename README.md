# Defect Prediction on Production Lines

## Project Description

The goal of this project is to develop an AI based solution for a real world [industry problem faced by Valeo](https://challengedata.ens.fr/challenges/36).  

[Valeo](https://www.valeo.com/fr/) is an industry leading French automotive supplier. In order to stay competitive, the company wants to develop a system that is able to identify defects on products before testing.  

Four files containing relevant data are available. The data are mainly values measured during production on different mounting stations as well as additional measures performed on test benches.

The target is to find the best prediction: **Output = f (inputs)**  

## Data Description

**ID** = PROC_TRACEINFO = it’s a unique code given to the product. Example: I-B-XA1207672-190701-00494.  
- XA1207672 is the reference.  
- 190701 is the date: here 01st of July of year 2019.  
- 00494 is the unique code given to the product, whatever it happens, the product will have this id number frozen forever.  

This number is increased by 1 each time we process a new product, every 12 seconds. So for example: I-B-XA1207672-190701-00495 is the next product.

**Inputs**: Input features are measures collected on different assembly stations with the sensors or devices connected to Programmable Logic Controllers which are storing all of them to keep the full quality traceability. Examples: OP070_V_1_angle_value, OP120_Rodage_I_value...  

**Output**: This is the result value of OP130 (test bench). Value 0 is assigned to OK samples (passed) and value 1 is assigned to KO samples (failed). This is the combined result of multiple electrical, acoustic and vibro-acoustic tests.  

## Method

The first step consisted in analysing and preparing the data as there are a few missing values and the classes are very imbalanced.  
Among several AI models from the Python [Scikit-Learn](https://scikit-learn.org/stable/) library, the best results come from the Naive Bayes Classifier. With a little bit of fine tuning (i.e. $var\_smoothing = 10^{-7}$) the ROC AUC score reached 0.6 with cross validation on the training data.  

For more details read: *report/report.pdf*  

The selected method for data preparation and the fine tuned model get a score of 0.6251613691744773 on the [website](https://challengedata.ens.fr), which corresponds to the 156th place at the time of writting.

## Team

- AIT BACHIR Romuald  
- ALLEMAND Fabien  
- FLORET Arthur  
- JARDOT Charles  