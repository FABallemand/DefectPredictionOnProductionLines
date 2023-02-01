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

## Solution

## To do list:

- Improve model performance (remove outliers, adding data, select most relevant features)  
- Decision trees & random forests  
- Add description, advantages and drawbacks for each method  
- Weight on loss function  
- Data augmentation  
- Smote  
- Anomaly detection (good because it will also work on anomalies that are not in the training data)  

- Precision/REcall -> Unbalanced
- ROC -> Balanced

## Team

- AIT BACHIR Romuald  
- ALLEMAND Fabien  
- FLORET Arthur  
- JARDOT Charles  