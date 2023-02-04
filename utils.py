import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score

from sklearn.preprocessing import StandardScaler

#=============================================================================#
#==== DATA ===================================================================#
#=============================================================================#

def loadTrainingData(train_inputs="data/train_inputs.csv", train_output="data/train_output.csv", remove_id=False, remove_capuchon_insertion=False):
    """Load training data set.

    Args:
        remove_id (bool, optional): Remove the columns containing IDs. Defaults to False.
        remove_capuchon_insertion (bool, optional): remove the column containing "capuchon_insertion" values. Defaults to False.

    Returns:
        (pandas.dataframe, pandas.dataframe): Training dataframes (input and output).
    """
    # New features names
    input_header = {"PROC_TRACEINFO": "id",
                    "OP070_V_1_angle_value": "angle_1",
                    "OP090_SnapRingPeakForce_value": "snap_ring_peak_force",
                    "OP070_V_2_angle_value": "angle_2",
                    "OP120_Rodage_I_mesure_value": "rodage_i",
                    "OP090_SnapRingFinalStroke_value": "snap_ring_final_stroke",
                    "OP110_Vissage_M8_torque_value": "vissage_m8_torque",
                    "OP100_Capuchon_insertion_mesure": "capuchon_insertion",
                    "OP120_Rodage_U_mesure_value": "rodage_u",
                    "OP070_V_1_torque_value": "torque_1",
                    "OP090_StartLinePeakForce_value": "start_line_peak_force",
                    "OP110_Vissage_M8_angle_value": "vissage_m8_angle",
                    "OP090_SnapRingMidPointForce_val": "snap_ring_midpoint_force",
                    "OP070_V_2_torque_value": "torque_2"}
    output_header = {"PROC_TRACEINFO": "id",
                     "Binar OP130_Resultat_Global_v": "result"}

    # Load data and rename columns
    train_input = pd.read_csv(train_inputs, header=0).rename(columns=input_header)
    train_output = pd.read_csv(train_output, header=0).rename(columns=output_header)

    # Remove "id" column
    if remove_id:
        train_input = train_input[train_input.columns[~train_input.columns.isin(["id"])]]
        train_output = train_output[train_output.columns[~train_output.columns.isin(["id"])]]
    # remove "capuchon_insertion" column
    if remove_capuchon_insertion:
        train_input = train_input[train_input.columns[~train_input.columns.isin(["capuchon_insertion"])]]

    return train_input, train_output

def modifyIndividual(individual, id=False, nb_features=12, max_modif_rate=0.005):
    """Slightly modify an individual. Randomly select a number of features which value will be modified by a small percentage.

    Args:
        individual (pandas.dataframe): Row of a dataframe representing an individual.
        id (bool, optional): ID of the individual. Defaults to False.
        nb_features (int, optional): Number of features describing the individual. Defaults to 12.
        modif_rate (float, optional): Maximum modification rate. Defaults to 0.005.

    Returns:
        pandas.dateframe: Row of a dataframe representing an individual.
    """
    # Select index to modify
    to_modify = [] # USE FEATURE NAME INSTEAD
    if id:
        to_modify = np.random.choice(nb_features, rd.randint(1, nb_features), False)
    else:
        to_modify = np.random.choice(nb_features, rd.randint(0, nb_features-1), False)

    # Idea 1: change by a value coherent with standard deviation or something
    # 
    # Idea 2 (implented): change by a given percentage
    for i in to_modify:
        individual.iloc[0, i] += rd.uniform(-max_modif_rate, max_modif_rate) * individual.iloc[0, i]
    
    return individual

def balanceClassesByRemoving(train_input, train_output):
    """Balance classes by removing some valid individuals so there is 50% valid and 50% defective individuals in the popualtion.

    Args:
        train_input (pandas.dataframe): Input dataframe.
        train_output (pandas.dataframe): Output dataframe.

    Returns:
        (pandas.dataframe, pandas.dataframe): New dataframes (input and output) with balanced classes.
    """
    # Index
    defect_index = train_output.index[train_output["result"] == 1].tolist()
    valid_index = train_output.index[train_output["result"] == 0].tolist()

    # Randomly remove some valid individuals
    rd.shuffle(valid_index) # Shuffle in order to eliminate "production correlation"
    train_input = train_input.iloc[valid_index[:len(defect_index)] + defect_index,:]
    train_output = train_output.iloc[valid_index[:len(defect_index)] + defect_index,:]

    return train_input, train_output

def balanceClassesByDuplicating(train_input, train_output, modify=False, id=False, nb_features=12, max_modif_rate=0.005):
    """Balance classes by duplicating some defective individuals so there is 50% valid and 50% defective individuals in the popualtion.

    Args:
        train_input (pandas.dataframe): Input dataframe.
        train_output (pandas.dataframe): Output dataframe.
        modify (bool, optional): Modify individual when duplicating. Defaults to False.

    Returns:
        (pandas.dataframe, pandas.dataframe): New dataframes (input and output) with balanced classes.
    """
    # Index
    defect_index = train_output.index[train_output["result"] == 1].tolist()
    valid_index = train_output.index[train_output["result"] == 0].tolist()

    # Duplicate defective individuals until classes proportion is 50%
    nb_defect = len(defect_index)
    nb_valid = len(valid_index)
    i = 0
    while nb_defect < nb_valid:
        defective_individual_input = train_input.iloc[defect_index[i],:].to_frame().T
        defective_individual_output = train_output.iloc[defect_index[i],:].to_frame().T
        if modify:
            # Slightly modify individual
            modifyIndividual(defective_individual_input, id=id, nb_features=nb_features, max_modif_rate=max_modif_rate) # Maybe does not work (now dataframe)
        train_input = pd.concat([train_input, defective_individual_input,])
        train_output = pd.concat([train_output, defective_individual_output])
        i = (i + 1) % len(defect_index)
        nb_defect += 1

    return train_input, train_output

def splitTrain(train_input, train_output, test_size=0.3, random_state=42):
    """Create training and testing dataframes.

    Args:
        train_input (pandas.dataframe): Input training dataframe.
        train_output (pandas.dataframe): Output training dataframe.
        test_size (float, optional): Proportion of global population used to create test set. Defaults to 0.3.
        random_state (int, optional): Random state. Defaults to 42.

    Returns:
        pandas.dataframe: Input training dataframe.
        pandas.dataframe: Input testing dataframe.
        pandas.dataframe: Output training dataframe.
        pandas.dataframe: Output testing dataframe.
    """
    X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def scaleInputData(X_train, X_test):
    """Scale input data.

    Args:
        X_train (pandas.dataframe): Input training dataframe.
        X_test (pandas.dataframe): Input testing dataframe.

    Returns:
        (pandas.dataframe, pandas.dataframe): Scaled input training and testing dataframes.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

#=============================================================================#
#==== MODEL ==================================================================#
#=============================================================================#

def modelEvaluation(model, X_train, y_train, cross_validation=5, model_name="Model", fig_name="unknown"):

    fig, axs = plt.subplots(1, 3, figsize=(20,10))

    # Cross validation
    y_pred = cross_val_predict(model, X_train, y_train, cv = cross_validation)
    # print(y_pred)
    y_score = cross_val_score(model, X_train, y_train, cv = cross_validation, scoring="accuracy")
    # print(y_score)

    # Precision/Recall/F1
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    # axs[1].set_title("Precisions/Recall/F1")
    axs[1].axis('off')
    data = [["Accuracy (cv 1)", str(y_score[0])],
        ["Accuracy (cv 2)", str(y_score[1])],
        ["Accuracy (cv 3)", str(y_score[2])],
        ["Accuracy (cv 4)", str(y_score[3])],
        ["Accuracy (cv 5)", str(y_score[4])],
        ["Average Accuray", str(np.mean(y_score))],
        ["Accuracy Standar Deviation", str(np.std(y_score))],
        ["Precision", str(precision)],
        ["Recall",str(recall)],
        ["F1", str(f1)]]
    axs[1].table(data, cellLoc='center', loc='center').set_fontsize(10)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred)
    axs[2].set_title("ROC (AUC score = " + str(auc_score) +")")
    axs[2].plot(fpr, tpr, linewidth=2)
    axs[2].plot([0,1],[0,1], 'k--')
    axs[2].axis([0,1,0,1])
    # axs[2].text(0.2, 0.1, "AUC = " + str(auc_score))
    # axs[2].xlabel("False Positive Rate")
    # axs[2].ylabel("True Positive Rate")

    # Confusion matrix
    axs[0].set_title("Confusion Matrix")
    from sklearn import metrics
    # metrics.ConfusionMatrixDisplay.from_predictions(y_train, y_pred).plot(ax=axs[0])
    metrics.ConfusionMatrixDisplay(confusion_matrix = metrics.confusion_matrix(y_train, y_pred),display_labels = [False, True]).plot(ax=axs[0])
    
    fig.suptitle(model_name + " Evaluation")
    plt.savefig("report/img/" + fig_name)
    plt.show()
