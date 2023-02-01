import pandas as pd

def loadTrainingData(remove_id=False, remove_capuchon_insertion=False):
    """Load training data set

    Args:
        remove_id (bool, optional): Remove the columns containing IDs. Defaults to False.
        remove_capuchon_insertion (bool, optional): remove the column containing "capuchon_insertion" values. Defaults to False.

    Returns:
        (pandas.dataframe, pandas.dataframe): Training dataframes (input and output)
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
    train_input = pd.read_csv("data/train_inputs.csv", header=0).rename(columns=input_header)
    train_output = pd.read_csv("data/train_output.csv", header=0).rename(columns=output_header)

    # Remove "id" column
    if remove_id:
        train_input = train_input[train_input.columns[~train_input.columns.isin(["id"])]]
        train_output = train_output["result"]
    # remove "capuchon_insertion" column
    if remove_capuchon_insertion:
        train_input = train_input[train_input.columns[~train_input.columns.isin(["capuchon_insertion"])]]

    return train_input, train_output
