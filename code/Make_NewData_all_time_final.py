from cmath import nan
import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as f

from rdkit import Chem

import dgl
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer, \
                          one_hot_encoding, atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons, \
                          atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot, \
                          bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot

from functools import partial

### Check if .csv file exists and read data
def check_csv_file(filename, header_check):
    ### Check if .csv file exists
    if os.path.exists(filename):
        ### Read data
        input_file = pd.read_csv(filename, header = header_check)
        return input_file
    else:
        print("###CHECK IF THE %s EXISTS###" % filename)
        sys.exit()

##############################################################################################
'''

Control Feature Data

'''
##############################################################################################
### extract intervention_pk(single dose)
def extract_intervention_pk(string):
    ### extract interverion value([77] -> 77)
    string = string[1:-1]
    ### check if "," exists
    if not "," in string:
        return int(string)
    else:
        return 
         
### Count Number of Data
def count_data(string):
    ### Num of PK data
    term1 = string.count(",") + 1
    ### Num of nan data
    term2 = string.count("nan")
    return term1-term2

### Make choice string(=Use to Make Patient Info)
def make_choice(dfpatient, index):
    
    ### Extract Choice Information(Dict)
    choice_dict = dfpatient.at[index, "choice"]
    choice_string = ""
    ### CASE1 with disease information
    if "disease" in choice_dict:
        ### CASE1-1 with disease + medication information
        if "medication" in  choice_dict:
            choice_string += choice_dict["disease"] + "," + choice_dict["medication"]
        ### CASE1-2 only disease
        else:
            choice_string += choice_dict["disease"] + ",NR"
    ### CASE2 with health information
    else:
        ### CASE2-1 with health + medication information
        if "medication" in choice_dict:
            choice_string += choice_dict["healthy"] + "," + choice_dict["medication"]
        ### CASE2-2 only health
        else:
            choice_string += choice_dict["healthy"] + ",NR"
    
    ### OUTPUT: (health or disease) + (medication)        
    return choice_string 

##############################################################################################
'''

Control smiles data

'''
##############################################################################################
### the chirality information defined in the AttentiveFP
def chirality(atom):  
    
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

### Make featurizer(AttentiveFPAtomFeaturizer and AttentiveFPBondFeaturizer)
def make_featurizer():
    
    ### AtomFeaturizer 
    ### Size: (N, 39)
    AttentiveFPAtomFeaturizer = BaseAtomFeaturizer(
    featurizer_funcs = {'h': ConcatFeaturizer([
        ### One hot encoding for the type of an atom: 16
        partial(atom_type_one_hot, allowable_set = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], 
                                   encode_unknown = True),
        ### One hot encoding for the degree of an atom: 6
        partial(atom_degree_one_hot, allowable_set = list(range(6))),
        atom_formal_charge, # Get formal charge for an atom: 1
        atom_num_radical_electrons, # Get the number of radical electrons for an atom: 1
        partial(atom_hybridization_one_hot, encode_unknown = True), # One hot encoding for the hybridization of an atom(SP, SP2, SP3, SP3D, SP3D2, None): 6
        atom_is_aromatic,  # Get whether the atom is aromatic: 1
        atom_total_num_H_one_hot, # One hot encoding for the total number of Hs of an atom(0, 1, 2, 3, 4): 5
        chirality] # One hot encoding for the Chirality: 3
    )})

    ### BondFeaturizer
    ### Size: (M, 11)
    AttentiveFPBondFeaturizer = BaseBondFeaturizer(
    featurizer_funcs={'e': ConcatFeaturizer([bond_type_one_hot, # One hot encoding for the type of a bond(SINGLE, DOUBLE, TRIPLE, AROMATIC): 4
                                            bond_is_conjugated, # Get whether the bond is conjugated: 1(?)
                                            bond_is_in_ring, # Get whether the bond is in a ring of any size: 1
                                            partial(bond_stereo_one_hot, # One hot encoding for the stereo configuration of a bond: 5(?)
                                                    allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE], encode_unknown = True)])})
                                                                                
    return AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer

##############################################################################################
'''

Preprocess data

'''
##############################################################################################
### Preprocess for timecourses Data
def preprocess_timecourses(timecourses, print_mode):
    
    ### Extract Required Columns
    df = timecourses[[### KEY(FILTER1)
                      "study_name", "label", 
                      ### INTERVENTION KEY(= single dose, FILTER2)
                      "intervention_pk",
                      ### tissue_label(= plasma), measurement_type_label(= concentration), unit(= gram / liter)(FILTER3)
                      "tissue_label", "measurement_type_label", "unit", 
                      ### TIME
                      "time", "time_unit", 
                      ### PK
                      "value", "mean", 
                      ### SUBSTANCE_LABEL(TARGET)
                      "substance_label",
                      ### FILTER5(GROUP, INDIVIDUAL KEY)
                      "group_pk", "individual_pk"
                    ]]
    
    ### Inital number of data
    if (print_mode == 1):
        print("Number of Usable Data(Before Filtering): ", len(df), "\n")
    
    ### Drop dulicate values by KEY(FILTER1)
    if (print_mode == 1):
        print("#####   FILTER1   #####")
    df.dropna(subset = ["study_name", "label"], inplace = True)
    if (print_mode == 1):
        print("Number of Usable Data(After filter1): ", len(df), "\n")
    
    ### Check single dose(FILTER2)
    df["intervention_pk"] = df["intervention_pk"].apply(extract_intervention_pk)
    if (print_mode == 1):
        print("#####   FILTER2   #####")
    df.dropna(subset = ["intervention_pk"], inplace = True)
    df = df.astype({"intervention_pk": "int64"})
    if (print_mode == 1):
        print("Number of Usable Data(After filter2): ", len(df), "\n")
    
    ### Check measurement_type_label, tissue_label, unit(FILTER3)
    if (print_mode == 1):
        print("#####   FILTER3   #####")
    drop_index = df[(df["measurement_type_label"] != "concentration") | (df["tissue_label"] != "plasma") | (df["unit"] != "gram / liter")].index 
    df.drop(drop_index, inplace = True)
    if (print_mode == 1):
        print("Number of Usable Data(After filter3): ", len(df), "\n")
    
    ### Make pk column(value + mean)
    ### (empty + mean) or (value + empty)
    df["pk"] = df["value"].fillna("") + df["mean"].fillna("")    
    
    ### Make df[patient_pk]: Patient Key
    ### For (0 + individual) or (group + 0) 
    df = df.fillna(0)
    df = df.astype({"group_pk": "int", "individual_pk": "int"})
    df["patient_pk"] = df["group_pk"] + df["individual_pk"]
    
    ### group, individual list for extracting data from dfGroup or dfIndividual
    group_list = list(df["group_pk"].unique())
    individual_list = list(df["individual_pk"].unique())
    
    ### Drop unrequired colunm: group_pk, individual_pk
    df.drop(["measurement_type_label", "tissue_label", "unit", "value", "mean", "group_pk", "individual_pk"], axis = 1, inplace = True)
    
    ### Data sumamry
    if (print_mode == 1):
        print("#####   SUMMARY DF   #####")
        print(df.head())
        print("df.columns: ", df.columns, "\n")
    
    return df, group_list, individual_list

### Preprocess for Group data and Invididual data
def process_patientdata(patientdb, key_pk, lookup_list, print_mode):
    ### Sorted by group_pk data ***IMPORTANT*** -> for binding choice values
    dfPatient = patientdb[[key_pk, "measurement_type", "choice"]]
    ### Extract required group_pk
    dfPatient = dfPatient[dfPatient[key_pk].isin(lookup_list) == True]
    ### Extract healthy, disease and medication information
    dfPatient_filtered = dfPatient[(dfPatient["measurement_type"] == "healthy") | (dfPatient["measurement_type"] == "disease") | (dfPatient["measurement_type"] == "medication")]

    ### Final Group Dataframe(empty)
    Patient = pd.DataFrame(columns = ["patient_pk", "choice"])
    
    ### For binding values to the same group
    for _, row in dfPatient_filtered.iterrows():
        num_patient = len(Patient)
        if row[key_pk] not in list(Patient["patient_pk"]):
            ### Make choice string 
            if num_patient > 0 :
                choice_string = make_choice(Patient, index[0])
                Patient.at[index[0], "choice"] = choice_string
            
            ### Update new choice value for patients existing in the dict
            choice_dict = {row["measurement_type"]: row["choice"]}
            Patient.loc[num_patient] = [row[key_pk], choice_dict]
            index = [num_patient]
        else:
            ### Update new choice value for patients existing in the dict
            index = Patient[Patient["patient_pk"] == row[key_pk]].index
            if row["measurement_type"] not in Patient.at[index[0], "choice"]:
                Patient.at[index[0], "choice"][row["measurement_type"]] = row["choice"]
            else:
                Patient.at[index[0], "choice"][row["measurement_type"]] = "NR"
    
    ### Make last choice string with dict       
    choice_string = make_choice(Patient, index[0])
    Patient.at[index[0], "choice"] = choice_string
    
    ### Data sumamry
    if (print_mode == 1):
        print("#####   SUMMARY GROUP   #####")
        print(Patient.head())
        print("df.columns: ", Patient.columns, "\n")
    
    return Patient

### Preprocess for input_data(combine data)
def preprocess_inputdata(df, Patient, Drug, dfSMILES, atom_feat, bond_feat, print_mode):
    
    ### Merge PK, Patient, Drug Information
    if (print_mode == 1):
        print("Number of Usable data: ", len(df))
    input_data = pd.merge(df, Patient, how = "left", left_on = "patient_pk", right_on = "patient_pk").dropna(axis = 0)
    if (print_mode == 1):
        print("Number of Usable data(after merging with Patient data): ", len(input_data))
    input_data = input_data.merge(Drug, how = "left", left_on = "intervention_pk", right_on = "intervention_pk").dropna(axis = 0)
    if (print_mode == 1):
        print("Number of data(after merging with Drug data): ", len(input_data))

    ### drop D-glucose drug data
    drop_index = input_data[(input_data["drug"] == "D-glucose")].index
    input_data.drop(drop_index, inplace = True)
    if (print_mode == 1):
        print("Number of data(without D-glucose data): ", len(input_data)) ### dose info x
    
    ### SMILES
    dfSMILES = dfSMILES[["drug", "cano_smiles"]]
    dfSMILES.rename(columns = {"drug": "substance_label"}, inplace = True)
    ### Merge SMILES data to input_data
    input_data = pd.merge(input_data, dfSMILES, how = "left").dropna(axis = 0)
    input_data.reset_index(drop = False, inplace = True)
    
    ### Make smiles index column
    input_data["data_index"] = ""
    input_data["patient_index"] = ""
    input_data["route_index"] = ""
    input_data["smiles_index"] = ""
    input_data["total_time"] = ""
    input_data["total_pk"] = ""
    input_data["num_pk"] = ""
    
    ### For one-hot encoding
    patient_info = list(input_data["choice"].dropna().unique())
    drug_route_info = list(input_data["route_label"].dropna().unique())
    drug_list = list(input_data["substance_label"].dropna().unique())
    
    if (print_mode == 1):
        print("\n###  EMBEDDING  ###")
        print("--------------------------------------------------------------------")
        print("patient_info( choice -", len(patient_info),"):\n", patient_info)
        print("--------------------------------------------------------------------")
        print("drug_route_info( route -", len(drug_route_info), "):\n", drug_route_info)
        print("--------------------------------------------------------------------\n")
        
        print("DGL Process")
        print("--------------------------------------------------------------------")
        print("Drug_info( drug -", len(drug_list), "):\n", drug_list)
    ### Extract unique data of smiles column
    unique_smiles_data = input_data.drop_duplicates(["cano_smiles"])[["cano_smiles"]]
    ### SMILES DATA SET                                                                    
    smiles_Data = MoleculeCSVDataset(df = unique_smiles_data, smiles_to_graph = smiles_to_bigraph, 
                                 node_featurizer = atom_feat, edge_featurizer = bond_feat, smiles_column = 'cano_smiles',
                                 cache_file_path = "test.bin")
    ### smiles list
    smiles_list = smiles_Data.smiles
    if (print_mode == 1):
        print("--------------------------------------------------------------------\n")
        print("--------------------------------------------------------------------")
    max_num_total = int(input_data["pk"].apply(count_data).max())
    if (print_mode == 1):
        print("Max Length of PK Data: ", max_num_total)
        print("--------------------------------------------------------------------\n")
    
    for idx, row in input_data.iterrows():
        ### Set data index and route index
        input_data.at[idx, "data_index"] = [idx]
        input_data.at[idx, "patient_index"] = [patient_info.index(row["choice"])]
        input_data.at[idx, "route_index"] = [drug_route_info.index(row["route_label"])]
        
        ### Add smiles index info
        input_data.at[idx, "smiles_index"] = [smiles_list.index(row["cano_smiles"])]
        
        ### One-hot encoding
        input_data.at[idx, "choice"] = [patient_info.index(row["choice"])]
        input_data.at[idx, "route_label"] = [drug_route_info.index(row["route_label"])]
        
        ### Make lists with time(h) and pk data
        time_list = list(map(float, row["time"][1:-1].split(", ")))
        if row["time_unit"] == "min":
            time_list = list(map(lambda x: round(x/60, 2), time_list))
        pk_list = list(map(str, row["pk"][1:-1].split(", ")))
        
        ### Remove nan data
        Num_PK_Data = len(pk_list)
        if "nan" in row["pk"]:
            for i in range(Num_PK_Data):
                if (pk_list[Num_PK_Data - i - 1] == 'nan'):
                    del pk_list[Num_PK_Data - i - 1]
                    del time_list[Num_PK_Data - i - 1]
                    
        pk_list = list(map(float, pk_list))
        
        ### Sort time and pk data
        sorted_time_list = list(np.sort(time_list))
        sorted_index = np.argsort(time_list) 
        sorted_pk_list = [pk_list[i] for i in sorted_index]             
        
        Num_PK_Data = len(sorted_pk_list)
        for i in range(max_num_total - Num_PK_Data):
            sorted_pk_list.append(0)
            sorted_time_list.append(0)
        
        ### Update Time, PK, Total_Time, Total_pk, Dose Data
        input_data.at[idx, "total_time"] = sorted_time_list
        input_data.at[idx, "total_pk"] = sorted_pk_list
        input_data.at[idx, "num_pk"] = [Num_PK_Data]
        input_data.at[idx, "dose"] = [float(row["dose"])]
    
    ### Drop Unnecessary columns
    input_data.drop(["time", "time_unit", "intervention_pk", "pk", "patient_pk"], axis = 1, inplace = True)
    return input_data, smiles_Data, patient_info, drug_route_info, drug_list


##############################################################################################
'''

Main

'''
##############################################################################################
### Make Input Data
def make_input(print_mode = 0):
    
    ### Warning off
    pd.set_option('mode.chained_assignment',  None)

    ### TIMECOURSES
    TIMECOURSES = check_csv_file("timecourses.csv", 0)
    ### PKGROUP
    PKGROUP = check_csv_file("groups.csv",0)
    ### PKINDIVIDUAL
    PKINDIVIDUAL = check_csv_file("individuals.csv",0)
    ### PKDRUG
    PKDRUG = check_csv_file("interventions.csv", 0)
    ### PKSMILES
    PKSMILES = check_csv_file("smiles.csv", 0)

    '''
    ##################
    TIMECOURSES
    ##################
    '''
    ### preprocess for timecourses and make group, individual list
    df, group_list, individual_list = preprocess_timecourses(TIMECOURSES, print_mode)
    
    '''
    ##################
    GROUP: ["patient_pk", "choice"]
    ##################
    '''
    ### preprocess for group and make patient_info
    Group = process_patientdata(PKGROUP, "group_pk", group_list, print_mode)
    
    '''
    ##################
    INDIVIDUAL: ["patient_pk", "choice"] (same method as group)
    ##################
    '''
    ### preprocess for individual and make patient_info
    Individual = process_patientdata(PKINDIVIDUAL, "individual_pk", individual_list, print_mode)
    
    '''
    ##################
    GROUP + INDIVIDUAL(Duplicated x)
    PATIENT: ["patient_pk", "choice"]
    ##################
    '''
    ### Make Patient dataframe
    Patient = pd.concat([Group, Individual])
    if (print_mode == 1):
        print("#####   SUMMARY PATIENT   #####")
        print(Patient.head(), "\n")
    
    '''
    ##################
    DRUG
    ##################
    '''
    ### Full data
    Drug = PKDRUG[["intervention_pk", "route_label", "substance_label", "application_label", "dose"]]
    ### Change column name
    Drug.rename(columns = {"substance_label": "drug"}, inplace = True)
    
    ### application_label(Filter)
    drop_index = Drug[(Drug["application_label"] != "single dose")].index 
    Drug.drop(drop_index, inplace = True)
    
    ### Need dose information(Filter) and change type
    Drug.dropna(subset = ["dose"], inplace = True)
    Drug = Drug.astype({"dose": "object"})
    
    if (print_mode == 1):
        print("#####   SUMMARY DRUG   #####")
        print(Drug.head(), "\n")
    
    '''
    ##################
    TOTAL_DATA(input_data)
    ##################
    '''
    ### Featurizer for SMILES DATA
    atom_feat, bond_feat = make_featurizer()
    ### preprocess for all data and make smiles_data(graphs, feature, ...)
    input_data, smiles_data, patient_info, route_info, drug_list = preprocess_inputdata(df, Patient, Drug, PKSMILES, atom_feat, bond_feat, print_mode)
    
    if (print_mode == 1):
        print("#####   SUMMARY INPUT_DATA   #####")
        print(input_data.head())
        print(input_data.columns, "\n")
    
    ### Noramlize time data, change unit for pk and dose data
    mean_time = torch.Tensor(input_data["total_time"]).mean()
    std_time = torch.Tensor(input_data["total_time"]).std()
    norm_time_info = [mean_time, std_time]
    normalized_time = (torch.Tensor(input_data["total_time"]) - mean_time) / std_time
    
    total_pk = torch.log(torch.Tensor(input_data["total_pk"]) * 1000000) ### g -> ug -> log
    total_pk = torch.nan_to_num(total_pk, neginf = -5) ### min = -4.1685(0 -> -5)
    mean_pk = total_pk.mean()   
    std_pk = total_pk.std()
    norm_pk_info = [mean_pk, std_pk]
    normalized_pk = (total_pk - mean_pk) / std_pk
    
    dose = torch.Tensor(input_data["dose"]) 
    
    choice = torch.Tensor(input_data["choice"])
    
    route = torch.Tensor(input_data["route_label"])

    num_pk = torch.Tensor(input_data["num_pk"])
    
    ### additional information
    total_input_remainder = torch.Tensor(input_data["smiles_index"] + input_data["data_index"] + input_data["total_time"] + input_data["patient_index"] + input_data["route_index"] + input_data["total_pk"]) 
    
    ### final_total_input
    ### time 0:19
    ### pk 19:38
    ### Num_pk 38
    ### choice 39
    ### route 40
    ### dose 41
    ### smiles_index 42
    ### data_index 43
    ### real_time 44:63
    ### patient_index 63
    ### route_index 64
    ### real_pk 65:84
    final_total_input = torch.cat([normalized_time, normalized_pk, num_pk, choice, route, dose, total_input_remainder], dim = -1) 
    
    ### Summary
    if (print_mode == 1):
        print("###  final_total_input  ###")
        print("--------------------------------------------------------------------")
        print("Num of final_total_input: ", len(final_total_input))
        print("Shape of final_total_input: ", final_total_input.shape)
        print("--------------------------------------------------------------------")
        print("Number of data by drug")
        for i in range(len(drug_list)):
            print("Total", drug_list[i], "Data:", len(input_data[(input_data["substance_label"] == drug_list[i])]))
            print("SMILES code:", smiles_data.smiles[i], "\n")
        print("--------------------------------------------------------------------")
    
    ### Additional Data
    ### Label
    label = input_data["study_name"] + "," + input_data["label"]
    
    if (print_mode == 1):
        print(list(input_data["study_name"].dropna().unique()))
        print("--------------------------------------------------------------------")

    return final_total_input, smiles_data, label, drug_list, patient_info, route_info, norm_time_info, norm_pk_info
    ### Data with all timepoints
