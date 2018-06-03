'''
	file_ops.py

'''

import h5py
import numpy as np
import pandas as pd
import os
from features import *

def gather_images(data_parent_dir):
    '''
        input:
            * paths, a list of the paths that we need to input 
        output:
            * a dataframe containing image data, label, and name 
    '''
    print("preparing to read images")
    
    # initialize data structures 
    data = []
    label = []
    name = []
    convexities = []
    solidities = []
    extents = []
    major_axis_lengths = []
    minor_axis_lengths = []
    
    paths = os.listdir(data_parent_dir)

    scan_ids = []
    
    # iteration over paths, dirs
    for first_path in paths:
        if first_path[0] == '.': # check for random dot files that come up :( 
            continue
        local_dir = os.path.join(data_parent_dir, first_path)
        for image in os.listdir(local_dir):
            scan_ids.append("_".join((first_path, image[:-3])))
            with h5py.File(os.path.join(local_dir, image), 'r') as hf:
                data.append(np.array(hf.get('data')))
                label.append(np.array(hf.get('label')).item(0))
                name.append(np.array(hf.get('name')))
                
                # compute additional features
                convexity, solidity, extent, major_axis_length, minor_axis_length = get_props(hf.get('data')[:, :, 1])
                convexities.append(convexity)
                solidities.append(solidity)
                extents.append(extent)
                major_axis_lengths.append(major_axis_length)
                minor_axis_lengths.append(minor_axis_length)
         
    print(scan_ids[:10])
    scan_ids = ["P_" +scan_id for scan_id in scan_ids]
    
    d = {'pixel_data':data, 'label':label, 'name':name}
    
    d_computed = {'convexity': convexities, 'solidity': solidities, 'extent': extents, 'major_axis_length': major_axis_lengths, 'minor_axis_length': minor_axis_lengths}
   
    df_img = pd.DataFrame(data=d, index=scan_ids)
    df_computed = pd.DataFrame(data=d_computed, index=scan_ids)

    print(len(df[df['label'] == 0]))
    print(len(df[df['label'] == 1]))
    print(len(df))
    
    return (df_img, df_computed, scan_ids)


def drop_excess_rows(scan_ids, precomputed_df):

    drop_list = []
    true_list = []
    scan_ids = set(scan_ids)
    names = precomputed_df.index.values
    for name in names:
        if name not in scan_ids:
            drop_list.append(name)

    print("the drop list has: " + str(len(drop_list)))
    return precomputed_df.drop(drop_list)


def gather_semantic_features(semantics_path):
    semantic_df = pd.read_csv(semantics_path)
    semantic_df.dropna(inplace=True)
    return semantic_df


def encode_categorical_labels(semantic_df, semantic_feature_names):
    '''
        input:
        
        output:
        
    '''
    for feature in semantic_feature_names:
        le = preprocessing.LabelEncoder()
        le.fit(list(semantic_df[feature].astype(str)))
        semantic_df[feature] = le.transform(semantic_df[feature])
    return semantic_df


def one_hot_encoding(semantic_df, semantic_feature_names):
    '''
        input:
        
        output:
        
    '''
    enc = preprocessing.OneHotEncoder(sparse=False)
    semantic_one_hots = enc.fit_transform(semantic_df[semantic_feature_names])
    _, one_hot_length = semantic_one_hots.shape
    return (semantic_one_hots, one_hot_length)


def generate_semantic_df(semantics_path, semantic_feature_names, total_patientIDs):
	'''

		Here, we find that there are semantic descriptions of images that do not
		appear in the h5'd dataset, and also images that appear in this dataset 
		without corresponding semantic descriptions.  
	'''
    
	semantic_df = gather_semantic_features(semantics_path)
	semantic_df = encode_categorical_labels(semantic_df, semantic_feature_names)
	semantic_one_hots, one_hot_length = one_hot_encoding(semantic_df, semantic_feature_names)

	has_semantic = [s1 + "_" + s2 + "_" + s3 for (s1, s2, s3) in zip(list(semantic_df['patient_id']), list(semantic_df['side']), list(semantic_df['view']))]

	semantic_encoded_dict = {img:np.zeros(one_hot_length) for (idx, img) in enumerate(total_patientIDs)}

	for img, patient_id in enumerate(has_semantic): 
		if patient_id in semantic_encoded_dict.keys():
			semantic_encoded_dict[patient_id] = semantic_one_hots[img] 

	print('---')
	print(len(has_semantic))
	print(one_hot_length)
	print(len(total_patientIDs))
	print(len(semantic_encoded_dict.keys()))

	encoded_feature_names = ["one_hot #" + str(x) for x in range(1, one_hot_length+1)]
	semantic_encoded_df = pd.DataFrame.from_dict(semantic_encoded_dict, orient='index', columns=encoded_feature_names)
	print(semantic_encoded_df.index)
	#    semantic_encoded_df.set_index(total_patientIDs)
	return (semantic_encoded_df, encoded_feature_names)
    




            