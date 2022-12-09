## how to run it: python3 path_hierarchy.py realtive_path_to_input_directory output_file_name


import os
import errno
from collections import defaultdict
from os import path
import glob
import json
import pandas as pd

def return_label(path):

    clincial_data = pd.read_excel(r'./../../BCNB/patient-clinical-data.xlsx')
    clincial_data['label'] = clincial_data['Number of lymph node metastases'].apply(lambda x: 1 if x>0 else 0)

    id = os.path.basename(os.path.dirname(path))
    print(id)
    print(type(id))
    #print(int(id[:-3]))
    #condition = int(id[:-3])
    condition = int(id)
    print(condition)
    label_id = clincial_data[clincial_data['Patient ID'] == condition]['label']
    #print(label_id)
    label = str(label_id.iloc[0])
    #print(label)
    #print(type(label))
    return label


def path_hierarchy_inside_patches(path):
    
    f_base_str = str(path)
    new_f_base_str = f_base_str
    #print(new_f_base_str)
    needed = new_f_base_str.split('/')[3]
    file_n = new_f_base_str.split('/')[5]
    path_for_json = "original_WSI_patches/"  + needed + "/" + file_n
    hierarchy = path_for_json
    return hierarchy


def path_hierarchy_inside(path):



    hierarchy = {
        #'type': 'folder',
        'id': os.path.basename(os.path.dirname(path)),
        #'path': path_for_json[1:],
        'label': return_label(path),
    }
    try:
        hierarchy['patch_paths'] = [
            path_hierarchy_inside_patches(os.path.join(path, contents))
            for contents in os.listdir(path)
        ]
        
    except OSError as e:
        if e.errno != errno.ENOTDIR:
            raise
        #hierarchy['type'] = 'file'

    return hierarchy


def path_hierarchy(path):
    hierarchy = {
        #'type': 'folder',
        #'id': os.path.basename(path),
        #'path': path,
    }

    try:
        hierarchy = [
            path_hierarchy_inside(os.path.join(path, contents, file))
            for contents in os.listdir(path)
            for file in os.listdir(os.path.join(path, contents))
        ]
        
    except OSError as e:
        if e.errno != errno.ENOTDIR:
            raise
        #hierarchy['type'] = 'file'

    return hierarchy

if __name__ == '__main__':
    import json
    import sys

    try:
        directory = sys.argv[1]
        name_file = sys.argv[2]
    except IndexError:
        directory = "."

    #print(json.dumps(path_hierarchy(directory), indent=2, sort_keys=True))
    with open(name_file, "w") as outfile:
        outfile.write(json.dumps(path_hierarchy(directory), indent=2, sort_keys=True))