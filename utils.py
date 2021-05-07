"""
Script for util functions
"""
import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

def make_list(x):
    return x if isinstance(x, list) else [x]

def list_to_string(values):
    return "_".join(str(x) for x in make_list(values))


