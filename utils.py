"""
Script for util functions
"""
import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def list_to_string(values):
    return "_".join(str(x) for x in values)


