import os
import os.path
import warnings

import spreadsheet_io
import synapses
import neurites

if os.name in ('posix', 'mac'):
    test_spreadsheet_path = "./test_images/template_linux.xlsx"
elif os.name in ('nt'):
    test_spreadsheet_path = "./test_images/template_windows.xlsx"
else:
    test_spreadsheet_path = None
    warnings.warn("Could not detect operating system automatically. " + \
                  "Automatic tests won't run without user input.")

def prompt_for_existing_path(msg):
    valid = False
    attempt = 0
    while not valid:
        if attempt > 0:
            print("Path does not exist!")
        path = input(msg)
        valid = os.path.exists(path)
        attempt += 1
    return path

def count_synapses(spreadsheet_path):
    """
    Script that runs synapses.count using arguments supplied by spreadsheet at path.
    """
    spreadsheet_io.apply_and_append(
        spreadsheet_path,
        func=synapses.count,
        arguments=['neurite_marker',
                   'primary_synaptic_marker',
                   'secondary_synaptic_marker',
                   'min_synapse_size',
                   'max_synapse_size',
                   'min_synapse_brightness',
                   'show',
                   'save'],
        returns=['neurite_length',
                 'primary_count',
                 'secondary_count',
                 'dual_labelled']
    )
    return

def test_count_synapses(spreadsheet_path=test_spreadsheet_path):
    if spreadsheet_path == None:
        spreadsheet_path = prompt_for_existing_path("Please provide the full path to the spreadsheet: \n")
    return count_synapses(spreadsheet_path)

def isolate_neurites(spreadsheet_path):
    """
    Script that runs neurites.isolate
    using arguments supplied by spreadsheet at path.
    """
    spreadsheet_io.apply_and_append(
        spreadsheet_path,
        func=neurites.isolate,
        arguments=['neurite_marker',
                   'show',
                   'save'],
        returns=[]
    )
    return

def test_isolate_neurites(spreadsheet_path=test_spreadsheet_path):
    if spreadsheet_path == None:
        spreadsheet_path = prompt_for_existing_path("Please provide the full path to the spreadsheet: \n")
    return isolate_neurites(spreadsheet_path)
