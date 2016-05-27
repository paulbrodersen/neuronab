import spreadsheet_io; reload(spreadsheet_io)
import synapses; reload(synapses)
import neurites; reload(neurites)

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
