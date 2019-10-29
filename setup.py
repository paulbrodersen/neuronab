from setuptools import setup, find_packages
setup(
    name = 'neuronab',
    version = '0.0.0',
    description = 'A python package to isolate neuronal structures such as somata, neurites, and synapses from (immuno-)fluorescence images.',
    author = 'Paul Brodersen',
    author_email = 'paulbrodersen+neuronab@gmail.com',
    url = 'https://github.com/paulbrodersen/neuronab',
    download_url = 'https://github.com/paulbrodersen/neuronab/archive/0.0.0.tar.gz',
    keywords = ['numpy', 'microscopy', 'immunofluorescence', 'image analysis', 'neuroscience', 'neuron'],
    classifiers = [ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    platforms=['Platform Independent'],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'phasepack', 'scikit-image'],
)
