import pathlib

NUM_PROCESSES = 8

# used for downloading, calculation of time series etc
YEARS = range(2000, 2019)
MONTHS = range(1, 13)

DISTANCE_FACTORS = 2, 3, 4, 6

LOG_FILE = pathlib.Path(__file__).parent.parent / 'data' / 'logfile.log'

INTERIM_DIR = pathlib.Path(__file__).parent.parent / 'data' / 'interim'

EXTERNAL_DIR = pathlib.Path(__file__).parent.parent / 'data' / 'external'

FIGURES_DIR = pathlib.Path(__file__).parent.parent / 'figures'

FIGSIZE = (12, 7.5)

# are computations for constant distance factors obsolete? if yes, could be complete removed
COMPUTE_CONSTANT_DISTANCE_FACTORS = False
