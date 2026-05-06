import numpy as np

BASIN_LEVELS = ['EP', 'NA', 'NI', 'SA', 'SI', 'SP', 'WP']
BASIN_REFERENCE = 'EP'  # always the dropped reference level in basin dummies

QUANTILE_LEVELS = np.linspace(0.70, 0.95, 6)  # 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
