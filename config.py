CACHE_DIR  = "cache"
DATA_DIR = "data"
TMP_DIR = 'tmp'

# AS relationships
C_TO_P = 1
P_TO_C = -1
P_TO_P = 0
S_TO_S = 5

POP_TO_LOC = {
	'peering':{
		'amsterdam01': (52.359,4.933),
	}, 'vultr': {
		'amsterdam': (52.359,4.933),
		'atlanta': (33.749, -84.388),
		'bangalore': (12.940, 77.782),
		'chicago': (41.803,-87.710),
		'dallas': (32.831,-96.641),
		'delhi': (28.674,77.099),
		'frankfurt': (50.074, 8.643),
		'johannesburg': (-26.181, 27.993),
		'london' : (51.452,-.110),
		'losangelas': (34.165,-118.489),
		'madrid': (40.396,-3.678),
		'melbourne': (-37.858, 145.028),
		'mexico': (19.388, -99.138),
		'miami' : (25.786, -80.229),
		'mumbai' : (19.101, 72.869),
		'newyork': (40.802,-73.970),
		'paris': (48.836,2.308),
		'saopaulo' : (-23.561, -46.532),
		'seattle': (47.577, -122.373),
		'seoul': (37.683,126.942),
		'silicon': (37.312,-121.816),
		'singapore': (1.322,103.962),
		'stockholm': (59.365,17.943),
		'sydney': (-33.858,151.068),
		'tokyo': (35.650,139.619),
		'toronto': (43.679, -79.305),
		'warsaw': (52.248,21.027),
	},
}


POP_INTERFACE_INFO = {
    'amsterdam': {'dev': 'tap3', 'ip': '100.67.0.3'},
    'atlanta': {'dev': 'tap2', 'ip': '100.66.0.2'},
    'bangalore': {'dev': 'tap18', 'ip': '100.82.0.18'},
    'chicago': {'dev': 'tap8', 'ip': '100.72.0.8'},
    'dallas': {'dev': 'tap13', 'ip': '100.77.0.13'},
    'delhi': {'dev': 'tap19', 'ip': '100.83.0.19'},
    'frankfurt': {'dev': 'tap6', 'ip': '100.70.0.6'},
    'johannesburg': {'dev': 'tap27', 'ip': '100.91.0.27'},
    'london': {'dev': 'tap22', 'ip': '100.86.0.22'},
    'losangelas': {'dev': 'tap20', 'ip': '100.84.0.20'},
    'madrid': {'dev': 'tap16', 'ip': '100.80.0.16'},
    'melbourne': {'dev': 'tap25', 'ip': '100.89.0.25'},
    'mexico': {'dev': 'tap14', 'ip': '100.78.0.14'},
    'miami': {'dev': 'tap1', 'ip': '100.65.0.1'},
    'mumbai': {'dev': 'tap23', 'ip': '100.87.0.23'},
    'newyork': {'dev': 'tap12', 'ip': '100.76.0.12'},
    'paris': {'dev': 'tap9', 'ip': '100.73.0.9'},
    'saopaulo': {'dev': 'tap26', 'ip': '100.90.0.26'},
    'seattle': {'dev': 'tap7', 'ip': '100.71.0.7'},
    'seoul': {'dev': 'tap24', 'ip': '100.88.0.24'},
    'silicon': {'dev': 'tap21', 'ip': '100.85.0.21'},
    'singapore': {'dev': 'tap10', 'ip': '100.74.0.10'},
    'stockholm': {'dev': 'tap17', 'ip': '100.81.0.17'},
    'sydney': {'dev': 'tap5', 'ip': '100.69.0.5'},
    'tokyo': {'dev': 'tap4', 'ip': '100.68.0.4'},
    'toronto': {'dev': 'tap15', 'ip': '100.79.0.15'},
    'warsaw': {'dev': 'tap11', 'ip': '100.75.0.11'},
}