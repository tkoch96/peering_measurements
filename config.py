CACHE_DIR  = "cache"
DATA_DIR = "data"
TMP_DIR = 'tmp'
FIG_DIR = 'figures'

# private IP space -- from https://github.com/zmap/zmap/blob/master/conf/blacklist.conf
private_ips = [("0.0.0.0", 8), ("10.0.0.0", 8), ("100.64.0.0", 10), ("127.0.0.0",8), ("169.254.0.0",16),
	("172.16.0.0", 12), ("192.0.0.0", 24), ("192.0.2.0", 24), ("192.88.99.0", 24), ("192.168.0.0", 16), ("198.18.0.0", 15),
	("198.51.100.0", 24), ("203.0.113.0", 24), ("240.0.0.0",4), ("255.255.255.255", 32), ("224.0.0.0",4),
	("25.0.0.0",8)]

# AS relationships
C_TO_P = 1
P_TO_C = -1
P_TO_P = 0
S_TO_S = 5

EXCLUDE_PEERS = [6939,15169,9002,33891,9009,577,8075]


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
	# 'amsterdam': {'dev': 'tap1', 'ip': '100.65.0.1'},
	# 'atlanta': {'dev': 'tap2', 'ip': '100.66.0.3'},
	# 'bangalore': {'dev': 'tap3', 'ip': '100.67.0.5'},
	# 'chicago': {'dev': 'tap4', 'ip': '100.68.0.7'},
	# 'dallas': {'dev': 'tap5', 'ip': '100.69.0.9'},
	# 'delhi': {'dev': 'tap6', 'ip': '100.70.0.11'},
	# 'frankfurt': {'dev': 'tap7', 'ip': '100.71.0.13'},
	# 'johannesburg': {'dev': 'tap8', 'ip': '100.72.0.15'},
	# 'london': {'dev': 'tap9', 'ip': '100.73.0.17'},
	# 'losangelas': {'dev': 'tap10', 'ip': '100.74.0.19'},
	# 'madrid': {'dev': 'tap11', 'ip': '100.75.0.21'},
	# 'melbourne': {'dev': 'tap12', 'ip': '100.76.0.23'},
	# 'mexico': {'dev': 'tap13', 'ip': '100.77.0.25'},
	# 'miami': {'dev': 'tap14', 'ip': '100.78.0.27'},
	# 'mumbai': {'dev': 'tap15', 'ip': '100.79.0.29'},
	# 'newyork': {'dev': 'tap16', 'ip': '100.80.0.31'},
	# # 'osaka': {'dev': 'tap17', 'ip': '100.76.0.23'},
	# 'paris': {'dev': 'tap18', 'ip': '100.82.0.35'},
	# 'saopaulo': {'dev': 'tap19', 'ip': '100.83.0.37'},
	'seattle': {'dev': 'tap20', 'ip': '100.84.0.39'},
	# 'seoul': {'dev': 'tap21', 'ip': '100.85.0.41'},
	# 'silicon': {'dev': 'tap22', 'ip': '100.86.0.43'},
	# 'singapore': {'dev': 'tap23', 'ip': '100.87.0.45'},
	# 'stockholm': {'dev': 'tap24', 'ip': '100.88.0.47'},
	# 'sydney': {'dev': 'tap25', 'ip': '100.89.0.49'},
	# 'tokyo': {'dev': 'tap26', 'ip': '100.90.0.51'},
	# 'toronto': {'dev': 'tap27', 'ip': '100.91.0.53'},
	# 'warsaw': {'dev': 'tap28', 'ip': '100.92.0.55'},
	'amsterdam': {'dev': 'tap25', 'ip': '100.89.2.6'},
	'atlanta': {'dev': 'tap24', 'ip': '100.88.2.4'},
	'bangalore': {'dev': 'tap40', 'ip': '100.104.2.36'},
	'chicago': {'dev': 'tap30', 'ip': '100.94.2.16'},
	'dallas': {'dev': 'tap35', 'ip': '100.99.2.26'},
	'delhi': {'dev': 'tap41', 'ip': '100.105.2.38'},
	'frankfurt': {'dev': 'tap28', 'ip': '100.92.2.12'},
	'johannesburg': {'dev': 'tap49', 'ip': '100.113.2.54'},
	'london': {'dev': 'tap44', 'ip': '100.108.2.44'},
	'losangelas': {'dev': 'tap42', 'ip': '100.106.2.40'},
	'madrid': {'dev': 'tap38', 'ip': '100.102.2.32'},
	'melbourne': {'dev': 'tap47', 'ip': '100.111.2.50'},
	'mexico': {'dev': 'tap36', 'ip': '100.100.2.28'},
	'miami': {'dev': 'tap23', 'ip': '100.87.2.2'},
	'mumbai': {'dev': 'tap45', 'ip': '100.109.2.46'},
	'newyork': {'dev': 'tap34', 'ip': '100.98.2.24'},
	'paris': {'dev': 'tap31', 'ip': '100.95.2.18'},
	'saopaulo': {'dev': 'tap48', 'ip': '100.112.2.52'},
	# 'seattle': {'dev': 'tap29', 'ip': ''},
	'seoul': {'dev': 'tap46', 'ip': '100.110.2.48'},
	'silicon': {'dev': 'tap43', 'ip': '100.107.2.42'},
	'singapore': {'dev': 'tap32', 'ip': '100.96.2.20'},
	'stockholm': {'dev': 'tap39', 'ip': '100.103.2.34'},
	'sydney': {'dev': 'tap27', 'ip': '100.91.2.10'},
	'tokyo': {'dev': 'tap26', 'ip': '100.90.2.8'},
	'toronto': {'dev': 'tap37', 'ip': '100.101.2.30'},
	'warsaw': {'dev': 'tap33', 'ip': '100.97.2.22'}
}

cols = ['orange','red','black','gold','magenta','firebrick','salmon','orangered','lightsalmon','sienna','lawngreen','darkseagreen','palegoldenrod',
	'darkslategray','deeppink','crimson','mediumpurple','khaki','dodgerblue','lime','black','midnightblue',
	'lightsteelblue']
markers = ['_','.','<','^','>','D','*','o']

def get_col_marker(i):
	col = cols[i % len(cols)]
	marker = markers[i // len(cols)]
	return {"c":col,"marker":marker}




CAREFUL = False
TIME_OFFSET = 3898022400 ## time delta between pinger pipeline datetime and actual datetime (fixed)
RFD_WAIT_TIME = 2400 # PEERING testbed policy




