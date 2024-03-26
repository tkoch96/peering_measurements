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
		'vtramsterdam': (52.359,4.933),
		'vtratlanta': (33.749, -84.388),
		'vtrbangalore': (12.940, 77.782),
		'vtrchicago': (41.803,-87.710),
		'vtrdallas': (32.831,-96.641),
		'vtrdelhi': (28.674,77.099),
		'vtrfrankfurt': (50.074, 8.643),
		'vtrhonolulu': (21.354, -157.854),
		'vtrjohannesburg': (-26.181, 27.993),
		'vtrlondon' : (51.452,-.110),
		'vtrlosangelas': (34.165,-118.489),
		'vtrmadrid': (40.396,-3.678),
	 	'vtrmanchester': (53.48,-2.265),
		'vtrmelbourne': (-37.858, 145.028),
		'vtrmexico': (19.388, -99.138),
		'vtrmiami' : (25.786, -80.229),
		'vtrmumbai' : (19.101, 72.869),
		'vtrnewyork': (40.802,-73.970),
	 	'vtrosaka': (34.677,135.48),
	 	'vtrsantiago': (-33.487, -70.683),
		'vtrparis': (48.836,2.308),
		'vtrsaopaulo' : (-23.561, -46.532),
		'vtrseattle': (47.577, -122.373),
		'vtrseoul': (37.683,126.942),
		'vtrsilicon': (37.312,-121.816),
		'vtrsingapore': (1.322,103.962),
		'vtrstockholm': (59.365,17.943),
		'vtrsydney': (-33.858,151.068),
	 	'vtrtelaviv': (32.086,34.782),
		'vtrtokyo': (35.650,139.619),
		'vtrtoronto': (43.679, -79.305),
		'vtrwarsaw': (52.248,21.027),
	},
}


### You get pop->dev from client/var/mux2dev.txt
### Then run get_next_hops to get the nexthop IP addresses
POP_INTERFACE_INFO = {
	'vtramsterdam': {'dev': 'tap25', 'ip': '100.89.2.6'},
	 'vtratlanta': {'dev': 'tap24', 'ip': '100.88.2.4'},
	 'vtrbangalore': {'dev': 'tap40', 'ip': '100.104.2.36'},
	 'vtrchicago': {'dev': 'tap30', 'ip': '100.94.2.16'},
	 'vtrdallas': {'dev': 'tap35', 'ip': '100.99.2.26'},
	 'vtrdelhi': {'dev': 'tap41', 'ip': '100.105.2.38'},
	 'vtrfrankfurt': {'dev': 'tap28', 'ip': '100.92.2.12'},
	 'vtrhonolulu': {'dev': 'tap54', 'ip': '100.118.2.72'},
	 'vtrjohannesburg': {'dev': 'tap49', 'ip': '100.113.2.54'},
	 'vtrlondon': {'dev': 'tap44', 'ip': '100.108.2.44'},
	 'vtrlosangelas': {'dev': 'tap42', 'ip': '100.106.2.40'},
	 'vtrmadrid': {'dev': 'tap38', 'ip': '100.102.2.32'},
	 'vtrmanchester': {'dev': 'tap52', 'ip': '100.116.2.68'},
	 'vtrmelbourne': {'dev': 'tap47', 'ip': '100.111.2.50'},
	 'vtrmexico': {'dev': 'tap36', 'ip': '100.100.2.28'},
	 'vtrmiami': {'dev': 'tap23', 'ip': '100.87.2.2'},
	 'vtrmumbai': {'dev': 'tap45', 'ip': '100.109.2.46'},
	 'vtrnewyork': {'dev': 'tap34', 'ip': '100.98.2.24'},
	 'vtrosaka': {'dev': 'tap50', 'ip': '100.114.2.56'},
	 'vtrparis': {'dev': 'tap31', 'ip': '100.95.2.18'},
	 'vtrsantiago': {'dev': 'tap51', 'ip': '100.115.2.66'},
	 'vtrsaopaulo': {'dev': 'tap48', 'ip': '100.112.2.52'},
	 'vtrseattle': {'dev': 'tap20', 'ip': '100.84.0.39'},
	 'vtrseoul': {'dev': 'tap46', 'ip': '100.110.2.48'},
	 'vtrsilicon': {'dev': 'tap43', 'ip': '100.107.2.42'},
	 'vtrsingapore': {'dev': 'tap32', 'ip': '100.96.2.20'},
	 'vtrstockholm': {'dev': 'tap39', 'ip': '100.103.2.34'},
	 'vtrsydney': {'dev': 'tap27', 'ip': '100.91.2.10'},
	 'vtrtelaviv': {'dev': 'tap53', 'ip': '100.117.2.70'},
	 'vtrtokyo': {'dev': 'tap26', 'ip': '100.90.2.8'},
	 'vtrtoronto': {'dev': 'tap37', 'ip': '100.101.2.30'},
	 'vtrwarsaw': {'dev': 'tap33', 'ip': '100.97.2.22'}
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




