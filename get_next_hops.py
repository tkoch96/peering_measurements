from subprocess import check_output
import numpy as np
from config import *

taps = list(set([el['dev'] for el in POP_INTERFACE_INFO.values()]))

tap_to_nh = {}
for row in check_output('ip route show table 20000',shell=True).decode().split('\n'):
	fields = row.strip().split(' ')
	try:
		nh = fields[2]
	except IndexError:
		continue
	tap = fields[4]
	tap_to_nh[tap] = nh
	if len(tap_to_nh) == len(taps):
		break

print(tap_to_nh)

from helpers import *
print("Missing : {}".format(get_difference(taps, tap_to_nh)))

import pprint
for loc,obj in POP_INTERFACE_INFO.items():
	if obj['dev'] in tap_to_nh:
		obj['ip'] = tap_to_nh[obj['dev']]

pprint.pprint(POP_INTERFACE_INFO)

