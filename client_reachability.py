import os,csv, numpy as np

client_lats_d = csv.DictReader(open(os.path.join('data','client_lats.csv')),delimiter='\t')
client_lats = {}
for row in client_lats_d:
	ip,peer,lat = row['ip'], row['peer'], int(row['lat'])
	if lat == -1: continue
	try:
		client_lats[ip]
	except KeyError:
		client_lats[ip] = {}
	client_lats[ip][peer] = lat

x = [len(client_lats[ip]) for ip in client_lats]
u,c = np.unique(x,return_counts=True)
for _u, _c in zip(u,c):
	print("{} users have reachability to {} peers".format(_c, _u))