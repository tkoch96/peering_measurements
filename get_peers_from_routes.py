import glob,os
from config import *
from helpers import *

BASE_DIR = "/home/ubuntu/peering_measurements/"
PARSER = os.path.join(BASE_DIR, "client/utils/bird-route-parser/parse.py")
ROUTE_DIR = os.path.join(BASE_DIR, TMP_DIR)
from subprocess import call

## Temporary process output
tmp_all_peers_out_fn = os.path.join(TMP_DIR, 'tmp_peer_output.txt')
tmp_customer_out_fn = os.path.join(TMP_DIR, 'customer_info.txt')
tmp_pref_to_asn_out_fn = os.path.join(TMP_DIR, 'pref_to_will_carry.csv')
tmp_cc_out_fn = os.path.join(TMP_DIR, 'cc_from_routes.csv')

## Our outputs
all_peers_out_fn = os.path.join(DATA_DIR, 'vultr_peers.csv')
all_peers_out_fn_inference = os.path.join(DATA_DIR, 'vultr_peers_inferred.csv')
customer_out_fn = os.path.join(DATA_DIR, 'vultr_customers.csv')
pref_to_asn_out_fn = os.path.join(DATA_DIR, 'vultr_network_to_peers.csv')
cc_out_fn = os.path.join(CACHE_DIR, 'vultr_customer_cone_from_routes.csv')

if True:
	#### Make sure you actually want to parse all the routes, this rewrites the file
	with open(all_peers_out_fn,'w') as f:
		f.write('pop,peer,next_hop,type,ixp\n')
	with open(all_peers_out_fn_inference,'w') as f:
		f.write('pop,peer,next_hop,type,ixp\n')
	with open(customer_out_fn,'w') as f:
		f.write('pop,peer,next_hop,type,customer_prefix\n')
	with open(pref_to_asn_out_fn,'w') as f:
		f.write('pref,asn\n')
	with open(cc_out_fn,'w') as f:
		f.write('parent,child\n')

	for route_fn in glob.glob(os.path.join(ROUTE_DIR, '*_routes.txt')):
		print("Calling parse on {}".format(route_fn))
		call("{} --in {} --out {} --peers".format(PARSER, route_fn, tmp_all_peers_out_fn), shell=True)
		with open(all_peers_out_fn,'a') as f:
			for row in open(tmp_all_peers_out_fn, 'r'):
				f.write(row)
		with open(customer_out_fn, 'a') as f:
			for row in open(tmp_customer_out_fn, 'r'):
				f.write(row) 
		with open(pref_to_asn_out_fn,'a') as f:
			for row in open(tmp_pref_to_asn_out_fn,'r'):
				f.write(row)
		with open(cc_out_fn, 'a') as f:
			for row in open(tmp_cc_out_fn, 'r'):
				f.write(row)

	print("Forming network to peers out fn, don't exit...")
	network_to_peers = {}
	for row in open(pref_to_asn_out_fn, 'r'):
		ntwrk,peer = row.strip().split(',')
		try:
			network_to_peers[ntwrk].append(peer)
		except KeyError:
			network_to_peers[ntwrk] = [peer]
	with open(pref_to_asn_out_fn, 'w') as f:
		for ntwrk,peers in network_to_peers.items():
			peers = list(set(peers))
			f.write("{},{}\n".format(ntwrk, "-".join(peers)))
	cc = {}
	for row in open(cc_out_fn,'r'):
		parent_asn,child_asn = row.strip().split(',')
		try:
			cc[parent_asn][child_asn] = None
		except KeyError:
			cc[parent_asn] = {child_asn: None}
	with open(cc_out_fn,'w') as f:
		for parent,children in cc.items():
			f.write("{},{}\n".format(parent,"-".join(children)))
	print("Done, okay to exit")

popps = {}
pop_to_ixp = {}
peer_to_ixp = {}
peer_to_ixp_direct = {}
peer_to_pop_ixp_direct = {}
peer_to_pop_routeserver = {}
pop_to_nh = {}
with open(all_peers_out_fn_inference,'w') as f:
	for row in open(all_peers_out_fn, 'r'):
		f.write(row)
		if row.startswith('pop'): continue
		pop,peer,nh,tp,ixp = row.strip().split(',')
		try:
			popps[pop,peer].append(tp)
		except KeyError:
			popps[pop,peer] = [tp]
		if ixp != "None":
			try:
				pop_to_ixp[pop][ixp] = None
			except KeyError:
				pop_to_ixp[pop] = {ixp: None}
			if tp == 'ixp_direct':
				try:
					peer_to_ixp_direct[peer].append(ixp)
				except KeyError:
					peer_to_ixp_direct[peer] = [ixp]
				try:
					peer_to_pop_ixp_direct[peer].append(pop)
				except KeyError:
					peer_to_pop_ixp_direct[peer] = [pop]
				try:
					peer_to_ixp[peer].append(ixp)
				except KeyError:
					peer_to_ixp[peer] = [ixp]
				try:
					peer_to_pop_routeserver[peer].append(pop)
				except KeyError:
					peer_to_pop_routeserver[peer] = [pop]
			elif tp == 'routeserver':
				try:
					peer_to_ixp[peer].append(ixp)
				except KeyError:
					peer_to_ixp[peer] = [ixp]
				try:
					peer_to_pop_routeserver[peer].append(pop)
				except KeyError:
					peer_to_pop_routeserver[peer] = [pop]

		pop_to_nh[pop] = nh


for popp,tps in popps.items():
	if 'routeserver' in tps or 'ixp_direct' in tps:
		pop,peer = popp

		#### IF ANY(PEER IN IXP POP) --> ADD IN IXP ROUTESERVER FOR ALL IXP
		#### IF ANY(ALSO HAS IXP_DIRECT) --> ADD IN THOSE TOO

		for ixp in get_difference(pop_to_ixp[pop], peer_to_ixp[peer]): ##  routeserver
			## Assume if a peer is present at a PoP, that they're present
			## at all IXPs at a PoP
			with open(all_peers_out_fn_inference,'a') as f:
				f.write("{},{},{},{},{}\n".format(pop,peer,pop_to_nh[pop],'routeserver',ixp))
		if pop in peer_to_pop_ixp_direct.get(peer,[]):
			with open(all_peers_out_fn_inference,'a') as f:
				for ixp in get_difference(pop_to_ixp[pop], peer_to_pop_ixp_direct[peer]):
					f.write("{},{},{},{},{}\n".format(pop,peer,pop_to_nh[pop],'ixp_direct',ixp))

custs = {}
for row in open(customer_out_fn, 'r'):
	if row.startswith('pop'): continue
	pop,peer,_,tp,cstpfx = row.strip().split(',')
	try:
		custs[pop,peer].append(cstpfx)
	except KeyError:
		custs[pop,peer] = [cstpfx]

combs = {}
tp_to_popps = {}
for popp,tps in popps.items():
	tps = list(set(tps))
	try:
		combs[tuple(sorted(tps))] += 1
	except KeyError:
		combs[tuple(sorted(tps))] = 1
	try:
		tp_to_popps[tuple(sorted(tps))].append(popp)
	except KeyError:
		tp_to_popps[tuple(sorted(tps))] = [popp]
print("{} popps total".format(len(popps)))
print("{} PoPs".format(len(set(pop for pop,_ in popps))))
print("{} ASes".format(len(set(peer for _,peer in popps))))
print(combs)
print(tp_to_popps.keys())
print(list(set(popp[1] for tp in tp_to_popps for popp in tp_to_popps[tp] if 'privatepeer' in tp)))

from helpers import *
## Questions
## do customers announce their own prefixes or something else
## do they do the same thing in every location, different things within a location
all_pfxs = list(set(pfx.split('/')[0] for pfxs in custs.values() for pfx in pfxs))
pfx_to_asn = lookup_asn(all_pfxs)
cust_to_tp = {}
for pop,peer in custs:
	for pfx in custs[pop,peer]:
		asn = pfx_to_asn.get(pfx.split('/')[0])
		if asn is None or asn == 'NA':
			tp = 'unknown'
		elif int(peer) == int(asn):
			tp = 'byop'
		else:
			# what if customer doesn't come right after VULTR, I suppose...
			tp = 'somethingelse'
			print("ASN path hop : {}, cymruwhois ASN : {}".format(peer,asn))
		try:
			cust_to_tp[peer].append(tp)
		except KeyError:
			cust_to_tp[peer] = [tp]
u,c = np.unique(list(tuple(sorted(list(set(vals)))) for vals in cust_to_tp.values()), return_counts=True)
for _u,_c in zip(u,c):
	print("{} -- {}".format(_u,_c))


popps = {}
for row in open(all_peers_out_fn_inference, 'r'):
	if row.startswith('pop'): continue
	pop,peer,nh,tp,ixp = row.strip().split(',')
	try:
		popps[pop,peer].append(tp)
	except KeyError:
		popps[pop,peer] = [tp]
pop_asn_roughtypefn = 'cache/vultr_pop_asn_type.csv'
with open(pop_asn_roughtypefn, 'w') as f:
	for (pop,peer),tps in popps.items():
		# if 'provider' in tps:
		# 	tp = 'provider'
		# elif 'customer' in tps:
		# 	tp = 'customer'
		# elif 'ixp_direct' in tps or 'privatepeer' in tps:
		# 	tp = 'peer'
		# elif 'routeserver' in tps:
		# 	tp = 'route_server_peer'
		# else:
		# 	raise ValueError("Unrecognized type options for {} : {}".format(popp,tps))
		for tp in list(sorted(set(tps))):
			f.write("{},{},{}\n".format(pop,peer,tp))






