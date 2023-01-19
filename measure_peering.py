import netifaces as ni, os, re, socket, csv, time, json, numpy as np, pickle, geopy.distance, copy, glob
from constants import *
from subprocess import call, check_output
import subprocess

from pinger_wrapper import Pinger_Wrapper


CAREFUL =  True
np.random.seed(31415)
RFD_WAIT_TIME = 600 # PEERING testbed policy

### To get latency measurements
# start tcpdump
# for peer in pop
# measure to all users of interest from the peer
# end tcpdump
# post-process pcaps to look at latencies


### For pairwise preferences
# for pair in pairs
# start tcpdump
# measure to all users of interest
# end tcpdump
# parse tcpdump to look at who wins

def check_ping_responsive(ips):
	print("Checking responsiveness for {} IP addresses".format(len(ips)))
	ret = []
	max_l_seq = 3000
	n_chunks = int(np.ceil(len(ips) / max_l_seq))
	ip_chunks = split_seq(ips,n_chunks)
	for ips in ip_chunks:
		addresses_str = " ".join(ips)
		if addresses_str != "":
			out_fn = 'tmp.warts'
			scamp_cmd = 'sudo scamper -O warts -c "ping -c 1" -p 8000 -M tak2154atcolumbiadotedu'\
				' -l peering_interfaces -o {} -i {}'.format(out_fn, addresses_str)
			try:
				check_output(scamp_cmd, shell=True)
				cmd = "sc_warts2json {}".format(out_fn)
				out = check_output(cmd, shell=True).decode()
				for meas_str in out.split('\n'):
					if meas_str == "": continue
					measurement_obj = json.loads(meas_str)
					meas_type = measurement_obj['type']
					if meas_type == 'ping':
						dst = measurement_obj['dst']
						if measurement_obj['responses'] != []:
							ret.append(dst)
			except:
				# likely bad input
				import traceback
				traceback.print_exc()
				pass
	return ret

def measure_latency_ips(ips):
	print("Measuring latency for ".format(ips))
	ret = {ip:[] for ip in ips}
	addresses_str = " ".join(ips)
	if addresses_str != "":
		out_fn = 'tmp.warts'
		scamp_cmd = 'sudo scamper -O warts -c "ping -c 20" -p 8000 -M tak2154atcolumbiadotedu'\
			' -l peering_interfaces -o {} -i {}'.format(out_fn, addresses_str)
		try:
			check_output(scamp_cmd, shell=True)
			cmd = "sc_warts2json {}".format(out_fn)
			out = check_output(cmd, shell=True).decode()
			for meas_str in out.split('\n'):
				if meas_str == "": continue
				measurement_obj = json.loads(meas_str)
				meas_type = measurement_obj['type']
				if meas_type == 'ping':
					dst = measurement_obj['dst']
					if measurement_obj['responses'] != []:
						dst = measurement_obj['dst']
						ret[dst] = [response['rtt'] * .001 for response in measurement_obj['responses']]
		except:
			# likely bad input
			import traceback
			traceback.print_exc()
			pass
	return ret

def get_intersection(set1, set2):
	"""Gets intersection of two sets."""
	return list(set(set1) & set(set2))
def get_difference(set1, set2):
	"""Gets set1 - set2."""
	set1 = set(set1); set2 = set(set2)
	return list(set1.difference(set2))
def split_seq(seq, n_pieces):
	# splits seq into n_pieces chunks of approximately equal size
	# useful for splitting things to divide among processors
	newseq = []
	splitsize = 1.0/n_pieces*len(seq)
	for i in range(n_pieces):
		newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
	return newseq
DATA_DIR = "data"

def pref_to_ip(pref):
	return ".".join(pref.split(".")[0:3]) + ".1"
def ip_to_pref(ip):
	return ".".join(ip.split(".")[0:3]) + ".0/24"

class Peering_Pinger():
	def __init__(self, system, mode):
		self.run = {
			'calculate_latency': self.calculate_latency,
			'pairwise_preferences': self.pairwise_preferences,
			'quickvultrtesting': self.quickvultrtesting,
			'measure_prepainter': self.measure_prepainter, # measures performances to calculate painter
			'conduct_painter': self.conduct_painter, # conducts painter-calculated advertisement
		}[mode]

		assert system in ["peering", "vultr"]
		self.system = system

		all_muxes_str = check_output("sudo client/peering openvpn status", shell=True).decode().split('\n')
		if self.system == "peering":
			self.pops = ['amsterdam01']
		elif self.system == "vultr":
			self.pops = []
			for row in all_muxes_str:
				if row.strip() == '': continue
				self.pops.append(row.split(' ')[0])
		self.pops = sorted(self.pops)
		print("Working with PoPs : {}".format(self.pops))
		self.pop_to_intf = {}
		for pop in self.pops:
			this_mux_str = [mux_str for mux_str in all_muxes_str if pop in mux_str]
			assert len(this_mux_str) == 1
			this_mux_str = this_mux_str[0]
			self.pop_to_intf[pop] = "tap" + re.search("tap(\d+)", this_mux_str).group(1)
		
		self.default_intf = 'eth0'
		self.internal_ip = ni.ifaddresses(self.default_intf)[ni.AF_INET][0]['addr']
		self.internal_ip_bytes = socket.inet_aton(self.internal_ip)

		## By convention, I use the 241 for unicast
		self.unicast_pref = "184.164.241.0/24"
		self.unicast_addr = "184.164.241.1"

		### prefixes we can conduct experiments with
		self.available_prefixes = ["184.164.238.0/24","184.164.239.0/24","184.164.240.0/24", "184.164.241.0/24","184.164.242.0/24","184.164.243.0/24"]
		self.rfd_track_fn = os.path.join(CACHE_DIR, 'rfd_tracking.csv')
		self.active_experiments = {}

		popps = csv.DictReader(open(os.path.join(DATA_DIR, "{}_peers.csv".format(self.system)), 'r'))
		self.peers = {}
		self.peer_to_id = {}

		self.peer_macs = {}
		self.peer_mac_fn = os.path.join(DATA_DIR, '{}_peer_macs.csv'.format(self.system))
		if not os.path.exists(self.peer_mac_fn):
			self._call("touch {}".format(self.peer_mac_fn))
		with open(self.peer_mac_fn, 'r') as f:
			for row in f:
				if row[0:3] == 'pop': continue
				pop,sstm,pm,peer = row.strip().split("\t")
				if sstm != self.system: continue
				try:
					self.peer_macs[pop][pm] = peer
				except KeyError:
					self.peer_macs[pop] = {pm: peer}
		for pop in self.pops:
			try:
				self.peer_macs[pop]
			except KeyError:
				self.peer_macs[pop] = {}

		for row in popps:
			if self.system == "peering":
				pop, peer, session_id = row["BGP Mux"], row["Peer ASN"], row["Session ID"]
				if ":" in row["Peer IP address"]: continue
			elif self.system == "vultr":
				pop, peer, nh = row["pop"], row["peer"], row["next_hop"]
				session_id = peer #### Whats used by the -c option to identify a peer.. idk what this should be
			try:
				self.peers[pop].append(peer)
			except KeyError:
				self.peers[pop] = [peer]
			try:
				self.peer_to_id[pop,peer]
				# print("Warning -- multiple session ID's for {} {}".format(pop,peer))
			except KeyError:
				self.peer_to_id[pop,peer] = session_id
		for pop in self.peers:
			self.peers[pop] = list(set(self.peers[pop])) # remove dups

		self.reachable_dsts, self.measured_to = {}, {}
		if os.path.exists(os.path.join(DATA_DIR, "client_lats_{}.csv".format(self.system))):
			with open(os.path.join(DATA_DIR, "client_lats_{}.csv".format(self.system)), 'r') as f:
				for row in f:
					if row.startswith("ip"): continue
					client_dst, peer, rtt = row.strip().split('\t')
					if rtt != '-1':
						try:
							self.reachable_dsts[peer].append(client_dst)
						except KeyError:
							self.reachable_dsts[peer] = [client_dst]
					try:
						self.measured_to[peer].append(client_dst)
					except KeyError:
						self.measured_to[peer] = [client_dst]
		self.pop_to_loc = POP_TO_LOC[self.system]
		self.pop_vpn_lats = {}
		self.pop_vpn_lats_fn = os.path.join(CACHE_DIR, 'pop_vpn_lats_{}.pkl'.format(self.system))
		if os.path.exists(self.pop_vpn_lats_fn):
			self.pop_vpn_lats = pickle.load(open(self.pop_vpn_lats_fn,'rb'))
		self.vpn_ips = {'vultr': {
			'miami':'149.28.108.63',
			'atlanta':'144.202.17.174', 
			'amsterdam':'95.179.176.143',
			'tokyo':'108.61.180.64' ,
			'sydney':'149.28.166.89' ,
			'frankfurt':'45.76.88.182',
			'seattle':'137.220.39.64' ,
			'chicago':'149.28.116.32',
			'paris':'95.179.222.150',
			'singapore':'139.180.128.6', 
			'warsaw':'70.34.251.85', 
			'newyork':'207.148.19.78', 
			'dallas':'45.32.204.181',
			'mexico':'216.238.83.87',
			'toronto':'149.248.53.68',
			'madrid':'208.85.20.15' ,
			'stockholm':'70.34.214.25', 
			'bangalore':'139.84.136.51',
			'delhi':'139.84.162.121' ,
			'losangelas':'45.63.56.176',
			'silicon':'149.28.208.254' ,
			'london':'45.63.101.157',
			'mumbai':'65.20.81.121',
			'seoul':'141.164.58.13',
			'melbourne':'67.219.104.2',
			'saopaulo':'216.238.110.148',
			'johannesburg':'139.84.230.180',
		}}[self.system]
		self.pops_list = list(self.vpn_ips)

		# def parse_row_loc(row):
		# 	if row['lat'] == 'NaN' or row['lon'] == 'NaN':
		# 		return None
		# 	return (round(float(row['lat']),2), round(float(row['lon']),2))

		# ## Tabulates list of clients we probe at each PoP. Clients we probe are those within a certain distance.
		# client_pop_mapping_fn = os.path.join('cache', 'client_loc_to_pop_{}.pkl'.format(self.system))
		# if os.path.exists(client_pop_mapping_fn):
		# 	self.client_loc_to_pop, self.pop_to_client_loc = pickle.load(open(client_pop_mapping_fn, 'rb'))
		# else:
		# 	client_24_loc_d = list(csv.DictReader(open(os.path.join(DATA_DIR, "client_24_loc.csv"), 'r')))
		# 	locs = list(set([parse_row_loc(row) for row in client_24_loc_d]))
		# 	pop_locs = np.array([self.pop_to_loc[pop] for pop in self.pops])
		# 	self.client_loc_to_pop = {}
		# 	self.pop_to_client_loc = {}
		# 	for i,loc in enumerate(locs):
		# 		if loc is None: continue
		# 		closest_pop = np.argmin(np.linalg.norm(loc - pop_locs,axis=1))
		# 		if geopy.distance.geodesic(self.pop_to_loc[self.pops[closest_pop]], loc).km < 1000:
		# 			try:
		# 				self.pop_to_client_loc[self.pops[closest_pop]].append(loc)
		# 			except KeyError:
		# 				self.pop_to_client_loc[self.pops[closest_pop]] = [loc]
		# 			self.client_loc_to_pop[tuple(loc)] = self.pops[closest_pop]
		# 	pickle.dump([self.client_loc_to_pop, self.pop_to_client_loc], open(client_pop_mapping_fn,'wb'))

		## Natural client to PoP mapping for an anycast advertisement, learned from measurements
		self.pop_to_clients = {pop:[] for pop in self.pops}
		self.pop_to_clients_fn = os.path.join(DATA_DIR, 'client_to_pop.csv')
		if os.path.exists(self.pop_to_clients_fn):
			for row in open(self.pop_to_clients_fn, 'r'):
				pop,client = row.strip().split(',')
				self.pop_to_clients[pop].append(client)

		## clients that switch between pops, annoying
		self.multipop_clients_fn = os.path.join(DATA_DIR, 'multipop_clients.csv')
		self.multipop_clients = []
		if os.path.exists(self.multipop_clients_fn):
			for row in open(self.multipop_clients_fn, 'r'):
				client = row.strip()
				self.multipop_clients.append(client)

		## perhaps reachable addresses for clients
		# reachable_addr_metro_asn.kql, 5 days
		msft_data_d = list(csv.DictReader(open(os.path.join(DATA_DIR, 'reachable_addr_metro_asn_5d.csv'),'r')))
		self.all_client_addresses_from_msft = set([row['reachable_addr'] for row in msft_data_d \
			if ":" not in row['reachable_addr'] and row['reachable_addr'] != ''])
		print("{} UGs".format(len(set((row['metro'], row['asn']) for row in msft_data_d))))
		## get rid of these candidates since they're annoying
		self.dst_to_ug = {}
		self.ug_to_vol = {}
		for d in msft_data_d:
			ug = (d['metro'],d['asn'])
			if d['reachable_addr'] != "":
				self.dst_to_ug[d['reachable_addr']] = ug
			try:
				self.ug_to_vol[ug] += float(d['N'])
			except KeyError:
				self.ug_to_vol[ug] = float(d['N'])
		for fn in glob.glob(os.path.join(DATA_DIR, "hitlist_from_jiangchen", "*")):
			these_ips = set([row.strip() for row in open(fn,'r').readlines()])
			self.all_client_addresses_from_msft = self.all_client_addresses_from_msft.union(these_ips)
		self.all_client_addresses_from_msft = get_difference(self.all_client_addresses_from_msft, self.multipop_clients)
		## testing which addresses respond to ping, to focus on useable addresses
		self.addresses_that_respond_to_ping = {0:[],1:[]}
		self.addresses_that_respond_to_ping_fn = os.path.join(DATA_DIR, "addresses_that_respond_to_ping.csv")
		if os.path.exists(self.addresses_that_respond_to_ping_fn):
			with open(self.addresses_that_respond_to_ping_fn, 'r') as f:
				for row in f:
					# 1 -- responds, 0 -- doesn't
					ipadr,reach = row.strip().split(',')
					self.addresses_that_respond_to_ping[int(reach)].append(ipadr)
		# If an address has failed to respond once in history, exclude it
		self.addresses_that_respond_to_ping[1] = get_difference(self.addresses_that_respond_to_ping[1], 
				self.addresses_that_respond_to_ping[0])
		## we've measured from all destinations to these popps
		self.already_completed_popps_fn = os.path.join(CACHE_DIR, 'already_completed_popps.csv')

		self.default_ip = ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']


	### Route flap damping tracker, to ensure we don't advertise or withdraw too frequently
	def get_rfd_obj(self):
		ret = {}
		with open(self.rfd_track_fn, 'r') as f:
			for row in f:
				pref,t = row.strip().split(',')
				ret[pref] = int(t)
		for p in get_difference(self.available_prefixes, ret):
			## No data for this prefix yet, assume RFD timer isn't activated
			ret[p] = 0
		return ret

	def set_rfd_obj(self, rfd_obj):
		with open(self.rfd_track_fn, 'w') as f:
			for p, t in rfd_obj.items():
				f.write("{},{}\n".format(p,t))

	def check_rfd(self, cmd, **kwargs):
		### Check to make sure it hasn't been too soon since we last mucked with a prefix
		if "client/peering" not in cmd:
			## Unrelated command
			return
		if CAREFUL: # doesnt matter since we're not actually executing commands
			return
		if kwargs.get('override_rfd'): # manual override
			return
		if not ("announce" in cmd or "withdraw" in cmd): 
			# doesn't need to be rate limited
			return
		announce = "announce" in cmd
		if not announce:
			# only rate limit announcements
			return
		which_pref = [p for p in self.available_prefixes if p in cmd][0]
		rfd_time = self.get_rfd_obj()[which_pref]
		if time.time() - rfd_time < RFD_WAIT_TIME:
			t = RFD_WAIT_TIME - (time.time() - rfd_time) + 10
			print("Waiting {} minutes for RFD to die down.".format(t/60))
			time.sleep(t)

	def track_rfd(self, cmd, **kwargs):
		### We executed cmd which involved a prefix
		if "client/peering" not in cmd:
			## Unrelated command
			return
		announce = "announce" in cmd
		if not announce: 
			## dont rate limit withdrawals
			return
		which_pref = [p for p in self.available_prefixes if p in cmd][0]
		rfd_obj = self.get_rfd_obj()
		rfd_obj[which_pref] = int(time.time())
		self.set_rfd_obj(rfd_obj)

	def get_most_viable_prefix(self):
		### Get the prefix for which it's been the longest since we announced anything with it
		rfd_obj = self.get_rfd_obj()
		min_pref, min_t = None, None
		for pref,t in rfd_obj.items():
			if min_pref is None:
				min_pref = pref
				min_t = t
			if t < min_t:
				min_pref = pref
				min_t = t
		return min_pref

	#### End route flap damping trackers
	def _call(self, cmd, careful=False, **kwargs):
		print(cmd)
		self.check_rfd(cmd, **kwargs)
		if not careful:
			call(cmd, shell=True)
		self.track_rfd(cmd, **kwargs)

	def _check_output(self, cmd, careful=False, **kwargs):
		print(cmd)
		self.check_rfd(cmd, **kwargs)
		if not careful:
			return check_output(cmd,shell=True)
		self.track_rfd(cmd, **kwargs)

	def advertise_to_peers(self, pop, peers, pref, **kwargs):
			
		## check to make sure we're not advertising already
		cmd_out = self._check_output("sudo client/peering bgp adv {}".format(pop),careful=CAREFUL)
		if cmd_out is not None:
			if pref in cmd_out.decode():
				self.withdraw_from_pop(pop, pref)

		if self.system == 'peering':
			# advertise only to this peer
			self._call("sudo client/peering prefix announce -m {} -c 47065,{} {}".format( ### NEED TO UPDATE
				pop, self.peer_to_id[pop, peer], pref),careful=CAREFUL, **kwargs)
		elif self.system == 'vultr':
			#https://github.com/vultr/vultr-docs/tree/main/faq/as20473-bgp-customer-guide#readme
			community_str = " -c 20473,6000 " + " ".join(["-c 64699,{}".format(self.peer_to_id[pop, peer]) for peer in peers])
			self._call("sudo client/peering prefix announce -m {} {} {}".format(
				pop, community_str, pref),careful=CAREFUL, **kwargs)
			pass

	def advertise_to_popps(self, popps, pref):
		## Wrapper for advertise_to_peers, since advertisements are done on the command line by pop
		popps_by_pop = {}
		for pop,peer in popps:
			try:
				popps_by_pop[pop].append(peer)
			except KeyError:
				popps_by_pop[pop] = [peer]
		i=0
		for pop, peers in popps_by_pop.items():
			self.advertise_to_peers(pop, peers, pref, override_rfd=(i>0))
			i += 1

	def withdraw_from_pop(self, pop, pref):
		if self.system == 'peering':
			# advertise only to this peer
			self._call("sudo client/peering prefix withdraw -m {} -c 47065,{} {}".format(
				pop, self.peer_to_id[pop, peer], pref),careful=CAREFUL)
		elif self.system == 'vultr':
			# just do the opposite of advertise
			self._call("sudo client/peering prefix withdraw -m {} {}".format(
				pop, pref),careful=CAREFUL)

	def announce_anycast(self):
		pref = self.get_most_viable_prefix()
		for i, pop in enumerate(self.pops):
			self._call("sudo client/peering prefix announce -m {} {}".format(
				pop, pref),careful=CAREFUL,override_rfd=(i>0)) 
		return pref
	def withdraw_anycast(self, pref):
		for pop in self.pops:
			self._call("sudo client/peering prefix withdraw -m {} {}".format(
				pop, pref),careful=CAREFUL)

	def quickvultrtesting(self):
		### my home PC comes in through level3
		### so lets try hurricane electric and GTT in new york, as 6939 and 3257
		## switches my route to go through GTT, HE doesn't really work

		self.advertise_to_peers(['6939', '3257'])

	def get_advertisements_prioritized(self):
		### Returns a set of <prefix, popps> to advertise

		already_completed_popps = []
		if os.path.exists(self.already_completed_popps_fn):
			already_completed_popps = [tuple(row.strip().split(',')) for row in open(self.already_completed_popps_fn,'r')]

		if False:
			min_separation = 6000 # kilometers
			import geopy.distance 
			pop_clusters = []
			for pop, loc in self.pop_to_loc.items():
				found = False
				np.random.shuffle(pop_clusters)
				for pop_cluster in pop_clusters:
					conflict = False
					for p in pop_cluster:
						if geopy.distance.geodesic(self.pop_to_loc[p],loc).km < min_separation:
							conflict=True
							break
					if not conflict:
						pop_cluster.append(pop)
						found=True
						break
				if not found:
					pop_clusters.append([pop])
			print(pop_clusters)
		else:
			# partially manual
			pop_clusters = [['chicago', 'delhi', 'london', 'melbourne'], ['toronto', 'warsaw', 'silicon', 'mexico'], 
				['dallas', 'frankfurt', 'tokyo'], 
				['amsterdam', 'atlanta', 'bangalore', 'johannesburg', 'sydney'],
				['miami', 'mumbai', 'stockholm'], ['newyork', 'seoul', 'seattle','paris'],  
				['losangelas', 'madrid', 'saopaulo', 'singapore']]


		ret = []
		prefi = 0
		nprefs = len(self.available_prefixes)
		as_classifications_d = list(csv.DictReader(open(os.path.join(DATA_DIR, 'AS_Features_032022.csv'),'r')))
		as_classifications = {row['asNumber']: row for row in as_classifications_d}
		def popp_to_score(popp):
			pop, peer = popp			
			try:
				corresponding_classification = as_classifications[str(peer)]
				if corresponding_classification['ASCone'] == '': 
					return 0
				return int(float(corresponding_classification['ASCone']))
			except KeyError:
				pass
			return 0

		all_popps = [(pop,peer) for pop in self.peers for peer in self.peers[pop]]
		popps_to_focus_on = get_difference(all_popps, already_completed_popps)
		while len(popps_to_focus_on) > 0:
			### At each iteration, get the next-best set of advertisements to conduct
			popps_to_focus_on = get_difference(all_popps, already_completed_popps)
			effective_peers = {}
			for pop,peer in popps_to_focus_on:
				try:
					effective_peers[pop].append(peer)
				except KeyError:
					effective_peers[pop] = [peer]

			pop_scores = {pop: sum([popp_to_score((pop,peer)) for peer in effective_peers[pop]]) for pop in effective_peers}
			cluster_scores = {i:sum(pop_scores.get(pop,0) for pop in pop_cluster) for i,pop_cluster in enumerate(pop_clusters)}
			# sort cluster -> popps within each cluster
			success = False
			for clusteri,score in sorted(cluster_scores.items(), key = lambda el : -1 * el[1]):
				cluster = pop_clusters[clusteri]
				intrapop_scores = {pop: {peer: popp_to_score((pop,peer)) for peer in effective_peers.get(pop,[])} \
					for pop in cluster}
				sorted_intrapop_scores = {pop: sorted(intrapop_scores[pop].items(), key = lambda el : -1 * el[1]) for \
					pop in intrapop_scores}

				max_l = np.max([len(intrapop_scores[pop]) for pop in cluster])
				for i in range(max_l):
					this_pref_adv = []
					pref = self.available_prefixes[prefi % nprefs]
					already_using_peers = {}
					for pop in sorted_intrapop_scores:
						try:
							tmpi = copy.copy(i)
							while True: ### Enforce that we don't advertise the same prefix to the same peer in multiple locs
								peer,score = sorted_intrapop_scores[pop][i]
								try:
									already_using_peers[peer]
									i += 1
									continue
								except KeyError:
									already_using_peers[peer] = None
									break
							i = tmpi
							this_pref_adv.append((pref, pop, peer, score))	
							already_completed_popps.append((pop,peer))
						except IndexError:
							pass
					if len(this_pref_adv) > 0:
						ret.append(this_pref_adv)
						prefi += 1
						success = True
						break
				if success:
					break
		return ret


	def load_anycast_latencies(self):
		anycast_latencies = {}
		anycast_latency_fn = os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(self.system))
		if os.path.exists(anycast_latency_fn):
			for row in open(anycast_latency_fn, 'r'):
				client_network, latency = row.strip().split(',')
				anycast_latencies[client_network] = float(latency)

		return anycast_latencies

	def measure_vpn_lats(self):
		## Remeasure VPN lats
		vpn_lat_results = measure_latency_ips([self.vpn_ips[pop] for pop in self.pops_list])
		for pop,dst in zip(self.pops_list, vpn_lat_results):
			self.pop_vpn_lats[pop] = np.min(vpn_lat_results[dst])
		print("New VPN lats: {}".format(self.pop_vpn_lats))
		pickle.dump(self.pop_vpn_lats, open(self.pop_vpn_lats_fn,'wb'))

	def measure_prepainter(self):
		""" Conducts anycast measurements and per-ingress measurements, for input into a PAINTER calculation."""
		
		anycast_latencies = self.load_anycast_latencies()
		every_client_of_interest = self.get_reachable_clients()
		still_need_anycast = get_difference(every_client_of_interest, anycast_latencies)

		if False:#len(still_need_anycast) > 0: ### Should usually have this check turned on
			print(len(still_need_anycast))
			print(still_need_anycast[0:100])
			exit(0)
			self.measure_vpn_lats()
			### First, get anycast pop
			# ping everything from one pop
			print("getting client to pop ")
			pref = self.announce_anycast()
			measurement_src_ip = pref_to_ip(pref)
			pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
			for pop in self.pops:
				lats_results = pw.run([measurement_src_ip], 
					[self.pop_to_intf[pop]], still_need_anycast)[measurement_src_ip]
				break
			print("done getting client to pop")

			with open(self.pop_to_clients_fn, 'a') as f:
				for dst, meas in lats_results.items():
					pop = list(get_difference(([m['pop'] for m in meas]), [None]))
					if len(pop) > 1:
						with open(self.multipop_clients_fn, 'a') as f2:
							f2.write(dst + "\n")
						continue
					elif len(pop) == 0:
						continue
					pop = pop[0]
					f.write("{},{}\n".format(pop,dst))
					self.pop_to_clients[pop].append(dst)

			### Second, get anycast latency
			for pop in self.pops:
				print("Getting anycast latency for pop {}".format(pop))
				this_pop_clients = self.pop_to_clients[pop]
				lats_results = pw.run([measurement_src_ip] ,[self.pop_to_intf[pop]], 
						get_intersection(still_need_anycast, this_pop_clients),
						remove_bad_meas=True)[measurement_src_ip]
				minimum_latency = self.pop_vpn_lats[pop]
				print("PoP : {}, minimum latency : {}".format(pop, minimum_latency))
				with open(anycast_latency_fn,'a') as f:
					for client_dst, meas in lats_results.items():
						rtts = []
						for m in meas:
							if m['pop'] != pop or m['rtt'] is None:
								continue
							rtts.append(m['rtt'])
						if len(rtts) > 0:
							rtt = np.min(rtts)
							f.write("{},{}\n".format(client_dst,rtt - minimum_latency))
							anycast_latencies[client_dst] = rtt
						else:
							rtt = -1	
							f.write("{},{}\n".format(client_dst,rtt))
			
			### These just dont work for whatever reason
			still_need_anycast = get_difference(still_need_anycast, list(anycast_latencies))
			with open(self.addresses_that_respond_to_ping_fn, 'a') as f:
				for dst in still_need_anycast:
					f.write("{},{}\n".format(dst,0))

			self.withdraw_anycast(pref)

		# all_vol = sum(list(self.ug_to_vol.values()))
		# have_vol = 0
		# counted_ug = {}
		# for dst,lat in anycast_latencies.items():
		# 	ug = self.dst_to_ug[dst]
		# 	try:
		# 		counted_ug[ug]
		# 	except KeyError:
		# 		have_vol += self.ug_to_vol[ug]
		# 		counted_ug[ug] = None

		# print("Have anycast latency for {} percent of volume".format(have_vol * 100 / all_vol))
			
		### Next, get latency from all users to all popps
		prefix_popps = self.get_advertisements_prioritized()
		print(prefix_popps[0:5])
		popp_lat_fn = os.path.join(CACHE_DIR, "{}_ingress_latencies_by_dst.csv".format(self.system))
		## example adv_set [('184.164.240.0/24', 'newyork', '3356', 48782), 
			# ('184.164.240.0/24', 'seoul', '2914', 20577), ('184.164.240.0/24', 'seattle', '3356', 48782), 
			# ('184.164.240.0/24', 'paris', '3356', 48782)],
		# just measure to every client for the hell of it
		every_client_of_interest = [dst for dst,lat in anycast_latencies.items() if lat != -1]

		n_prefs = len(self.available_prefixes)
		n_adv_rounds = int(np.ceil(len(prefix_popps) / n_prefs))
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		for adv_round_i in range(n_adv_rounds):
			self.measure_vpn_lats()
			adv_sets = prefix_popps[n_prefs * adv_round_i: n_prefs * (adv_round_i+1)]
			print(adv_sets)
			srcs = []
			pref_set = []
			popps_set = []
			pops_set = {}
			adv_set_to_taps = {}
			for i, adv_set in enumerate(adv_sets):
				pref = self.get_most_viable_prefix()
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
				popps = [(pop,peer) for _,pop,peer,_ in adv_set]
				popps_set.append(popps)
				self.advertise_to_popps(popps, pref)

				pops = list(set([popp[0] for popp in popps]))
				pops_set[i] = pops
				adv_set_to_taps[i] = [self.pop_to_intf[pop] for pop in pops]
			max_n_pops = max(len(v) for v in adv_set_to_taps.values())	
			for pop_iter in range(max_n_pops):
				srcs_set = []
				taps_set = []
				peers_set = []
				this_pops_set = []
				for adv_set_i in range(n_prefs):
					try:
						# different prefixes have different numbers of PoPs involved
						# measurements are done per PoP, so this is a bit awkward
						taps_set.append(adv_set_to_taps[adv_set_i][pop_iter])
						srcs_set.append(srcs[adv_set_i])
						this_pop = pops_set[adv_set_i][pop_iter]
						this_pops_set.append(this_pop)
						peers_set.append([popp[1] for popp in popps_set[adv_set_i] if popp[0] \
							== this_pop][0])
					except IndexError:
						pass
				print("PoP iter {}".format(pop_iter))
				print(srcs_set)
				print(taps_set)
				print(this_pops_set)
				print(peers_set)
				lats_results = pw.run(srcs_set, taps_set, every_client_of_interest,
					remove_bad_meas=True)
				for src,pop,peer in zip(srcs_set, this_pops_set, peers_set):
					with open(popp_lat_fn,'a') as f:
						for client_dst, meas in lats_results[src].items():
							rtts = []
							for m in meas:
								if m['pop'] != pop or m['rtt'] is None:
									continue
								rtts.append(m['rtt'])
							if len(rtts) > 0:
								rtt = np.min(rtts)
								f.write("{},{},{},{}\n".format(client_dst,pop,peer,rtt - self.pop_vpn_lats[pop]))
				exit(0)
			for pref, popps in zip(pref_set, popps_set):
				for pop in set([popp[0] for popp in popps]):
					self.withdraw_from_pop(pop, pref)
			with open(self.already_completed_popps_fn, 'a') as f:
				for popps in popps_set:
					for pop,peer in popps:
						f.write("{},{}\n".format(pop,peer))

	def conduct_painter(self):

		# load advertisement
		# for prefix in budget (rate-limited)
		# popp in advertisemnet
		# conduct the advertisement
		# measure from every user
		# save perfs

		def get_advs_by_budget(ranked_advs):
			# ranked advs is a dict budget -> all advertisements up to budget
			# Separate advertisements by prefix to get pfx -> advs this prefix
			in_last = None
			advs_by_pid = {}
			budgets = sorted(list(ranked_advs))
			lens = [len(ranked_advs[b]) for b in budgets]
			start = 0
			for i,b in enumerate(budgets): # awkward but it enables backwards compatibility
				advs_by_pid[b] = ranked_advs[b][start:lens[i]]
				start = lens[i]
			return advs_by_pid

		ranked_advs_fn = os.path.join(DATA_DIR, 'ranked_advs.pkl')
		if not os.path.exists(ranked_advs_fn):
			all_ranked_advs = pickle.load(open('/home/tom/Transit/code/koch/ingress_opt/cache/painter_adv_cache/ranked_advs.pkl','rb'))
			adv_soln = all_ranked_advs['all_time-3000-painter_v4-dont_ignore_anycast-vultr']['painter_v4']
			pickle.dump(adv_soln, open(ranked_advs_fn,'wb'))
		else:
			adv_soln = pickle.load(open(ranked_advs_fn,'rb'))

		advs_by_budget = get_advs_by_budget(adv_soln)
		print(advs_by_budget)
		popp_lat_fn = os.path.join(CACHE_DIR, "{}_adv_latencies.csv".format(self.system))
		anycast_latencies = self.load_anycast_latencies()
		every_client_of_interest = [dst for dst,lat in anycast_latencies.items() if lat != -1]
		for budget, popps in advs_by_budget:
			pref = self.get_most_viable_prefix()
			measurement_src_ip = pref_to_ip(pref)
			pw = Pinger_Wrapper(measurement_src_ip, self.pops, self.pop_to_intf)
			self.advertise_to_popps(popps, pref)
			for pop in set([popp[0] for popp in popps]):
				lats_results = pw.run(self.pop_to_intf[pop], every_client_of_interest,
					remove_bad_meas=True)
				with open(popp_lat_fn,'a') as f:
					for client_dst, meas in lats_results.items():
						rtts = []
						for m in meas:
							if m['pop'] != pop or m['rtt'] is None:
								continue
							rtts.append(m['rtt'])
						if len(rtts) > 0:
							rtt = np.min(rtts)
							f.write("{},{},{},{}\n".format(budget,client_dst,pop,rtt - self.pop_vpn_lats[pop]))
			for pop in set([popp[0] for popp in popps]):
				self.withdraw_from_pop(pop, pref)
			with open(popp_lat_fn,'a') as f:
				# record which popps we're talking about
				popps_str = ",".join(["-".join(popp) for popp in popps])
				f.write("newadv,{},{}\n".format(budget,popps_str))


	def calculate_latency(self):
		# # Get latency from VM to clients
		# cp = Custom_Pinger(self.default_ip, 'eth0', client_dsts)
		# cp.run()
		# lats = cp.get_finished_meas()
		# self.parse_latencies(lats, peer='VM')

		# # Get latency from mux to VM
		# pref = self.announce_anycast()
		# cp = Customer_Pinger(self.unicast_addr, self.pop_intf, [self.unicast_addr])
		# lats = cp.get_finished_meas()
		# self.parse_latencies(lats, peer='VM')
		# self.withdraw_anycast(pref)

		# Get latencies (and reachabilities) from all clients to all peers
		for peer in self.peers[self.pop]:
			client_dsts = self.get_pop_to_clients(self.pop, peer=peer)
			if client_dsts == []: 
				print("Done with peer {}, skipping".format(peer))
				continue
			print("Measuring to {} dsts, peer : {}.".format(len(client_dsts), peer))
			self.advertise_to_peer(peer)
			ta = time.time() 

			cp = Custom_Pinger(self.unicast_addr, self.pop_intf, client_dsts)
			cp.run()

			lats = cp.get_finished_meas()

			print("Done measuring, {} results.".format(len(lats)))


			self.parse_latencies(lats, peer=peer)
			# We don't need to explicitly withdraw due to the nature of the advertisement

			tslp = np.max([ min_t_between - (time.time() - ta), 1])
			print("Sleeping for {} seconds.".format(tslp))
			time.sleep(tslp)

		self.withdraw_from_peer(peer)
	
	def pairwise_preferences(self):
		# Get pairwise preferences
		for peer_i in self.peers[self.pop]:
			self.advertise_to_peer(peer_i)
			for peer_j in self.peers[self.pop]:
				if peer_j == peer_i: break

				# get set of clients for whom peer i and peer j are both reachable
				client_dsts = get_intersection(self.reachable_dsts[self.pop,peer_i], self.reachable_dsts[self.pop, peer_j])
				self.advertise_to_peer(peer_j)
				taj = time.time()

				# Measure to these clients
				cp = Customer_Pinger(self.unicast_addr, self.pop_intf, client_dsts)
				cp.run()
				lats = cp.get_finished_meas()

				# note which peer is preferred
				self.parse_preferences(lats)

				while time.time() - taj < min_t_between:
					tslp = time.time() - (taj + min_t_between) + 1
					time.sleep(tslp)
				self.withdraw_from_peer(peer_j)

			self.withdraw_from_peer(peer_i)

	def parse_latencies(self, lats, peer=None):
		if peer is not None and peer != "VM":
			peer_macs = []
			for meas in lats.values():
				if meas['peer_mac'] is not None:
					for pm in meas['peer_mac']:
						peer_macs.append(pm)
			peer_macs = list(set(peer_macs))
			if peer_macs != []: 
				if len(peer_macs) > 1:
					print("WARNING -- only 1 peer active but 2 macs registered: {}".format(peer_macs))
				for pm in peer_macs:
					self.peer_macs[self.pop][pm] = peer
				with open(self.peer_mac_fn, 'a') as f:
					for pm in peer_macs:
						f.write("{}\t{}\t{}\n".format(self.pop, pm, peer))

		with open(os.path.join(DATA_DIR, "client_lats.csv"), 'a') as f:
			for client_dst in lats:
				rtt,peer_mac = lats[client_dst]['rtt'], lats[client_dst]['peer_mac']
				if peer_mac is not None:
					peer_mac = peer_mac[0]
				if rtt != -1:
					if peer != "VM":
						_peer = self.peer_macs[self.pop][peer_mac]
					else:
						_peer = "VM"
					try:
						self.reachable_dsts[self.pop, _peer].append(client_dst)
					except KeyError:
						self.reachable_dsts[self.pop, _peer] = [client_dst]
				f.write("{}\t{}\t{}\n".format(client_dst, peer, rtt))

	def parse_preferences(self, lats, peer_i, peer_j):
		with open(os.path.join(DATA_DIR, "client_preferences.csv"), 'a') as f:
			for client_dst, meas in lats.items():
				peer = self.peer_macs[self.pop][meas['peer_mac'][0]]
				if peer == peer_i:
					bit = '0'
				else:
					bit = '1'
				f.write("{}\t{}\t{}\t{}\n".format(client_dst, peer_i, peer_j, bit))

	def get_reachable_clients(self, peer=None):
		dsts = list(self.all_client_addresses_from_msft)
		if peer is not None and peer in self.measured_to:
			dsts = get_difference(dsts, self.measured_to[peer])

		## see which ones respond to ping
		# check if we've already measured responsiveness
		already_know_responsiveness = self.addresses_that_respond_to_ping[0] + self.addresses_that_respond_to_ping[1]
		dont_know_responsiveness = get_difference(dsts, already_know_responsiveness)
		if len(dont_know_responsiveness) > 0:
			responsive_dsts = check_ping_responsive(dont_know_responsiveness) + get_intersection(dsts, self.addresses_that_respond_to_ping[1])
		else:
			responsive_dsts = get_intersection(dsts, self.addresses_that_respond_to_ping[1])
		responsive_dsts = list(set(responsive_dsts))
		self.addresses_that_respond_to_ping[1] += get_intersection(dsts, responsive_dsts)
		self.addresses_that_respond_to_ping[0] += get_difference(dont_know_responsiveness, responsive_dsts)
		with open(self.addresses_that_respond_to_ping_fn, 'w') as f:
			for i in [0,1]:
				for dst in self.addresses_that_respond_to_ping[i]:
					f.write("{},{}\n".format(dst,i))

		return responsive_dsts

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="calculate_latency",
		choices=['calculate_latency','pairwise_preferences','quickvultrtesting',
		'measure_prepainter', 'conduct_painter'])
	parser.add_argument('--system', required=True, 
		choices=['peering','vultr'],help="Are you using PEERING or VULTR?")
	args = parser.parse_args()

	pp = Peering_Pinger(args.system,args.mode)
	pp.run()