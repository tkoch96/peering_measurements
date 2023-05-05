import os, re, socket, csv, time, json, numpy as np, pickle, geopy.distance, copy, glob, tqdm
from config import *
from helpers import *
from subprocess import call, check_output
import subprocess
from generic_measurement_utils import AS_Utils_Wrapper

class Deployment_Measure_Wrapper():
	def __init__(self, **kwargs):

		self.utils = AS_Utils_Wrapper()
		self.utils.check_load_as_rel()
		self.utils.update_cc_cache()

		self.addresses_that_respond_to_ping_fn = os.path.join(DATA_DIR, "addresses_that_respond_to_ping.csv")
		all_muxes_str = check_output("sudo client/peering openvpn status", shell=True).decode().split('\n')
		if self.system == "peering":
			self.pops = ['amsterdam01']
		elif self.system == "vultr":
			self.pops = []
			for row in all_muxes_str:
				if row.strip() == '': continue
				self.pops.append(row.split(' ')[0])
		self.pops = sorted(self.pops)
		self.pop_to_intf = {}
		for pop in self.pops:
			this_mux_str = [mux_str for mux_str in all_muxes_str if pop in mux_str]
			assert len(this_mux_str) == 1
			this_mux_str = this_mux_str[0]
			self.pop_to_intf[pop] = "tap" + re.search("tap(\d+)", this_mux_str).group(1)
		
		### prefixes we can conduct experiments with
		self.available_prefixes = ["184.164.238.0/24","184.164.239.0/24","184.164.240.0/24", 
			"184.164.241.0/24","184.164.242.0/24","184.164.243.0/24"]
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

		### All connections (ingresses), learned from BGP routes
		self.popps = []
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
			self.popps.append((pop,peer))
		for pop in self.peers:
			self.peers[pop] = list(set(self.peers[pop])) # remove dups

		### Providers, learned from BGP routes
		self.provider_popps = []
		for row in open(os.path.join(CACHE_DIR, '{}_provider_popps.csv'.format(self.system)),'r'):
			self.provider_popps.append(tuple(row.strip().split(',')))

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

		self.limit_pops = False
		self.include_pops = ['miami', 'atlanta','newyork','dallas','losangelas']
		if self.limit_pops:
			self.pops_list = get_intersection(self.pops_list, self.include_pops)
			self.peers = {pop:self.peers[pop] for pop in self.include_pops}
			self.popps = [(pop,peer) for pop,peer in self.popps if pop in self.include_pops]
			self.pop_to_loc = {pop:self.pop_to_loc[pop] for pop in self.include_pops}
			self.pops = self.include_pops
		print("Working with PoPs : {}".format(self.pops))

		if kwargs.get('quickinit', False): return

		## Natural client to PoP mapping for an anycast advertisement, learned from measurements
		self.pop_to_clients = {pop:[] for pop in self.pops}
		self.pop_to_clients_fn = os.path.join(DATA_DIR, 'client_to_pop.csv')
		self.client_to_pop = {}
		if os.path.exists(self.pop_to_clients_fn):
			for row in open(self.pop_to_clients_fn, 'r'):
				client,pop = row.strip().split(',')
				if self.limit_pops:
					if pop not in self.include_pops:
						continue
				self.pop_to_clients[pop].append(client)
				self.client_to_pop[client] = pop
		## clients that switch between pops, annoying
		self.multipop_clients_fn = os.path.join(DATA_DIR, 'multipop_clients.csv')
		self.multipop_clients = []
		if os.path.exists(self.multipop_clients_fn):
			for row in open(self.multipop_clients_fn, 'r'):
				client = row.strip()
				self.multipop_clients.append(client)

		load_msft = False
		self.all_client_addresses_from_msft = set()
		if load_msft:
			## perhaps reachable addresses for clients
			self.dst_to_ug = {}
			self.ug_to_vol = {}
			for row in tqdm.tqdm(csv.DictReader(open(os.path.join(DATA_DIR,
				'client24_metro_vol_20220620.csv'),'r')), desc="Loading Microsoft user data"):
				ug = (row['metro'], row['asn'])
				try:
					self.ug_to_vol[ug] += float(row['N'])
				except KeyError:
					self.ug_to_vol[ug] = float(row['N'])
				self.dst_to_ug[row['client_24']] = ug
				if np.random.random() > .9999:
					break
			# reachable_addr_metro_asn.kql, 5 days
			self.all_client_addresses_from_msft = {}
			ug_to_loc = {}
			for row in tqdm.tqdm(csv.DictReader(open(os.path.join(DATA_DIR, 
				'reachable_addr_metro_asn_5d.csv'),'r')),desc="Loading reachable addresses."):
				ug = (row['metro'],row['asn'])
				if ":" not in row['reachable_addr'] and row['reachable_addr'] != '':
					self.all_client_addresses_from_msft[row['reachable_addr']] = None
					self.dst_to_ug[ip32_to_24(row['reachable_addr'])] = ug
				if ":" not in row['reachable_addr']:
					ug_to_loc[ug] = (float(row['lat']), float(row['lon']))
					try:
						self.ug_to_vol[ug] += float(row['N'])
					except KeyError:
						self.ug_to_vol[ug] = float(row['N'])
					
			self.all_client_addresses_from_msft = set(self.all_client_addresses_from_msft)

		for fn in tqdm.tqdm(glob.glob(os.path.join(DATA_DIR, "hitlist_from_jiangchen", "*")),
			desc="Loading hitlists from JC"):
			if self.limit_pops:
				pop = re.search("anycast\_ip\_(.+)\.txt",fn).group(1)
				if pop not in self.include_pops:
					continue
			these_ips = set([row.strip() for row in open(fn,'r').readlines()])
			self.all_client_addresses_from_msft = self.all_client_addresses_from_msft.union(these_ips)
		self.all_client_addresses_from_msft = get_difference(self.all_client_addresses_from_msft, self.multipop_clients)
		## we've measured from all destinations to these popps
		self.already_completed_popps_fn = os.path.join(CACHE_DIR, 'already_completed_popps.csv')

		
		if load_msft:
			self.check_load_addresses_that_respond_to_ping()
			ugs_in_ping = {}

			for ip in self.addresses_that_respond_to_ping[1]:
				ntwrk = ip32_to_24(ip)
				try:
					ugs_in_ping[self.dst_to_ug[ntwrk]] = None
				except KeyError:
					pass
			print("{} UGs thus far respond to ping, {} pct of volume".format(len(ugs_in_ping),
				sum(self.ug_to_vol[ug] for ug in ugs_in_ping) * 100.0 / sum(list(self.ug_to_vol.values()))))
			with open(os.path.join(CACHE_DIR, '{}_used_ug_locs.csv'.format(self.system)), 'w') as f:
				for ug in ug_to_loc:
					lat,lon = ug_to_loc[ug]
					f.write("{},{},{}\n".format(lat,lon,self.ug_to_vol[ug]))

		### Get AS to org mapping and vice versa for compatibility with PAINTER pipeline
		peer_to_org_fn = os.path.join(CACHE_DIR, 'vultr_peer_to_org.csv')
		self.peer_to_org,self.org_to_peer = {}, {}
		for row in open(peer_to_org_fn,'r'):
			peer,org = row.strip().split(',')
			if int(org) >= 1e6:
				org = int(org)
			else:
				org = org.strip()
			try:
				self.org_to_peer[org].append(peer.strip())
			except KeyError:
				self.org_to_peer[org] = [peer.strip()]
			self.peer_to_org[peer] = org
		for org in self.org_to_peer:
			self.org_to_peer[org] = list(set(self.org_to_peer[org]))

		self.maximum_inflation = kwargs.get('maximum_inflation', 1)

		if load_msft:
			## Free up memory
			del self.dst_to_ug
			del self.ug_to_vol

	def pop_to_close_clients(self, pop):
		### measure to any client whose catchment is within some number of km
		try:
			self.pop_to_close_clients_cache
		except AttributeError:
			self.pop_to_close_clients_cache = {}
		try:
			return self.pop_to_close_clients_cache[pop]
		except KeyError:
			pass
		ret = set()
		for p in self.pop_to_loc:
			if geopy.distance.geodesic(self.pop_to_loc[p],self.pop_to_loc[pop]).km < 5000:
				ret = ret.union(self.pop_to_clients[p])
		self.pop_to_close_clients_cache[pop] = ret
		return ret

	### Route flap damping tracker, to ensure we don't advertise or withdraw too frequently
	def get_rfd_obj(self):
		ret = {}
		if os.path.exists(self.rfd_track_fn):
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

	def advertise_to_pop(self, pop, pref, **kwargs):
		## Advertises to all peers at a single pop
		self._call("sudo client/peering prefix announce -m {} {}".format(
				pop, pref),careful=CAREFUL,**kwargs) 

	def withdraw_from_pop(self, pop, pref):
		if self.system == 'peering':
			# advertise only to this peer
			self._call("sudo client/peering prefix withdraw -m {} -c 47065,{} {}".format(
				pop, self.peer_to_id[pop, peer], pref),careful=CAREFUL)
		elif self.system == 'vultr':
			# just do the opposite of advertise
			self._call("sudo client/peering prefix withdraw -m {} {}".format(
				pop, pref),careful=CAREFUL)

	def announce_anycast(self, pref=None):
		if pref is None:
			pref = self.get_most_viable_prefix()
		for i, pop in enumerate(self.pops):
			self._call("sudo client/peering prefix announce -m {} {}".format(
				pop, pref),careful=CAREFUL,override_rfd=(i>0)) 
		return pref
	def withdraw_anycast(self, pref):
		for pop in self.pops:
			self._call("sudo client/peering prefix withdraw -m {} {}".format(
				pop, pref),careful=CAREFUL)

	def get_condensed_targets(self, ips):
		"""Takes list of IP addresses, returns one IP address per UG. If we can't map an IP address
			to a UG we include it anyway."""
		print("Loading condensed target data")

		## If you want to use msft data, could do that
		# ip_ug_d = list(csv.DictReader(open(os.path.join(DATA_DIR, 'client24_metro_vol_20220620.csv'),'r')))
		# ip_to_ug = {row['client_24']: (row['metro'], row['asn']) for row in ip_ug_d}
		used_ugs = {}
		ret = []

		for ip in tqdm.tqdm(ips, desc="Condensing targets..."):
			ntwrk = ip32_to_24(ip)
			try:
				used_ugs[ntwrk]
			except KeyError:
				used_ugs[ntwrk] = None
				ret.append(ip)
		print("Condensed {} targets to {} targets.".format(len(ips), len(ret)))
		return ret

	def get_advertisements_prioritized(self, exclude_providers=False):
		### Returns a set of <prefix, popps> to advertise

		already_completed_popps = []
		if os.path.exists(self.already_completed_popps_fn):
			already_completed_popps = [tuple(row.strip().split(',')) for row in open(self.already_completed_popps_fn,'r')]


		ret = []
		prefi = 0
		nprefs = len(self.available_prefixes)
		as_classifications_d = list(csv.DictReader(open(os.path.join(DATA_DIR, 'AS_Features_032022.csv'),'r')))
		as_classifications = {row['asNumber']: row for row in as_classifications_d}
		def popp_to_score(popp):
			pop, peer = popp			
			if int(peer) > 60000:
				### TODO -- implement larger ASNs
				return -10
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

		if exclude_providers:
			popps_to_focus_on = get_difference(all_popps, self.provider_popps)

		popps_to_focus_on = get_intersection(popps_to_focus_on, self.popp_to_clients) # focus on popps with clients
		# TODO -- implement larger ASNs
		popps_to_focus_on = [popp for popp in popps_to_focus_on if int(popp[1]) <= 65000]

		if self.limit_pops:
			popps_to_focus_on = [(pop,peer) for pop,peer in popps_to_focus_on if pop in self.include_pops]

		t_start = time.time()
		n_gotten = 0
		while len(popps_to_focus_on) > 0:
			### At each iteration, get the next-best set of advertisements to conduct
			
			popps_to_focus_on = get_difference(popps_to_focus_on, already_completed_popps)
			ranked_popps = sorted(popps_to_focus_on, key = lambda el : popp_to_score(el))

			print("{} left to assign".format(len(popps_to_focus_on)))
			if n_gotten > 0:
				print("{} s per assignment".format(round((time.time() - t_start) / n_gotten,2)))

			this_set = [] # holds popps for this adv round
			n_per_pop = {}
			while True:
				# see if we can find something to add to the set
				found=False
				n_fails = 0
				for p in get_difference(ranked_popps,this_set):
					if p in self.provider_popps:
						## Trivially overlaps with everything
						this_set = [p]
						found = False
						break
					valid = True
					try:
						if n_per_pop[p[0]] >= 10:
							continue
					except KeyError:
						pass
					this_clients_set = self.popp_to_clientasn.get(p,[])
					if len(this_clients_set) == 0:
						continue 
					for entry in this_set:
						compare_clients_set = self.popp_to_clientasn.get(entry,[])
						if len(compare_clients_set) == 0:
							continue
						maxi = np.maximum(len(this_clients_set), len(compare_clients_set))
						arr1,arr2 = {},{}
						for i in range(maxi):
							try:
								this_el = this_clients_set[i]
								arr1[this_el] = None
							except IndexError:
								pass
							try:
								compare_el = compare_clients_set[i]
								arr2[compare_el] = None
							except IndexError:
								pass
							try:
								arr2[this_el]
								valid = False
								break
							except KeyError:
								pass
							try:
								arr1[compare_el]
								valid = False
								break
							except KeyError:
								pass
						if not valid:
							break
					if valid:
						this_set.append(p)
						try:
							n_per_pop[p[0]] += 1
						except KeyError:
							n_per_pop[p[0]] = 1

						found = True
						n_fails = 0
						break
					else:
						n_fails += 1
						# shortcut -- assume that if it takes enough iters then there just won't be anything else
						if n_fails == 20: 
							break
				if not found: 
					# cant find anything else to add
					break
			if len(this_set) > 0:
				print(this_set)
				already_completed_popps = already_completed_popps + this_set
				n_gotten += len(this_set)
				ret.append(this_set)
			else:
				break
		return ret

	def check_construct_client_to_peer_mapping(self, forceparse=False):
		### Goal is to create client to popps mapping (so that we can create popp to clients mapping)
		import pytricia
		client_to_popps_fn = os.path.join(CACHE_DIR, 'client_to_popps.csv')
		cust_cone_fn = os.path.join(CACHE_DIR, '{}_cust_cone.pkl'.format(self.system))
		if not os.path.exists(client_to_popps_fn) or forceparse:
			self.network_to_popp = pytricia.PyTricia()
			providers = {}
			cust_cone = {}
			for fn in glob.glob(os.path.join(TMP_DIR, '*routes.txt')):
				pop = re.search("\/(.+)\_routes",fn).group(1)
				if self.limit_pops:
					if pop not in self.include_pops:
						continue
				print("Parsing {} routes".format(pop))
				with open(fn, 'r') as f:
					for row in f:
						row = row.strip()
						if 'via' in row:
							pref = row.split(' ')[0]
						elif 'as_path' in row:
							as_path = row.split(' ')[1:]	
							first_hop = None
							for as_hop in as_path:
								if as_hop == "": continue
								as_hop = int(as_hop)
								if (as_hop >= 64512 and as_hop <= 65534) or as_hop == 20473:
									continue
								first_hop = as_hop
								break
							if not first_hop: continue
							peer = str(first_hop)
							try:
								self.network_to_popp[pref].append((pop,peer))
							except KeyError:
								self.network_to_popp[pref] = [(pop,peer)]
							try:
								providers[peer]
							except KeyError:
								asn_pref = self.utils.parse_asn(pref)
								if asn_pref is None: continue
								try:
									cust_cone[peer][asn_pref] = None
								except KeyError:
									cust_cone[peer] = {asn_pref:None}
						elif '(20473,100)' in row:
							## provider
							provider = row.split(' ')[2]
							provider = provider.split(',')[1].replace(')','').strip()
							providers[provider] = None
			popps = list(set(popp for pref in self.network_to_popp for popp in self.network_to_popp[pref]))
			for asn in cust_cone:
				cust_cone[asn] = list(cust_cone[asn])
			### append the routeviews prefixes
			## this is really expensive memory-wise
			# for pop,peer in popps:
			# 	all_asns = cust_cone.get(peer,[])
			# 	for asn in all_asns:
			# 		prefs = self.utils.routeviews_asn_to_pref.get(asn)
			# 		if prefs is not None:
			# 			for pref in prefs:
			# 				try:
			# 					self.network_to_popp[pref].append((pop,peer))
			# 				except KeyError:
			# 					self.network_to_popp[pref] = [(pop,peer)]
			for pref in self.network_to_popp:
				self.network_to_popp[pref] = list(set(self.network_to_popp[pref]))

			providers = list(providers)
			provider_popps = list(set([popp for popp in popps if popp[1] in providers]))
			with open(os.path.join(CACHE_DIR, '{}_provider_popps.csv'.format(self.system)),'w') as f:
				for pop,peer in provider_popps:
					f.write("{},{}\n".format(pop,peer))
			for ntwrk in self.network_to_popp:
				self.network_to_popp[ntwrk] = list(set(self.network_to_popp[ntwrk]))
			clients = self.get_reachable_clients(limit=False)
			print("Writing client to popps data")
			with open(client_to_popps_fn ,'w') as f:
				for client in clients:
					ntwrk = self.network_to_popp.get_key(client)
					if ntwrk is None:
						continue
					this_client_popps = set(self.network_to_popp.get(ntwrk))
					while ntwrk is not None:
						this_client_popps = this_client_popps.union(set(self.network_to_popp.get(ntwrk)))
						ntwrk = self.network_to_popp.parent(ntwrk)
					this_client_popps = get_difference(this_client_popps, provider_popps) # trivial
					if len(this_client_popps) > 0:
						popps_str = "-".join([popp[0]+"|"+popp[1]  for popp in this_client_popps])
						f.write("{},{}\n".format(client,popps_str))
			del clients
			del self.network_to_popp
			del self.addresses_that_respond_to_ping
			pickle.dump(cust_cone, open(cust_cone_fn,'wb'))
			print("Done writing client to popps data")

		cust_cone = pickle.load(open(cust_cone_fn,'rb'))
		self.popp_to_clients = {}
		self.provider_popps = []
		for row in open(os.path.join(CACHE_DIR, '{}_provider_popps.csv'.format(self.system)),'r'):
			self.provider_popps.append(tuple(row.strip().split(',')))
		for row in tqdm.tqdm(open(client_to_popps_fn, 'r'),desc="Loading client to popps."):
			client, popps_str = row.strip().split(',')
			if popps_str.strip() == "": continue
			popps = popps_str.split("-")
			popps = list(set([tuple(popp.split("|")) for popp in popps]))
			for popp in popps:
				if self.limit_pops:
					if popp[0] not in self.include_pops:
						continue
				try:
					self.popp_to_clients[popp].append(client)
				except KeyError:
					self.popp_to_clients[popp] = [client]

		for popp in self.popp_to_clients:
			unreach_fn = os.path.join(CACHE_DIR, 'unreachable_dsts', "{}-{}.csv".format(
				popp[0],popp[1]))
			unreachable_clients = []
			if os.path.exists(unreach_fn):
				for row in open(unreach_fn, 'r'):
					unreachable_clients.append(row.strip())
			try:
				self.popp_to_clients[popp] = get_difference(self.popp_to_clients[popp], 
					unreachable_clients)
			except KeyError:
				pass
		every_address = list(set([client for popp in self.popp_to_clients for client in self.popp_to_clients[popp]]))
		self.utils.lookup_asns_if_needed(list(set([ip32_to_24(addr) for addr in every_address])))
		print("Every address includes {} addresses".format(len(every_address)))
		##### CAIDA AS data is way too noisy to motivate probing
		##### one option in the future is to consider this an overestimate of CCs and the 
		#### routes an underestimate. Then I can probe addresses based on the under,
		#### and limit advertisements based on the over

		### Also look in the customer cone of all the peers, since we don't get all routes
		### Then we can be confident CCs don't overlap to the extent that our CC data is complete
		asn_to_ip = {}
		# self.utils.lookup_asns_if_needed(list(set([ip32_to_24(ip) for ip in every_address])))
		print("Forming ASN to IP mapping")
		for addr in every_address:
			asn = self.utils.parse_asn(addr)
			if asn is None:continue
			try:
				asn_to_ip[asn].append(addr)
			except KeyError:
				asn_to_ip[asn] = [addr]
		print("Tabulating {} popps to client mapping from CCs".format(len(self.popps)))
		for popp in get_difference(self.popps, self.provider_popps):
			if self.limit_pops:
				if popp[0] not in self.include_pops:
					continue
			# peer = self.utils.parse_asn(popp[1])
			# try:
			# 	self.popp_to_clients[popp]
			# except KeyError:
			# 	self.popp_to_clients[popp] = []
			# for asn in self.utils.get_cc(peer):
			# 	self.popp_to_clients[popp] += asn_to_ip.get(asn,[])
			pop,peer = popp
			try:
				self.popp_to_clients[popp]
			except KeyError:
				self.popp_to_clients[popp] = []
			for asn in cust_cone.get(peer,[]):
				self.popp_to_clients[popp] += asn_to_ip.get(asn,[])
		del asn_to_ip
		to_del = [popp for popp in self.popp_to_clients if len(self.popp_to_clients[popp]) == 0]
		for popp in to_del:
			del self.popp_to_clients[popp]
		## Form inverse mapping
		print("Tabulating client to popp mapping from BGP data")
		self.client_to_popps = {client: [] for client in every_address}
		for popp in get_difference(self.popp_to_clients,self.provider_popps):
			if self.limit_pops:
				if popp[0] not in self.include_pops:
					continue
			for client in self.popp_to_clients[popp]:
				self.client_to_popps[client].append(popp)
		for client in every_address:
			self.client_to_popps[client] += self.provider_popps
			self.client_to_popps[client] = list(set(self.client_to_popps[client]))

		### TODO -- maybe reconsider something here, since this is a lot of extra memory
		# for trivial information
		for popp in self.provider_popps:
			if self.limit_pops:
				if popp[0] not in self.include_pops:
					continue
			self.popp_to_clients[popp] = every_address

		#### Create popp to asn mapping to make conflict finding easier
		print("Tabulating popp to clientasn mapping")
		self.popp_to_clientasn = {}
		for popp,clients in self.popp_to_clients.items():
			if popp in self.provider_popps: continue
			## Exclude clients which we know don't have a path
			pop,peer = popp
			unreachable_fn = os.path.join(CACHE_DIR, 'unreachable_dsts', 
				'{}-{}.csv'.format(pop,peer))
			if os.path.exists(unreachable_fn):
				bad_clients = []
				for row in open(unreachable_fn, 'r'):
					bad_clients.append(row.strip())
				self.popp_to_clients[popp] = get_difference(self.popp_to_clients[popp], bad_clients)

			self.popp_to_clientasn[popp] = {}
			for client in clients:
				casn = self.utils.parse_asn(client)
				if casn is None:
					casn = ip32_to_24(client)
				self.popp_to_clientasn[popp][casn] = None
			self.popp_to_clientasn[popp] = list(self.popp_to_clientasn[popp])


		# tmpip = '104.18.202.233'
		# advs = [('miami', '32787'), ('miami', '32771'), ('miami', '19531'), ('miami', '8038')]
		# print(self.client_to_popps[tmpip])
		# print(get_intersection(self.client_to_popps[tmpip], advs))
		# for popp,clients in self.popp_to_clients.items():
		# 	if tmpip in clients:
		# 		if popp in self.client_to_popps[tmpip]:
		# 			print("{} correctly has {}".format(popp,tmpip))
		# 		else:
		# 			print("{} incorrectly has {}".format(popp,tmpip))
		# for popp in advs:
		# 	clients = self.popp_to_clients[popp]
		# 	if tmpip in clients:
		# 		print("Adv {} has {}".format(popp,tmpip))
		# 	else:
		# 		print("Adv {} doesn't have {}".format(popp,tmpip))
		# exit(0)

		print("{} popps with clients".format(len(self.popp_to_clients)))

		n_clients_per_popp = list([len(self.popp_to_clients[popp]) for popp in self.popp_to_clients])
		x,cdf_x = get_cdf_xy(n_clients_per_popp,logx=True)
		import matplotlib.pyplot as plt
		plt.semilogx(x,cdf_x)
		plt.xlabel("Number of Clients")
		plt.ylabel("CDF of PoPPs")
		plt.grid(True)
		plt.savefig("figures/n_clients_per_popp.pdf")
		plt.clf(); plt.close()

	def load_anycast_latencies(self):
		anycast_latencies = {}
		self.anycast_latency_fn = os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(self.system))
		c_to_pop = {}
		if os.path.exists(self.anycast_latency_fn):
			for row in open(self.anycast_latency_fn, 'r'):
				_, client_network, latency,pop = row.strip().split(',')
				anycast_latencies[client_network] = float(latency)
				c_to_pop[client_network] = pop

		return anycast_latencies

	def measure_vpn_lats(self):
		## Remeasure VPN lats
		vpn_lat_results = measure_latency_ips([self.vpn_ips[pop] for pop in self.pops_list])
		for pop,dst in zip(self.pops_list, vpn_lat_results):
			self.pop_vpn_lats[pop] = np.min(vpn_lat_results[dst])
		print("New VPN lats: {}".format(self.pop_vpn_lats))
		pickle.dump(self.pop_vpn_lats, open(self.pop_vpn_lats_fn,'wb'))

	def load_per_popp_meas(self, fn):
		from analyze_measurements import Measurement_Analyzer
		ma = Measurement_Analyzer()
		res = parse_ingress_latencies_mp(fn)
		return res['meas_by_popp'], res['meas_by_ip']

	


	def convert_org_to_peer(self, org, pop):
		if type(org) == str:
			org = org.strip()
		peers = self.org_to_peer[org]
		if len(peers) == 1:
			return peers[0]
		at_pop = {p:p in self.peers[pop] for p in peers}
		## (ITS A PROBLEM)
		# print("Problem, {} maps to {}, at pop {} : {}".format(org,peers,pop,at_pop))
		return [p for p,v in at_pop.items() if v][0]


	

	

	def find_needed_pingable_targets(self):
		import pytricia

		print("reading unreachable dsts")
		unreachable_clients_by_peer = {}
		for fn in glob.glob(os.path.join(CACHE_DIR, 'unreachable_dsts', '*')):
			pop,peer = fn.split("/")[-1].split("-")
			peer = peer[0:-4]
			try:
				unreachable_clients_by_peer[peer]
			except KeyError:
				unreachable_clients_by_peer[peer] = []
			for row in open(fn,'r'):
				unreachable_clients_by_peer[peer].append(row.strip())

		self.check_load_addresses_that_respond_to_ping()

		all_targs_to_probe = {}
		providers = list(set(peer for pop,peer in self.provider_popps))
		peeri=0
		for row in open(os.path.join(CACHE_DIR, 'possible_pref_targets_by_peer.csv'),'r'):
			peeri+=1
			peer, prefs = row.strip().split(',')
			if peer in providers: continue
			this_peer_tri = pytricia.PyTricia()
			prefs = [pref.strip() for pref in prefs.split('-') if pref.strip() != ""]
			prefs = [pref for pref in prefs if int(pref.split('/')[1]) < 30]
			for pref in prefs:
				this_peer_tri[pref] = None
			print("{} : Peer {} has {} prefs".format(peeri,peer,len(prefs)))
			if len(prefs) == 0 : continue
			## for each routable prefix, if I've probed something in it that responds to ping
			## but not route to peer, exclude the prefix from consideration
			exclude_prefs = []

			# clients_tested = unreachable_clients_by_peer.get(peer,[]) + \
			# 	self.addresses_that_respond_to_ping[0]
			clients_tested = unreachable_clients_by_peer.get(peer,[])
			self.utils.check_load_ip_to_asn()
			for client in clients_tested:
				client += "/32"
				pref_adv = this_peer_tri.get_key(client)
				done = False
				while not done and pref_adv != "0.0.0.0/0" and pref_adv is not None:
					exclude_prefs.append(pref_adv)
					# look at covering prefixes, if they are also announced
					# if there are none announced, this function call returns 0.0.0.0/0
					parent = this_peer_tri.parent(pref_adv)
					if parent is None or parent == "0.0.0.0/0":
						done = True
					else:
						pref_adv = parent
			print("Already probed {} of these".format(len(set(exclude_prefs))))
			prefs = get_difference(prefs, exclude_prefs)
			all_targs_to_probe[peer] = []
			if len(prefs) >= 5e3:
				# likely incorrect customer cone
				# only include prefs in this peer's AS
				prefs = self.utils.routeviews_asn_to_pref.get(peer)
				if prefs is None:
					continue
				prefs = [pref for pref in prefs if int(pref.split('/')[1]) < 30]
				print("Incorrect, working through {} prefs instead".format(len(prefs)))
			else:
				print("Working through remaining {} prefs.".format(len(prefs)))
			if len(prefs) == 0:
				with open(os.path.join(CACHE_DIR, 'no_pingable_targets_by_peer.csv'),'w') as f:
					f.write("{}\n".format(peer))
				continue
			base_arr = np.arange(2**24)
			for pref in prefs:
				if pref.strip() == "": continue
				pref,preflength = pref.split("/")
				if pref == "0.0.0.0": continue
				parts = pref.split(".")
				ipnum = (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
					(int(parts[2]) << 8) + int(parts[3])
				n_addr = 2**(32 - int(preflength)) - 2
				if n_addr <= 0:
					continue
				rand_hosts = ipnum+1+np.random.choice(base_arr[0:n_addr], size=np.minimum(n_addr,10),replace=False)

				all_targs_to_probe[peer] += list(rand_hosts)
			all_targs_to_probe[peer] = ['.'.join([str(ipint >> (i << 3) & 0xFF)
				for i in range(4)[::-1]]) for ipint in all_targs_to_probe[peer]] 
		all_peers_targs_to_probe = list(set([el for hosts in all_targs_to_probe.values() for el in hosts]))
		self.all_client_addresses_from_msft = self.all_client_addresses_from_msft + all_peers_targs_to_probe
		self.all_client_addresses_from_msft = list(set(self.all_client_addresses_from_msft))
		responsive_targets = self.get_reachable_clients()

	def limit_to_interesting_clients(self, clients):
		### limit to clients who have a path to >= N peers
		
		###### TODO -- remake this function to be looking into measurement results to see which clients actually
		### have paths to interesting peers, then returning those clients and/or limiting
		### all relevant objects to only use these interesting clients

		print("Limiting to clients of interest...")
		popp_lat_fn = os.path.join(CACHE_DIR, "{}_ingress_latencies_by_dst.csv".format(self.system))
		meas_by_popp, meas_by_ip = self.load_per_popp_meas(popp_lat_fn)

		interesting_clients = []
		for popp in self.popps:
			if popp in self.provider_popps: continue
			these_clients = meas_by_popp.get(popp, [])
			interesting_clients = interesting_clients + these_clients

		interesting_clients = list(set(interesting_clients))
		print("Limited {} clients to {} clients.".format(len(clients), len(interesting_clients)))

		return interesting_clients

	def check_load_addresses_that_respond_to_ping(self):
		try:
			self.addresses_that_respond_to_ping
			return
		except AttributeError:
			pass
		print("Loading addresses that respond to ping.")
		### addresses for which we have and have not previously gotten a response to a ping
		self.addresses_that_respond_to_ping = {0:[],1:[]}
		if os.path.exists(self.addresses_that_respond_to_ping_fn):
			with open(self.addresses_that_respond_to_ping_fn, 'r') as f:
				for row in f:
					# 1 -- responds, 0 -- doesn't
					ipadr,reach = row.strip().split(',')
					self.addresses_that_respond_to_ping[int(reach)].append(ipadr)
		# If an address has failed to respond once in history, exclude it
		self.addresses_that_respond_to_ping[1] = get_difference(self.addresses_that_respond_to_ping[1], 
				self.addresses_that_respond_to_ping[0])
		self.all_client_addresses_from_msft = list(set(self.all_client_addresses_from_msft + self.addresses_that_respond_to_ping[1]))

	def get_reachable_clients(self, limit=False):
		self.check_load_addresses_that_respond_to_ping()
		dsts = list(self.all_client_addresses_from_msft)

		## see which ones respond to ping
		# check if we've already measured responsiveness
		already_know_responsiveness = self.addresses_that_respond_to_ping[0] + self.addresses_that_respond_to_ping[1]
		dont_know_responsiveness = get_difference(dsts, already_know_responsiveness)
		if len(dont_know_responsiveness) > 0:
			responsive_dsts = check_ping_responsive(dont_know_responsiveness) + self.addresses_that_respond_to_ping[1]
		else:
			responsive_dsts = self.addresses_that_respond_to_ping[1]
		responsive_dsts = list(set(responsive_dsts))
		self.addresses_that_respond_to_ping[1] += get_intersection(dsts, responsive_dsts)
		self.addresses_that_respond_to_ping[0] += get_difference(dont_know_responsiveness, responsive_dsts)
		if len(dont_know_responsiveness) > 0:
			# Changed information
			print("WARNING -- writing to file, don't exit")
			time.sleep(5) # give time to register that we shouldn't exit
			with open(self.addresses_that_respond_to_ping_fn, 'w') as f:
				for i in [0,1]:
					for dst in self.addresses_that_respond_to_ping[i]:
						f.write("{},{}\n".format(dst,i))
			print("Done writing to file")
		print("Found {} responsive destinations in file.".format(len(responsive_dsts)))
		if limit:
			responsive_dsts = self.limit_to_interesting_clients(responsive_dsts)
			print("Limited to {} interesting responsive destinations.".format(len(responsive_dsts)))

		for pop in self.pop_to_clients:
			self.pop_to_clients[pop] = get_intersection(self.pop_to_clients[pop], responsive_dsts)

		return responsive_dsts