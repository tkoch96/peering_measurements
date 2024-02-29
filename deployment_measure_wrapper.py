import os, re, socket, csv, time, json, numpy as np, pickle, geopy.distance, copy, glob, tqdm
from config import *
from helpers import *
from subprocess import call, check_output
import subprocess
from generic_measurement_utils import AS_Utils_Wrapper
from pinger_wrapper import Pinger_Wrapper

asn_to_clients_fn = os.path.join(CACHE_DIR, 'asn_to_probable_clients.csv')

class Deployment_Measure_Wrapper():
	def __init__(self, **kwargs):
		#### Speeds up loading everything considerably if you set this to false
		self.recalc_probable_clients = False

		self.system = 'vultr'
		self.max_n_communities = 0

		self.addresses_that_respond_to_ping_fn = os.path.join(DATA_DIR, "addresses_that_respond_to_ping.csv")
		all_muxes_str = check_output("sudo client/peering openvpn status", shell=True).decode().split('\n')
		if self.system == "peering":
			self.pops = ['amsterdam01']
		elif self.system == "vultr":
			self.pops = []
			for row in all_muxes_str:
				if row.strip() == '': continue
				if 'vtr' not in row: continue
				self.pops.append(row.split(' ')[0][3:])
		self.pops = sorted(self.pops)
		self.pop_to_intf = {}
		for pop in self.pops:
			this_mux_str = [mux_str for mux_str in all_muxes_str if "vtr"+pop in mux_str]
			assert len(this_mux_str) == 1
			this_mux_str = this_mux_str[0]
			self.pop_to_intf[pop] = "tap" + re.search("tap(\d+)", this_mux_str).group(1)

		### prefixes we can conduct experiments with
		self.available_prefixes = ["184.164.238.0/24","184.164.239.0/24","184.164.240.0/24", 
			"184.164.241.0/24","184.164.242.0/24","184.164.243.0/24"]
		# self.available_prefixes = ["184.164.238.0/24","184.164.239.0/24","184.164.240.0/24", 
		# 	"184.164.242.0/24","184.164.243.0/24"]

		# self.available_prefixes = ['138.185.230.0/24']
		self.rfd_track_fn = os.path.join(CACHE_DIR, 'rfd_tracking.csv')
		self.active_experiments = {}
		self.limit_to_yunfan = kwargs.get('limit_to_yunfan', False)

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
		self.popps = {}
		self.pop_to_ixps = {}
		self.popps_to_ixps = {}
		provider_popp_fn = os.path.join(CACHE_DIR, '{}_provider_popps.csv'.format(self.system))
		popps = csv.DictReader(open(os.path.join(DATA_DIR, "{}_peers_inferred.csv".format(self.system)), 'r'))
		for row in popps:
			if self.system == "peering":
				pop, peer, session_id = row["BGP Mux"], row["Peer ASN"], row["Session ID"]
				if ":" in row["Peer IP address"]: continue
				try:
					self.peers[pop].append(peer)
				except KeyError:
					self.peers[pop] = [peer]
				try:
					self.peer_to_id[pop,peer]
					# print("Warning -- multiple session ID's for {} {}".format(pop,peer))
				except KeyError:
					self.peer_to_id[pop,peer] = session_id
				self.popps[pop,peer] = None
			elif self.system == "vultr":
				pop, peer, nh, tp, ixp = row["pop"], row["peer"], row["next_hop"], row["type"], row["ixp"]
				try:
					self.peers[pop].append(peer)
				except KeyError:
					self.peers[pop] = [peer]
				try:
					self.popps[pop,peer].append(tp)
				except KeyError:
					self.popps[pop,peer] = [tp]
				if ixp != '' and ixp != 'None':
					try:
						self.popps_to_ixps[pop,peer][ixp] = None
					except KeyError:
						self.popps_to_ixps[pop,peer] = {ixp: None}
					try:
						self.pop_to_ixps[pop][ixp] = None
					except KeyError:
						self.pop_to_ixps[pop] = {ixp:None}
		# print(self.pop_to_ixps)
		# print(list(set([ixp for pop in self.pop_to_ixps for ixp in self.pop_to_ixps[pop]])))

		if self.system == 'vultr':
			# for popp,tps in self.popps.items():
			# 	tps = list(sorted(set(tps)))
			# 	if len(tps) > 1: 
			# 		if tps == ['privatepeer', 'provider']:
			# 			self.popps[popp] = 'provider'
			# 		elif tps == ['provider', 'routeserver']:
			# 			self.popps[popp] = 'provider'
			# 		elif tps == ['privatepeer','provider','routeserver']:
			# 			self.popps[popp] = 'provider'
			# 		elif tps == ['privatepeer', 'routeserver']:
			# 			self.popps[popp] = 'private_peer'check_construct_client_to_peer_mapping
			# 		elif tps == ['ixp_direct', 'routeserver']:
			# 			self.popps[popp] = 'ixp_direct'
			# 	else:
			# 		self.popps[popp] = tps[0]
			with open(provider_popp_fn, 'w') as f:
				for popp,rels in self.popps.items():
					if 'provider' in rels:
						f.write("{},{}\n".format(popp[0],popp[1]))
	
			## 0:IXP, IXP:Peer			
			self.community_ixps_type_1 = ['6777','63221','24115','62499','6695','43252','47228',
				'56890','8714','19996','63221','13538','63034','51706','33108','19996',
				'24115','55518','48793','6895','7606','45686',
				'11670','52005','24224','7606','59200','61522']
			## 65000:0, 64960:Peer
			self.community_ixps_type_2 = ['58941','58942','58943']
			## 65001:Peer
			self.community_ixps_type_3 = ['26162']
			## IXP:0, IXP:Peer
			self.community_ixps_type_4 = ['49378']
			community_ixps = self.community_ixps_type_1 + self.community_ixps_type_2 + \
				self.community_ixps_type_3 + self.community_ixps_type_4
			self.no_community_ixps = list(set([ixp for pop in self.pop_to_ixps for ixp in self.pop_to_ixps[pop]]))
			self.no_community_ixps = get_difference(self.no_community_ixps, community_ixps)

		self.peer_to_pops = {}
		for pop in self.peers:
			self.peers[pop] = list(set(self.peers[pop])) # remove dups
			for peer in self.peers[pop]:
				try:
					self.peer_to_pops[peer].append(pop)
				except KeyError:
					self.peer_to_pops[peer] = [pop]

		### Providers, learned from BGP routes
		self.provider_popps, self.provider_peers = [], []
		for row in open(provider_popp_fn,'r'):
			self.provider_popps.append(tuple(row.strip().split(',')))
			self.provider_peers.append(self.provider_popps[-1][1])
		self.provider_peers = list(set(self.provider_peers))

		self.pop_to_loc = POP_TO_LOC[self.system]
		self.pop_vpn_lats = {}
		self.pop_to_clients_fn = os.path.join(DATA_DIR, 'client_to_pop.csv')
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
			'osaka': '64.176.44.110',
			'santiago': '64.176.14.19',
			'manchester': '64.176.187.76',
			'telaviv': '64.176.173.34',
			'honolulu': '208.83.237.194',

		}}[self.system]
		self.pops_list = list(self.vpn_ips)

		self.include_pops = kwargs.get('include_pops', self.pops)
		if len(self.include_pops) < len(self.pops):
			self.pops_list = get_intersection(self.pops_list, self.include_pops)
			self.peers = {pop:self.peers[pop] for pop in self.include_pops}
			self.popps = [(pop,peer) for pop,peer in self.popps if pop in self.include_pops]
			self.pop_to_loc = {pop:self.pop_to_loc[pop] for pop in self.include_pops}
			self.pops = self.include_pops
		print("Working with PoPs : {}".format(self.pops))
		if kwargs.get('quickinit', False): return

		## Load utils like AS to org mapping		
		self.utils = AS_Utils_Wrapper()
		self.utils.check_load_siblings()
		self.utils.check_load_as_rel()
		self.utils.update_cc_cache()


		
		## clients that switch between pops, annoying
		self.multipop_clients_fn = os.path.join(DATA_DIR, 'multipop_clients.csv')
		self.multipop_clients = []
		if os.path.exists(self.multipop_clients_fn):
			for row in open(self.multipop_clients_fn, 'r'):
				client = row.strip()
				self.multipop_clients.append(client)

		## we've measured from all destinations to these popps
		self.already_completed_popps_fn = os.path.join(CACHE_DIR, 'already_completed_popps.csv')

		### Get AS to org mapping and vice versa for compatibility with PAINTER pipeline
		peer_to_org_fn = os.path.join(CACHE_DIR, 'vultr_peer_to_org.csv')
		peer_orgs = list(set(self.utils.parse_asn(peer) for pop,peer in self.popps))
		with open(peer_to_org_fn, 'w') as f:
			for peer_org in peer_orgs:
				asns = self.utils.org_to_as.get(peer_org, [])
				if len(asns) > 0:
					asns_str = "-".join(sorted(asns))
					f.write("{},{}\n".format(peer_org,asns_str))

		self.maximum_inflation = kwargs.get('maximum_inflation', 1)

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

	def get_communtiy_str_vultr_adv_to_peers_old(self, pop, peers):
		#### Basic version, ignore all routeservers
		### (so many popps with only routeserver connections will be unreachable)

		community_str = ""
		peers_filtered = list([peer for peer in peers if int(peer) <= 65000])
		if len(peers_filtered) != len(peers):
			print("Note -- ignoring some peers as larger ASNs not yet supported.")
		if len(peers_filtered) == 0:
			raise ValueError("must be nonzero number of peers")

		community_str += "-c 20473,6000 "

		for peer in peers:
			community_str += "-c 64699,{} ".format(peer)

		### Be superstitious and actively exclude providers
		providers_this_pop = [popp[1] for popp in self.provider_popps if popp[0] == pop]
		for exclude_provider in get_difference(providers_this_pop, peers):
			community_str += "-c 64600,{} ".format(exclude_provider)

		return community_str

	def get_communtiy_str_vultr_adv_to_peers(self, pop, peers):
		### Want to advertise to these peers and only these peers at this pop
		### ideally would include all ixp-specific rules
		community_str = ""
		peers_filtered = list([peer for peer in peers if int(peer) <= 65000])
		if len(peers_filtered) != len(peers):
			print("Note -- ignoring some peers as larger ASNs not yet supported.")
		if len(peers_filtered) == 0:
			raise ValueError("must be nonzero number of peers")

		this_pop_providers = list([peer for _pop,peer in self.popps if 'provider' in self.popps[_pop,peer] and pop == _pop])

		## for peers with more than one connection type at a PoP (which is many of them)
		## we prefer provider connections first, then public peer, then private peer, then routeserver
		exclude_onward = set() 

		providers_include = get_intersection(peers_filtered, this_pop_providers)
		providers_exclude = get_difference(this_pop_providers, peers_filtered)
		for provider_exclude in providers_exclude:
			community_str += "-c 64600,{} ".format(provider_exclude)

		exclude_onward = exclude_onward.union(set(providers_include))

		community_str += "-c 20473,6601 " # no export to direct peers via the IXP
		direct_ixp_peers_include = [peer for peer in peers_filtered if 'ixp_direct' in self.popps[pop,peer]]
		direct_ixp_peers_inlcude = get_difference(direct_ixp_peers_include, exclude_onward)
		for direct_ixp_peer in direct_ixp_peers_include:
			community_str += "-c 64699,{} ".format(direct_ixp_peer)

		exclude_onward = exclude_onward.union(set(direct_ixp_peers_include))

		this_pop_private_peers = list([peer for _pop,peer in self.popps if 'privatepeer' in self.popps[_pop,peer] and pop == _pop])
		this_pop_private_peers = get_difference(this_pop_private_peers, exclude_onward)
		private_peers_include = get_intersection(peers_filtered, this_pop_private_peers)
		private_peers_exclude = get_difference(this_pop_private_peers, peers_filtered)

		for private_peer_exclude in private_peers_exclude:
			community_str += "-c 64600,{} ".format(private_peer_exclude)

		exclude_onward = exclude_onward.union(set(private_peers_include))

		this_pop_routeserver_peers = list([peer for _pop,peer in self.popps if 'routeserver' in self.popps[_pop,peer] and pop == _pop])
		routeserver_peers = get_intersection(peers_filtered, this_pop_routeserver_peers)
		routeserver_peers = get_difference(routeserver_peers, exclude_onward)

		if len(routeserver_peers) > 0:
			community_str += "-c 20473,6602 " ## Export to route servers
			## all IXPs at the PoP
			all_ixps = self.pop_to_ixps[pop]
			## all IXPs with peers we want to advertise to
			ixps_used = list(set(ixp for peer in routeserver_peers for ixp in self.popps_to_ixps[pop,peer]))
			## IXPs at the PoP that we definitely don't want to advertise to
			unused_ixps = get_difference(all_ixps, ixps_used)
			for ixp in unused_ixps:
				## dont advertise to these IXPs
				community_str += '-c 64600,{} '.format(ixp)
			for ixp in ixps_used:
				if ixp in self.no_community_ixps:
					## We can't limit advertisements because we don't know how to, so block all advertisements to this IXP
					community_str += '-c 64600,{} '.format(ixp)
					continue
			peers_by_ixp = {}
			for peer in routeserver_peers:
				for ixp in sorted(self.popps_to_ixps[pop,peer]): # since we sort, we'll use the same IXP for all of them if we can
					if ixp in unused_ixps or ixp in self.no_community_ixps: continue
					try:
						peers_by_ixp[ixp].append(peer)
					except KeyError:
						peers_by_ixp[ixp] = [peer]
					#### Just use one IXP, since routing is very likely the same to all
					break
			for ixp,this_ixp_peers in peers_by_ixp.items():
				## Advertise to this 
				community_str += self.ixp_to_limit_str(ixp, this_ixp_peers)
		return community_str	

	def ixp_to_limit_str_old(self, ixp, peers):
		community_str = ""
		if ixp == '45686':
			## JPNAP, Euro-IX standard https://www.euro-ix.net/en/forixps/large-bgp-communities/
			community_str = "-c 45686,0,0 " # no export
			for peer in peers:
				community_str += "-l 45686,1,{} ".format(peer) # override no export to this peer
		elif ixp == '26162':
			## NIC.BR http://docs.ix.br/doc/politica-de-tratamento-de-communities-no-ix-br-v4_3.pdf
			for peer in peers:
				community_str += "-l 65001,0,{} ".format(peer)
		elif ixp == '24115':
			## Equinix, docs here https://deploy.equinix.com/developers/docs/metal/bgp/global-communities/
			## but I got this community from VULTR support
			for peer in peers:
				community_str += "-l 24115,1,{} ".format(peer)
		else:
			print("WARNING : IXP {} not implemented in ixp_to_limit_str".format(ixp))
		return community_str

	def ixp_to_limit_str(self, ixp, peers):
		### All of these are from jiangchen
		## https://docs.google.com/document/d/1MWMaPGJV9GalA_AeFMjupsgwhGklAyzZLThtGJRPCt4/edit

		## 0:IXP, IXP:Peer			
		# self.community_ixps_type_1 = ['6777','63221','24115','62499','6695','43252','47228',
		# 	'56890','8714','19996','63221','13538','63034','51706','33108','19996',
		# 	'24115','55518','48793','6895','24224','7606','45686',
		# 	'11670','52005','24224','7606','59200','61522']
		# ## 65000:0, 64960:Peer
		# self.community_ixps_type_2 = ['58941','58942','58943']
		# ## 65001:Peer
		# self.community_ixps_type_3 = ['26162']
		# ## IXP:0, IXP:Peer
		# self.community_ixps_type_4 = ['49378']

		community_str = ""
		if ixp in self.community_ixps_type_1:
			## 0:IXP, IXP:Peer			
			community_str += "-c 0,{} ".format(ixp) # no export to any peer
			for peer in peers:
				community_str += '-c {},{} '.format(ixp,peer) # override no export to a specific peer
		elif ixp in self.community_ixps_type_2:
			## 65000:0, 64960:Peer
			community_str += "-c 65600,0 "
			for peer in peers:
				community_str += "-c 64960,{} ".format(peer)
		elif ixp in self.community_ixps_type_3:
			## 65001:Peer
			for peer in peers:
				community_str += "-c 65001,{} ".format(peer)
		elif ixp in self.community_ixps_type_4:
			community_str += "-c {},0 ".format(ixp)
			for peer in peers:
				community_str += "-c {},{} ".format(ixp,peer)
		else:
			print("WARNING : IXP {} not implemented in ixp_to_limit_str".format(ixp))
		return community_str

	def advertise_to_peers(self, pop, peers, pref, **kwargs):
		## check to make sure we're not advertising already
		cmd_out = self._check_output("sudo client/peering bgp adv vtr{}".format(pop),careful=CAREFUL)
		if cmd_out is not None:
			if pref in cmd_out.decode():
				print("WARNING ---- ALREADY ADVERTISING {} TO {}".format(pref, pop))
				time.sleep(5)
				self.withdraw_from_pop(pop, pref)

		if self.system == 'peering':
			# advertise only to this peer
			self._call("sudo client/peering prefix announce -m {} -c 47065,{} {}".format( ### NEED TO UPDATE
				pop, self.peer_to_id[pop, peer], pref),careful=CAREFUL, **kwargs)
		elif self.system == 'vultr':
			#https://github.com/vultr/vultr-docs/tree/main/faq/as20473-bgp-customer-guide#readme
			community_str = self.get_communtiy_str_vultr_adv_to_peers(pop, peers)
			n_communities = community_str.count('-c')
			self.max_n_communities = np.maximum(n_communities, self.max_n_communities)
			self._call("sudo client/peering prefix announce -m vtr{} {} {}".format(
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
		if not CAREFUL:
			time.sleep(30) # route convergence

	def advertise_to_pop(self, pop, pref, **kwargs):
		## Advertises to all peers at a single pop
		self._call("sudo client/peering prefix announce -m vtr{} {}".format(
				pop, pref),careful=CAREFUL,**kwargs)

	def advertise_to_pops(self, pops, pref, **kwargs):
		for i,pop in enumerate(pops):
			## Advertises to all peers at a single pop
			self._call("sudo client/peering prefix announce -m vtr{} {}".format(
					pop, pref),careful=CAREFUL,**kwargs,override_rfd=(i>0)) 

	def withdraw_from_pop(self, pop, pref):
		if self.system == 'peering':
			# advertise only to this peer
			self._call("sudo client/peering prefix withdraw -m {} -c 47065,{} {}".format(
				pop, self.peer_to_id[pop, peer], pref),careful=CAREFUL)
		elif self.system == 'vultr':
			# just do the opposite of advertise
			self._call("sudo client/peering prefix withdraw -m vtr{} {} &".format(
				pop, pref),careful=CAREFUL)

	def announce_anycast(self, pref=None):
		if pref is None:
			pref = self.get_most_viable_prefix()
		for i, pop in enumerate(self.pops):
			self._call("sudo client/peering prefix announce -m vtr{} {}".format(
				pop, pref),careful=CAREFUL,override_rfd=(i>0)) 
		return pref
	def withdraw_anycast(self, pref):
		for pop in self.pops:
			self._call("sudo client/peering prefix withdraw -m vtr{} {}".format(
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
		np.random.seed(31415)
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

		all_popps = list(self.popps)
		popps_to_focus_on = get_difference(all_popps, already_completed_popps)
		print("{} popps total, {} left to measure to".format(len(all_popps), len(popps_to_focus_on)))

		if exclude_providers:
			popps_to_focus_on = get_difference(popps_to_focus_on, self.provider_popps)
			print("After removing providers, {} popps".format(len(popps_to_focus_on)))
			popps_to_focus_on = get_intersection(popps_to_focus_on, self.popp_to_clientasn) # focus on popps with clients
			print("After removing popps with no clients, {} popps".format(len(popps_to_focus_on)))

		# TODO -- implement larger ASNs
		popps_to_focus_on = [popp for popp in popps_to_focus_on if int(popp[1]) <= 65000]
		print("Limiting to {} popps with ASNs we've implemented".format(len(popps_to_focus_on)))
		popps_to_focus_on = [(pop,peer) for pop,peer in popps_to_focus_on if pop in self.pops]
		print("Limiting to {} popps with at PoPs we're focusing on".format(len(popps_to_focus_on)))

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
				for p in sorted(get_difference(ranked_popps,this_set)):
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
						# print("{} -- {}, intersection: {}".format(
						# 	p,entry,get_intersection(this_clients_set,compare_clients_set)[0:50]))
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

	def get_clients_by_popp(self, popps, verb=False):
		self.check_construct_client_to_peer_mapping()
		every_client_of_interest = self.get_reachable_clients()
		if popps == 'all':
			## return every client in a list
			return every_client_of_interest
		else:
			# return popp -> clients
			ret = {}
			for popp in popps:
				if popp in self.provider_popps: continue # trivial
				ret[popp] = []
				for asn in self.popp_to_clientasn.get(popp,[]):
					if verb:
						print("Adding clients from asn {} since its a child of the popp {}".format(asn,popp))
					ret[popp] += copy.deepcopy(self.asn_to_clients.get(asn,[]))
					if verb:
						if len(ret[popp]) == 0:
							print("{} has no clients in client asn {}".format(popp,asn))
			return ret
			
	def get_popps_by_client(self, client, verb=False):
		casn = self.utils.parse_asn(client)
		if casn is None:
			casn = ip32_to_24(client)

		if verb:
			print("Popps for {} ({}) in data structure are {}".format(casn, client, self.clientasn_to_popp.get(casn) ))

		return copy.deepcopy(self.clientasn_to_popp.get(casn, []))

	def check_construct_client_to_peer_mapping(self, forceparse=False):
		### Goal is to create client to popps mapping (so that we can create popp to clients mapping)
		try:
			self.popp_to_clientasn
			return
		except AttributeError:
			pass
		peer_to_clients_fn = os.path.join(CACHE_DIR, 'peer_to_clients.csv')
		if not os.path.exists(peer_to_clients_fn) or forceparse:

			vultr_peer_asns = list(set(self.utils.parse_asn(peer) for pop,peer in self.popps))
			# peer_ccs = {}
			# for peer in vultr_peer_asns:
			# 	for peer_child in self.utils.get_cc(peer):
			# 		try:
			# 			peer_ccs[peer_child].append(peer)
			# 		except KeyError:
			# 			peer_ccs[peer_child] = [peer]
			# peer_child_to_stats = {} # store CC size, number of peers its a child of
			# for peer_child,peers in sorted(peer_ccs.items(), key = lambda el : -1 * len(el[1])):
			# 	peer_child_to_stats[peer_child] = {'cc': len(self.utils.get_cc(peer_child)), 'n_peers': len(peers)}

			# x = list([el['cc'] for el in peer_child_to_stats.values()])
			# y = list([el['n_peers'] for el in peer_child_to_stats.values()])
			# import matplotlib.pyplot as plt
			# plt.scatter(x,y)
			# plt.xlabel("CC Size")
			# plt.ylabel("Number of Peers Child Of")
			# plt.savefig('figures/bad_cc_investigation.pdf')

			print("Creating popp to clients mapping")
			clients = self.get_reachable_clients(limit=False)
			# self.utils.lookup_asns_if_needed(list(set([ip32_to_24(addr) for addr in clients])))

			
			asn_to_parents = {}
			all_asns = set(list(self.utils.cc_cache)).union(set(self.utils.parse_asn(client) for client in clients))
			all_asns = all_asns.union(set(list(self.utils.as_to_customers)))
			for parent in tqdm.tqdm(all_asns, desc="Parsing CC data from CAIDA to tabulate parents and children..."):
				for child in self.utils.get_cc(parent):
					try:
						asn_to_parents[child][parent] = None
					except KeyError:
						asn_to_parents[child] = {parent: None}

			for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, '{}_customer_cone_from_routes.csv'.format(
				self.system)),'r'), desc="Reading customer cone data from Vultr routes..."):
				if row.startswith('parent'): continue
				parent,children = row.strip().split(',')
				parent = self.utils.parse_asn(parent)
				for child in children.split('-'):
					child = self.utils.parse_asn(child)
					try:
						asn_to_parents[child][parent] = None
					except KeyError:
						asn_to_parents[child] = {parent: None}
				for child in self.utils.get_cc(parent):
					try:
						asn_to_parents[child][parent] = None
					except KeyError:
						asn_to_parents[child] = {parent: None}
			pcone_of_interest = asn_to_parents.get(self.utils.parse_asn('62490'))

			default_current_cone = {self.utils.parse_asn(asn):None for asn in self.provider_peers}
			pcone_cache = {}
			def get_parent_cone(asn, current_cone={}, 
				recursion_allowed = 0, dbg = False):
				current_cone = copy.deepcopy(current_cone)
				current_cone[asn] = None
				try:
					if not dbg:
						return pcone_cache[asn]
					else:
						print("in get parent cone for asn {}".format(asn))
				except KeyError:
					pass
				last_length = -1
				this_length = len(current_cone)
				while last_length != this_length:
					if dbg: print("{} {}".format(last_length, this_length))
					for parent in asn_to_parents.get(asn, []):
						if dbg:
							print("raw parent: {}".format(parent))
						if parent == asn: continue
						try:
							default_current_cone[parent]
							if dbg:
								print("skipping parent {} because its in dcc".format(parent))
							continue # trivial
						except KeyError:
							pass
						if dbg:
							print("{} a parent of {}, adding".format(parent,asn))
						if recursion_allowed > 0:
							try:
								current_cone[parent]
							except KeyError:
								for gparent in get_parent_cone(parent, current_cone=copy.deepcopy(current_cone),
								 	recursion_allowed=recursion_allowed - 1, dbg=dbg):
									if dbg:
										print("{} a parent of {}, adding".format(gparent, parent))
									current_cone[gparent] = None
						else:
							current_cone[parent] = None

					recursion_allowed += 1
					last_length = this_length
					this_length = len(current_cone)

				pcone_cache[asn] = copy.deepcopy(current_cone)
				return current_cone


			peer_to_client_asns = {}
			asn_to_clients = {}
			client_asns_debug = []
			for client in tqdm.tqdm(clients, desc="Forming peer to client mapping..."):
				# ntwrk = network_to_peers.get_key(client)
				# if ntwrk is None:
				# 	continue
				this_client_asn = self.utils.parse_asn(client)
				# this_client_asn = self.utils.parse_asn('8048')
				if this_client_asn is None: continue
				try:
					asn_to_clients[this_client_asn].append(client)
					continue
				except KeyError:
					asn_to_clients[this_client_asn] = [client]
				
				this_client_peers = get_intersection(vultr_peer_asns, 
					get_parent_cone(this_client_asn, dbg=this_client_asn in client_asns_debug))
				
				this_client_peers = get_difference(this_client_peers, default_current_cone) # trivial
				if this_client_asn in client_asns_debug:
					print("{} -- {}".format(this_client_asn,this_client_peers))
				for peer in this_client_peers:
					try:
						peer_to_client_asns[peer][this_client_asn] = None
					except KeyError:
						peer_to_client_asns[peer] = {this_client_asn: None}

			with open(peer_to_clients_fn, 'w') as f:
				for peer,this_peer_clients in peer_to_client_asns.items():
					for client in this_peer_clients:
						f.write("{},{}\n".format(peer,client))

			with open(asn_to_clients_fn, 'w') as f:
				for asn,this_asn_clients in asn_to_clients.items():
					for client in this_asn_clients:
						f.write('{},{}\n'.format(asn,client))
				
			del clients
			del self.addresses_that_respond_to_ping

		self.popp_to_clientasn = {}
		for row in tqdm.tqdm(open(peer_to_clients_fn,'r'),
			desc="Loading peer to client mapping..."):
			peer,client = row.strip().split(',')
			peer = self.utils.parse_asn(peer)
			client = self.utils.parse_asn(client)
			if peer is None or client is None: continue
			for _peer in self.utils.org_to_as.get(peer, [peer]):
				for pop in self.peer_to_pops.get(_peer,[]):
					try:
						self.popp_to_clientasn[pop,_peer].append(client)
					except KeyError:
						self.popp_to_clientasn[pop,_peer] = [client]
		self.asn_to_clients = {}
		for row in tqdm.tqdm(open(asn_to_clients_fn,'r'),desc="Loading asn to clients fn..."):
			asn,client = row.strip().split(',')
			asn = self.utils.parse_asn(asn)
			if asn is None: continue
			try:
				self.asn_to_clients[asn].append(client)
			except KeyError:
				self.asn_to_clients[asn] = [client]
		### Limit probeable clients to a maximum number per client ASN
		limit_to_max = 1000
		np.random.seed(31415)
		for asn in sorted(self.asn_to_clients, key = lambda el : int(el)):
			this_asn_clients = self.asn_to_clients[asn]
			np.random.shuffle(this_asn_clients)
			self.asn_to_clients[asn] = this_asn_clients[0:limit_to_max]

		print("{} popps with clients".format(len(self.popp_to_clientasn)))
		#### Create popp to asn mapping to make conflict finding easier
		self.clientasn_to_popp = {}
		for popp,this_popp_clients in tqdm.tqdm(self.popp_to_clientasn.items(),
			desc="tabulating clientasn to popp mapping..."):
			if popp in self.provider_popps: continue
			# ## Exclude clients which we know don't have a path
			# pop,peer = popp
			# unreachable_fn = os.path.join(CACHE_DIR, 'unreachable_dsts', 
			# 	'{}-{}.csv'.format(pop,peer))
			# if os.path.exists(unreachable_fn):
			# 	bad_clients = []
			# 	for row in open(unreachable_fn, 'r'):
			# 		bad_clients.append(row.strip())
			# 	self.popp_to_clientasn[popp] = get_difference(self.popp_to_clientasn[popp], bad_clients)

			for client in this_popp_clients:
				try:
					self.clientasn_to_popp[client].append(popp)
				except KeyError:
					self.clientasn_to_popp[client] = [popp]
		for casn,popps in self.clientasn_to_popp.items():
			self.clientasn_to_popp[casn] = list(set(popps))
		# for popp,casns in self.popp_to_clientasn.items():
		# 	if len(casns) > 20000:
		# 		print("{} has {} client ASNs".format(popp,len(casns)))
		n_clients_per_popp = list([len(self.popp_to_clientasn[popp]) for popp in self.popp_to_clientasn])
		x,cdf_x = get_cdf_xy(n_clients_per_popp,logx=True)
		import matplotlib.pyplot as plt
		plt.semilogx(x,cdf_x)
		plt.xlabel("Number of Client ASNs")
		plt.ylabel("CDF of PoPPs")
		plt.grid(True)
		plt.savefig("figures/n_networks_per_popp.pdf")
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
		### addresses for which we have and have not previously gotten a response to a ping
		self.addresses_that_respond_to_ping = {0:[],1:[]}
		if os.path.exists(self.addresses_that_respond_to_ping_fn):
			with open(self.addresses_that_respond_to_ping_fn, 'r') as f:
				for row in tqdm.tqdm(f,desc="Loading addresses that respond to ping."):
					# 1 -- responds, 0 -- doesn't
					ipadr,reach = row.strip().split(',')
					self.addresses_that_respond_to_ping[int(reach)].append(ipadr)
		# If an address has failed to respond once in history, exclude it
		self.addresses_that_respond_to_ping[1] = get_difference(self.addresses_that_respond_to_ping[1], 
				self.addresses_that_respond_to_ping[0])

	def check_load_yunfan_prefixes(self):
		try:
			self.yunfan_user_pref_tri
		except AttributeError:
			import pytricia
			self.yunfan_user_pref_tri = pytricia.PyTricia()
			for row in open(os.path.join(CACHE_DIR, 'yunfan_prefixes_with_users.csv')):
				if row.strip() == 'prefix': continue
				self.yunfan_user_pref_tri[row.strip()] = None

	def get_reachable_clients(self, limit=False, force=False):
		try:
			if not force:
				return self.reachable_clients
		except AttributeError:
			pass
		
		### We've pinged addresses to see if they respond and cached the answers, load this data
		self.check_load_addresses_that_respond_to_ping()
		
		if not self.recalc_probable_clients:
			### We've probed everything and we know what's reachable and what isn't
			### Faster to just skip the rest of this function
			self.reachable_clients = self.addresses_that_respond_to_ping[1]
			return self.reachable_clients

		self.check_load_probable_address_source_data()
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

		## Limit to yunfan user prefixes
		if self.limit_to_yunfan:
			self.check_load_yunfan_prefixes()
			limited_responsive_dsts = {}
			n_choose_each_pref = 3
			for dst in responsive_dsts:
				pref = self.yunfan_user_pref_tri.get_key(dst + '/32')
				if pref is not None:
					try:
						limited_responsive_dsts[pref].append(dst)
					except KeyError:
						limited_responsive_dsts[pref] = [dst]
			for pref in list(limited_responsive_dsts):
				if len(limited_responsive_dsts[pref]) > n_choose_each_pref:
					dsts = np.random.choice(limited_responsive_dsts[pref], size=3)
				else:
					dsts = limited_responsive_dsts[pref]
				limited_responsive_dsts[pref] = dsts
			responsive_dsts = list(set([dst for pref,dsts in limited_responsive_dsts.items() for dst in dsts]))
			print("After limiting to Yunfan's prefixes, looking at {} responsive dsts".format(
				len(responsive_dsts)))
		self.check_load_pop_to_clients()
		for pop in self.pop_to_clients:
			self.pop_to_clients[pop] = get_intersection(self.pop_to_clients[pop], responsive_dsts)

		self.reachable_clients = responsive_dsts

		return responsive_dsts

	def conduct_measurements_to_prefix_popps(self, prefix_popps, every_client_of_interest, 
		popp_lat_fn, **kwargs):
		### Prefix_popps is a list of sets of popps, where each element in the list is the set of popps you'd like to advertise
		### the same prefix to
		### every_client_of_interest is either a single array of clients who you want to limit (every advertisement's)
		### focus to, or a list of clients correponding to those prefix_popps
		### popp_lat_fn is the output filename
		n_prefs = len(self.available_prefixes)
		n_adv_rounds = int(np.ceil(len(prefix_popps) / n_prefs))
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)

		using_manual_clients = kwargs.get('using_manual_clients', False)

		only_providers = kwargs.get('only_providers', False)
		np.random.seed(31415)

		propagate_time = kwargs.get('propagate_time', 10)

		for adv_round_i in range(n_adv_rounds):
			if not CAREFUL:
				self.measure_vpn_lats()
			adv_sets = prefix_popps[n_prefs * adv_round_i: n_prefs * (adv_round_i+1)]
			for asi,adv_set in enumerate(adv_sets):
				print("Adv set {}:\n {}".format(asi,adv_set))
			srcs = []
			pref_set = []
			popps_set = []
			pops_set = {}
			adv_set_to_taps = {}
			for i, popps in enumerate(adv_sets):
				if not CAREFUL:
					pref = self.get_most_viable_prefix()
				else:
					pref = self.available_prefixes[i]
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
				popps_set.append(popps) # each prefix will be used to measure these corresponding popps
				self.advertise_to_popps(popps, pref)

				pops = list(set([popp[0] for popp in popps])) # pops corresponding to this prefix
				pops_set[i] = pops
				adv_set_to_taps[i] = [self.pop_to_intf[pop] for pop in pops]
			print(self.max_n_communities)
			continue	

			if using_manual_clients:
				for pref,adv_set in zip(pref_set, adv_sets):
					print(adv_set)
					advsetstr = "--".join("-".join(list(el)) for el in adv_set)
					print(advsetstr)
					with open(popp_lat_fn, 'a') as f:
						f.write("{},{}\n".format(pref,advsetstr))

			if not CAREFUL:
				print("Waiting {}s for announcements to propagate...".format(propagate_time))
				time.sleep(propagate_time)
			max_n_pops = max(len(v) for v in adv_set_to_taps.values())	
			for pop_iter in range(max_n_pops):
				srcs_set = []
				taps_set = []
				this_pops_set = []
				clients_set = []
				adv_set_is = []
				for adv_set_i in range(len(adv_sets)):
					try:
						# different prefixes have different numbers of PoPs involved
						# measurements are done per PoP, so this is a bit awkward
						taps_set.append(adv_set_to_taps[adv_set_i][pop_iter])
						srcs_set.append(srcs[adv_set_i])
						this_pop = pops_set[adv_set_i][pop_iter] # one pop at a time
						this_pops_set.append(this_pop)
						adv_set_is.append(adv_set_i)
						if not using_manual_clients:
							if not only_providers:
								these_clients = set()
								for popp in popps_set[adv_set_i]:
									# print("Adding clients in adv set {}, popp {}".format(adv_set_i,popp))
									## Get any client who we know can route through these ingresses
									if popp[0] != this_pop: continue ## popps corresponding to focused pop
									this_popp_clients = list(self.get_clients_by_popp([popp]).values())[0]
									these_clients = these_clients.union(set(this_popp_clients))
									these_clients = set(get_intersection(these_clients, every_client_of_interest))
							else:
								these_clients = every_client_of_interest
							clients_set.append(sorted(list(these_clients)))
						else:
							### manually setting the clients to ping
							clients_set.append(every_client_of_interest[adv_round_i * 6 + adv_set_i])
					except IndexError:
						import traceback
						traceback.print_exc()
						pass
				print("PoP iter {}".format(pop_iter))
				print(srcs_set)
				print(taps_set)
				print(this_pops_set)
				print(popps_set)
				print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))

				if CAREFUL:
					t_meas = int(time.time())
					if not using_manual_clients:
						for src,pop,dsts,asi in zip(srcs_set, this_pops_set, clients_set, adv_set_is):
							print("{} {}".format(asi, popps_set[asi]))
							with open(popp_lat_fn,'a') as f:
								for client_dst in dsts:
									if len(popps_set[asi]) == 1:
										if popps_set[asi][0] in self.provider_popps:
											client_has_path = [popps_set[asi][0]]
										else:
											client_has_path = get_intersection(self.get_popps_by_client(client_dst), popps_set[asi])
									else:
										client_has_path = get_intersection(self.get_popps_by_client(client_dst), popps_set[asi])

									if len(client_has_path) != 1:
										print("WEIRDNESS -- client {} should have one path to \n{}\n but has many/none {}".format(
											client_dst,popps_set[asi],self.get_popps_by_client(client_dst,verb=True)))
										print("{} {} {}".format(src,pop,client_has_path))
										exit(0)
										continue
									clientpathpop, clientpathpeer = client_has_path[0]
									if clientpathpop != pop:
										print("Client path pop != pop for {}, {} vs {}".format(client_dst,clientpathpop,pop))
										exit(0)
										continue

				lats_results = pw.run(srcs_set, taps_set, clients_set,
					remove_bad_meas=False)
				pickle.dump([lats_results, srcs_set, this_pops_set,clients_set, adv_set_is], open('tmp/tmp.pkl','wb'))

				t_meas = int(time.time())
				for src,pop,dsts,asi in zip(srcs_set, this_pops_set, clients_set, adv_set_is):
					with open(popp_lat_fn,'a') as f:
						for client_dst in dsts:
							if not using_manual_clients:
								if len(popps_set[asi]) == 1:
									if popps_set[asi][0] in self.provider_popps:
										client_has_path = [popps_set[asi][0]]
									else:
										client_has_path = get_intersection(self.get_popps_by_client(client_dst), popps_set[asi])
								else:
									client_has_path = get_intersection(self.get_popps_by_client(client_dst), popps_set[asi])

								if len(client_has_path) != 1:
									print("WEIRDNESS -- client {} should have one path to \n{}\n but has many/none {}".format(
										client_dst,popps_set[asi],self.get_popps_by_client(client_dst,verb=True)))
									print("{} {} {}".format(src,pop,client_has_path))
									continue
								clientpathpop, clientpathpeer = client_has_path[0]
								if clientpathpop != pop:
									print("Client path pop != pop for {}, {} vs {}".format(client_dst,clientpathpop,pop))
									continue
								rtts = []
								for meas in lats_results[src].get(client_dst, []):
									### For this advertisement, there's only one ingress this client has a valid path to
									if meas.get('startpop','start') != meas.get('endpop','end'): continue
									if meas['pop'] != pop or meas['rtt'] is None:
										continue
									rtts.append(meas['rtt'])
								if len(rtts) > 0:
									rtt = np.min(rtts)
									f.write("{},{},{},{},{},{}\n".format(t_meas,client_dst,
										clientpathpop,clientpathpeer,round(rtt,4),round(rtt - self.pop_vpn_lats[pop],4)))
								else:
									### Need to save the information that this client could not reach its
									### intended destination, even though we thought it might have
									with open(os.path.join(CACHE_DIR, "unreachable_dsts", "{}-{}.csv".format(
										clientpathpop,clientpathpeer)),'a') as f2:
										f2.write("{}\n".format(client_dst))
							else:
								rtts = []
								for meas in lats_results[src].get(client_dst, []):
									### Make sure startpop is dstpop so we measure an RTT
									if meas.get('startpop','start') != meas.get('endpop','end'): continue
									if meas['pop'] != pop or meas['rtt'] is None:
										continue
									rtts.append(meas['rtt'])
								if len(rtts) > 0:
									rtt = np.min(rtts)
									f.write("{},{},{},{},{},{}\n".format(src,t_meas,client_dst,
										pop,round(rtt,4),round(rtt - self.pop_vpn_lats[pop],4)))


				del lats_results
			for pref, popps in zip(pref_set, popps_set):
				for pop in set([popp[0] for popp in popps]):
					self.withdraw_from_pop(pop, pref)
			if not CAREFUL:
				if kwargs.get('logcomplete',True):
					with open(self.already_completed_popps_fn, 'a') as f:
						for popps in popps_set:
							for pop,peer in popps:
								f.write("{},{}\n".format(pop,peer))

	def check_load_pop_to_clients(self):
		try:
			self.pop_to_clients
			return
		except AttributeError:
			pass
		## Natural client to PoP mapping for an anycast advertisement, learned from measurements
		self.pop_to_clients = {pop:[] for pop in self.pops}
		self.client_to_pop = {}
		print("Loading PoP to clients...")
		if os.path.exists(self.pop_to_clients_fn):
			for row in open(self.pop_to_clients_fn, 'r'):
				client,pop = row.strip().split(',')
				if pop not in self.pops:
					continue
				self.pop_to_clients[pop].append(client)
				self.client_to_pop[client] = pop

	def check_load_probable_address_source_data(self):
		"""Loads addresses from Microsoft and the ISI hitlist, to help find probable addresses in v4 space."""
		try:
			self.all_client_addresses_from_msft
			return
		except AttributeError:
			pass
		self.all_client_addresses_from_msft = set()
		## perhaps reachable addresses for clients
		self.dst_to_ug = {}
		self.ug_to_vol = {}
		for row in tqdm.tqdm(csv.DictReader(open(os.path.join(DATA_DIR,
			'client24_metro_vol_20220620.csv'),'r')), desc="Loading Microsoft user data"):
			break
			ug = (row['metro'], row['asn'])
			try:
				self.ug_to_vol[ug] += float(row['N'])
			except KeyError:
				self.ug_to_vol[ug] = float(row['N'])
			self.dst_to_ug[row['client_24']] = ug

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
		self.check_load_addresses_that_respond_to_ping()
		self.all_client_addresses_from_msft = list(set(list(self.all_client_addresses_from_msft) + list(self.addresses_that_respond_to_ping[1])))
	
		import bz2
		these_ips = []
		for row in tqdm.tqdm(bz2.open(os.path.join(DATA_DIR ,'internet_address_hitlist_it105w-20230926.fsdb.bz2'), 'r'),
			desc="Reading hitlist from Loqman..."):
			row = row.decode()
			if row.startswith('#'): continue
			row = row.strip().split('\t')
			these_ips.append(row[2])
		self.all_client_addresses_from_msft = set(self.all_client_addresses_from_msft).union(set(these_ips))
		### basically the ISI hitlist
		for fn in tqdm.tqdm(glob.glob(os.path.join(DATA_DIR, "hitlist_from_jiangchen", "*")),
			desc="Loading hitlists from JC"):
			pop = re.search("anycast\_ip\_(.+)\.txt",fn).group(1)
			if pop not in self.pops:
				continue
			these_ips = set([row.strip() for row in open(fn,'r').readlines()])
			self.all_client_addresses_from_msft = set(self.all_client_addresses_from_msft).union(these_ips)

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
		self.all_client_addresses_from_msft = list(self.all_client_addresses_from_msft)


if __name__ == "__main__":
	dmw = Deployment_Measure_Wrapper()
	dmw.recalc_probable_clients = True
	dmw.check_construct_client_to_peer_mapping(forceparse=True)	
