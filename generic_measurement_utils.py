import os, json, pickle, numpy as np, pytricia, glob, gzip, datetime, re, tqdm, copy, sys
from ripe.atlas.cousteau import AtlasRequest

sys.path.append("../")

from peering_measurements.vasilis_traceroute import Traceroute
from peering_measurements.measurement_cache import Measurement_Cache
from peering_measurements.helpers import *
from peering_measurements.config import *


class AS_Utils_Wrapper:
	"""Helper class that loads utilities for working with ASNs."""
	def __init__(self, *args, **kwargs):
		self.asn_cache_file = "ip_to_asn.csv" # we slowly build the cache over time, as we encounter IP addresses
		self.ip_to_asn = {}
		self.as_rel = {}
		self.ql = kwargs.get('quickload',False)

	def get_figure(self, nrow=1, ncol=1, fw=12,fh=7,fs=18):
		import matplotlib, matplotlib.pyplot as plt
		plt.rcParams["figure.figsize"] = [fw,fh]
		matplotlib.rc('font', **{'size': fs})
		f,ax = plt.subplots(nrow,ncol)
		return f,ax

	def save_fig(self, fn):
		"""Helper function for saving figures."""
		import matplotlib.pyplot as plt
		try:
			fn = self.campaign['plot_id'] + "-" + fn
		except AttributeError: pass
		plt.savefig(os.path.join(FIGURE_DIR,fn),bbox_inches='tight',)
		plt.clf()
		plt.close()

	def check_load_measurement_cache(self, **kwargs):
		try:
			self.mc
		except AttributeError:
			self.mc = Measurement_Cache(**kwargs)
			self.mc.mas_to_meas_id = {}
			for mid in self.mc.data["meas"]:
				for p in self.mc.data["meas"][mid]:
					mas = self.get_probe_metro_as(p)[0]
					if mas[0] is not None:
						try:
							self.mc.mas_to_meas_id[mas].append((mid, p))
						except KeyError:
							self.mc.mas_to_meas_id[mas] = [(mid,p)]

	def lookup_asns_if_needed(self, d):
		"""Looks up ASes associated with IP addresses in d. Uses cymruwhois Python library, which is slow. 
			Hence we cache answers."""
		# d is list or dict where elements are CIDR prefix strings
		dont_know = []
		for el in d:
			pref = el.split('/')[0]
			pref_24 = ip32_to_24(pref)
			if is_bad_ip(pref_24): continue # don't resolve private IP addresses
			try:
				self.ip_to_asn[pref_24]
			except KeyError:
				if self.routeviews_pref_to_asn.get(pref_24) is not None: continue
				dont_know.append(pref_24)
		if dont_know == []: return
		ret = lookup_asn(dont_know)
		for k,v in ret.items():
			self.ip_to_asn[k] = v
		self.save_ip_to_asn()	
	
	def save_ip_to_asn(self):
		"""IP to ASN mapping is saved in a Python pickle file."""
		print("Saving IP to ASN don't exit.")
		ip2asn_fn = os.path.join(CACHE_DIR, self.asn_cache_file)
		with open(ip2asn_fn, 'w') as f:
			f.write("ip\tasn\n")
			for ip,asn in self.ip_to_asn.items():
				f.write("{}\t{}\n".format(ip,asn))
		print("Done.")
		
	def parse_asn(self, ip_or_asn, with_siblings=True):
		"""Make sure you've tried to look up this IP address' ASN before calling this function."""
		# if input is IP address, converts to organzation
		# if input is ASN, converts to organization
		# if input is organization, leaves it alone
		# if we don't know the ASN of the IP address, returns None
		if ip_or_asn is None: return None
		ip_or_asn = str(ip_or_asn)
		if ip_or_asn == "": return None
		if type(ip_or_asn) == str:
			if "." in ip_or_asn:
				ip_or_asn = ip_or_asn.split("/")[0]
				# IP
				try:
					asn = self.ip_to_asn[ip32_to_24(ip_or_asn)]
				except KeyError:
					asn = self.routeviews_pref_to_asn.get(ip_or_asn + "/32")
					if asn is None: return None
			else:
				asn = ip_or_asn
		else:
			asn = ip_or_asn
		if asn is None:
			return asn
		if type(asn) == str:
			if asn.lower() == 'unknown' or asn.lower() == 'none' or asn.lower() == "na" or asn.lower() == "null":
				return None
		if int(asn) > 1e6:
			# Already converted to organization
			asn = int(asn)
		else:
			asn = str(asn)
		if with_siblings:
			try:
				asn = self.as_siblings[asn]
			except KeyError:
				pass
		return asn

	def check_load_ip_to_asn(self):
		"""Loads IP to ASN mapping if we haven't already."""
		if self.ip_to_asn != {}: return
		ip2asn_fn = os.path.join(CACHE_DIR, self.asn_cache_file)
		self.ip_to_asn = {}
		for ipaddr in ['80.249.212.38','2001:7f8:1::a502:473:1','206.72.211.9','2001:504:13::211:9','206.72.211.53','2001:504:13::211:53','218.100.9.159','2001:de8:c:2:0:2:473:1','206.71.12.56','2001:504:40:12::1:56','206.53.202.75','2001:504:61::4ff9:0:1','185.1.170.166','2001:7f8:9e::4ff9:0:1','80.81.196.21','2001:7f8::4ff9:0:1','185.1.210.159','2001:7f8:3d::4ff9:0:1','185.1.192.56','2001:7f8:a0::4ff9:0:1','103.27.171.242','2401:7500:fff6::112','185.1.208.165','2001:7f8:44::4ff9:0:1','206.82.104.55','2001:504:36::4ff9:0:1','206.197.210.63','2606:7c80:3375:50::63','202.77.90.44','2001:df0:680:6::2c','202.77.88.33','2001:df0:680:5::21','178.216.41.200','2001:7f8:5b::452','89.46.145.95','2001:678:3ac::348','208.115.137.46','2001:504:0:4:0:2:473:1','201.149.98.5','2001:504:0:11:0:2:473:1','27.111.229.119','2001:de8:4::2:473:1','185.1.107.52','2001:7f8:c1::2:473:1','203.190.230.144','2001:de8:5::2:473:1','195.182.219.135','2001:7f8:42::2:473:2','185.1.240.204','2001:7f8:12a::204','193.149.1.186','2001:7f8:f::186','38.10.0.39','2001:df2:1900:7::39','103.77.108.58','2001:df2:1900:2::58','206.41.108.73','2001:504:40:108::1:73','37.49.237.90','2001:7f8:54::1:90','187.16.214.112','2001:12f8::214:112','187.16.214.111','2001:12f8::214:111','196.60.97.11','2001:43f8:1f0::1:11','210.171.225.160','2001:de8:8::2:473:1','210.173.184.144','2001:7fa:7:2:0:2:473:1','210.173.177.85','2001:7fa:7:1:0:2:473:1','192.145.251.66','2001:7fa:8::1f','195.66.226.176','2001:7f8:4::4ff9:1','195.66.244.105','2001:7f8:4:2::4ff9:1','103.26.71.96','2001:dea:0:30::60','103.26.68.93','2001:dea:0:10::5d','196.60.10.191','2001:43f8:6d0::10:191','194.68.128.86','2001:7f8:d:fe::86','195.69.119.86','2001:7f8:d:fb::86','194.68.123.86','2001:7f8:d:ff::86','195.245.240.86','2001:7f8:d:fc::86','218.100.53.14','2001:7fa:11:4:0:4ff9:0:1','198.32.160.157','2001:504:1::a502:473:1','149.112.50.46','2001:504:125:e1::46','45.68.16.50','2801:14:9000::2:473:1','103.16.102.146','2001:de8:12:100::146','206.81.81.66','2001:504:16::4ff9','206.108.35.117','2001:504:1a::35:117','218.100.78.69','2001:7fa:11:1:0:4ff9:0:1']:
			if ":" in ipaddr: continue
			self.ip_to_asn[ip32_to_24(ipaddr)] = '20473'
		for twentyfour in ['255','238', '239', '240', '241', '242', '243']:
			self.ip_to_asn['184.164.{}.0'.format(twentyfour)] = '47065'
		if os.path.exists(ip2asn_fn):
			pbar = tqdm.tqdm(total=1500000, desc="Loading IP to ASN cache")
			ip2asn_d = csv.DictReader(open(ip2asn_fn,'r'), delimiter='\t')
			for row in ip2asn_d:
				self.ip_to_asn[row['ip']] = self.parse_asn(row['asn'], with_siblings=False)
				pbar.update(1)
			if np.random.random() > .99:
				# periodically delete ones we couldn't find just to check
				print("\nRandomly causing checks for unknown IP addresses")
				to_del = []
				for ip,asn in self.ip_to_asn.items():
					if asn == 'NA' or asn == "None":
						to_del.append(ip)
				print("Deleting {} ip addresses".format(len(to_del)))
				for ip in to_del: del self.ip_to_asn[ip]
				self.save_ip_to_asn()				
		self.ip_to_asn["*"] = None # unknown hops are associated with ASN None

		self.routeviews_asn_to_pref = {}
		self.routeviews_pref_to_asn = pytricia.PyTricia()
		# https://publicdata.caida.org/datasets/routing/routeviews-prefix2as/2022/02/
		routeviews_pref_to_as_fn = os.path.join(DATA_DIR, "routeviews-rv2-20240101-1200.pfx2as.gz")
		with gzip.open(routeviews_pref_to_as_fn) as f:
			for row in f:
				pref,l,asn = row.decode().strip().split('\t')
				asns = asn.split(",")
				for asn in asns:
					for _asn in asn.split('_'):
						_asn = self.parse_asn(_asn, with_siblings=False)
						try:
							self.routeviews_asn_to_pref[_asn].append(pref + "/" + l)
						except KeyError:
							self.routeviews_asn_to_pref[_asn] = [pref + "/" + l]

						self.routeviews_pref_to_asn[pref + "/" + l] = _asn

	def check_load_as_rel(self):
		"""Loads AS relationships if we haven't already. Downloaded from here: 
		https://homes.cs.washington.edu/~yuchenj/problink-as-relationships/
		"""
		self.check_load_siblings()
		if self.as_rel == {}:
			with open(os.path.join(DATA_DIR, "20230801.as-rel.txt"),'r') as f:
				for row in f:
					if row.startswith("#"): continue
					fields = row.strip().split('|')
					as1 = self.parse_asn(fields[0])
					as2 = self.parse_asn(fields[1])
					if fields[-1] == '0':
						self.as_rel[as1, as2] = P_TO_P # peers
						self.as_rel[as2, as1] = P_TO_P
					elif fields[-1] == '-1':
						self.as_rel[as1,as2] = P_TO_C # 0 provider of 1
						self.as_rel[as2,as1] = C_TO_P # 1 customer of 0
					else:
						# siblings, maybe do something
						pass
			# build (immediate) customer mapping from as rel
			self.as_to_customers = {}
			self.as_to_providers = {}
			for (as_1, as_2), rel in self.as_rel.items():
				if rel == P_TO_C:
					try:
						self.as_to_customers[as_1].append(as_2)
					except KeyError:
						self.as_to_customers[as_1] = [as_2]
					try:
						self.as_to_providers[as_2].append(as_1)
					except KeyError:
						self.as_to_providers[as_2] = [as_1]
				elif rel == C_TO_P:
					try:
						try:
							self.as_to_customers[as_1]
						except KeyError:
							pass
						if as_1 == as_2: continue
						self.as_to_customers[as_2].append(as_1)
					except KeyError:
						self.as_to_customers[as_2] = [as_1]
					try:
						self.as_to_providers[as_1].append(as_2)
					except KeyError:
						self.as_to_providers[as_1] = [as_2]
			for k in self.as_to_customers:
				self.as_to_customers[k] = list(set(self.as_to_customers[k]))
			for k in self.as_to_providers:	
				self.as_to_providers[k] = list(set(self.as_to_providers[k]))

	def update_cc_cache(self):
		"""Updates customer cone cache with ASNs for which we don't already have the CC calculated."""
		self.check_load_as_rel()
		cc_cache_fn = os.path.join(CACHE_DIR, "customer_cone_cache.pkl")
		if not os.path.exists(cc_cache_fn):
			self.cc_cache = {}
		else:
			self.cc_cache = pickle.load(open(cc_cache_fn, 'rb'))
		self.check_load_ip_to_asn()
		all_asns = set(self.ip_to_asn.values()).union(list(self.cc_cache))
		for a in all_asns:
			self.get_cc(a)
		pickle.dump(self.cc_cache, open(cc_cache_fn, 'wb'))	

	def get_cc(self, asn, dbg=False):
		"""Get customer cone of an asn, according to our AS relationship data."""
		found = {}
		def get_customers(_asn, dbg=dbg):
			ret = [_asn]
			try:
				if not dbg:
					ret = ret + self.cc_cache[_asn]
				else:
					raise KeyError
			except KeyError:
				try:
					for customer in self.as_to_customers[_asn]:
						try:
							found[customer]
							continue
						except KeyError:
							pass
						if dbg:
							print("Adding customer {}, parent is {}".format(customer, _asn))
						found[customer] = None
						this_asn_cc = get_customers(customer,dbg=dbg)
						ret = ret + this_asn_cc
				except KeyError:
					pass
				ret = list(set(ret))
				self.cc_cache[_asn] = ret
			return list(set(ret))
		return get_customers(asn)	

	def check_load_valid_bgp_paths(self, **kwargs):
		try:
			self.valid_bgp_paths
		except AttributeError:
			self.bgp_paths_cache_fn = os.path.join(CACHE_DIR, "ping_campaign_valid_bgp_paths.csv")
			if os.path.exists(self.bgp_paths_cache_fn):
				self.valid_bgp_paths = load_bgp_paths_ping_campaign(self.bgp_paths_cache_fn, **kwargs)
			else:
				self.valid_bgp_paths = {}
		self.check_load_bmp_cone(**kwargs)

	def get_as_rel(self, as1, as2):
		"""Returns AS relationship between AS1 and AS2 (or their corresponding orgs if we have that 
			data). Throws a KeyError if we don't know it."""
		as1 = self.parse_asn(as1)
		as2 = self.parse_asn(as2)
		rel = self.as_rel[as1,as2]
		return rel
	
	def check_load_siblings(self):
		""" Loads siblings file, creates mapping of asn -> organization and vice-versa.
			Useful for checking if two ASNs are basically the same."""

		# In most places, we treat ASes as the same if they are siblings (\ie owned by the same organization)
		# We treat siblings the same since treating siblings separatetly would make the logic for various things
		# like calculating if a route is Valley Free uglier
		try:
			self.as_siblings
			return
		except AttributeError:
			pass
		print("Loading siblings")
		self.as_siblings = {}
		uid = int(1e6) # each organization is defined as an integer > 1e6 since real ASNs are less than 1e6
		def check_add_siblings(sib1, sib2, uid,v=0):
			# Check to see if we have created sibling groups for either of these AS's
			have_s1 = False
			have_s2 = False
			try:
				self.as_siblings[sib1]
				have_s1 = True
			except KeyError:
				pass
			try:
				self.as_siblings[sib2]
				have_s2 = True
			except KeyError:
				pass
			# if we haven't seen these ASes before, form a new organization for these ASes
			if not have_s1 and not have_s2:
				self.as_siblings[sib1] = uid
				self.as_siblings[sib2] = uid
				uid += 1
			elif have_s1: # S1 and S2 are siblings -- update our data structure
				this_sib_uid = self.as_siblings[sib1]
				self.as_siblings[sib2] = this_sib_uid
			else:
				this_sib_uid = self.as_siblings[sib2]
				self.as_siblings[sib1] = this_sib_uid
			return uid
		# It is important that these files stay the same, and are loaded in the same order, or else we have to recalculate lots of things in the cache
		self.siblings_fns = ["vasilis_siblings_20200816.txt","as_relationships_20210201.txt",'microsoft_siblings_20220606.json'] # from Vasilis Giotsas
		if False:
			siblings_fn = os.path.join(DATA_DIR, self.siblings_fns[0])
			with open(siblings_fn, 'r') as f:
				for row in f:
					sib_1, sib_2 = row.strip().split(' ')
					uid = check_add_siblings(sib_1,sib_2,uid)
			siblings_fn = os.path.join(DATA_DIR, self.siblings_fns[1])
			with open(siblings_fn,'r') as f:
				# add the siblings from as relationships file as well
				for row in f:
					fields = row.strip().split('|')
					if fields[-1] != '1': continue
					sib_1, sib_2 = fields[0:2]
					uid = check_add_siblings(sib_1,sib_2,uid)
		siblings_fn = os.path.join(DATA_DIR, self.siblings_fns[2])
		siblings_data_obj = json.load(open(siblings_fn,'r')) # NOTE -- Microsoft siblings data provides no new information afaik
		for k,v in siblings_data_obj.items():
			as0 = v['siblingAs'][0]
			if int(as0) == 20473: continue
			for as1 in v['siblingAs'][1:]:
				if int(as1) == 20473: continue
				uid = check_add_siblings(str(as0), str(as1), uid,v=1)
		# form the inverse image of the mapping
		self.org_to_as = {}
		for sib_as, org_id in self.as_siblings.items():
			try:
				self.org_to_as[org_id]
			except KeyError:
				self.org_to_as[org_id] = []
			self.org_to_as[org_id].append(sib_as)

		self.msft_org = self.as_siblings.get('8075', '8075')

class Atlas_Wrapper(AS_Utils_Wrapper):
	"""Helper functions associated mostly with parsing RIPE Atlas measurements."""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.all_active_probes = []
		self.probe_to_front_end = False
		self.useable_probes = None
		self.ignore_dc_numbers = IGNORE_DC_NUMBERS
		self.dns_ip_to_fe = {}

	def load_traceroute_helpers(self):
		# Traceroute parsing helper class from Vasilis Giotsas
		self.check_load_ip_to_asn()
		self.tr = Traceroute(ip_to_asn = self.parse_asn)
		ixp_ip_members_file = os.path.join(DATA_DIR,"ixp_members.merged-20190826.txt") # from Vasilis Giotsas
		ixp_prefixes_file = os.path.join(DATA_DIR,"ixp_prefixes.merged-20190826.txt") # from Vasilis Giotsas
		self.tr.read_ixp_ip_members(ixp_ip_members_file)
		self.tr.read_ixp_prefixes(ixp_prefixes_file)

	def get_all_active_ripe_probes(self, too_old = 1):
		"""Gets all active probes from RIPE Atlas. Updates the list if 
			the last fetch was more than too_old days ago."""
		self.check_load_ip_to_asn()
		if self.all_active_probes != []:
			return
		probe_pkl_fns = glob.glob(os.path.join(CACHE_DIR, 'active_probes', "active_probes_*.pkl"))
		update_active_probes = True
		today = datetime.datetime.today()
		today = (today.year,today.month,today.day)
		# Get the date we last fetched probes, and see if we need to fetch them again
		if probe_pkl_fns == []:
			old_active_probes = []
		else:
			youngest = [None, None]
			for probe_pkl_fn in probe_pkl_fns:
				last_datetime = re.search("active\_probes\_(.+)\_(.+)\_(.+).pkl",
					probe_pkl_fn)
				last_datetime_year = int(last_datetime.group(1))
				last_datetime_month = int(last_datetime.group(2))
				last_datetime_day = int(last_datetime.group(3))
				dlta = datetime.date(today[0], today[1], today[2]) - datetime.date(
					last_datetime_year, last_datetime_month, last_datetime_day)
				if youngest[0] is None:
					youngest[0] = dlta
					youngest[1] = probe_pkl_fn
				else:
					if dlta < youngest[0]:
						youngest[0] = dlta
						youngest[1] = probe_pkl_fn
				if dlta.days <= too_old:
					update_active_probes = False
				
			if update_active_probes:
				old_active_probes = pickle.load(open(youngest[1], 'rb'))
		
		# Pull the active probes from RIPE Atlas
		if update_active_probes:
			next_path = "/api/v2/probes/?page_size=100000"
			while True:
				print("Fetching probes at : {}".format(next_path))
				request = AtlasRequest(**{"url_path": next_path})
				(is_success, response) = request.get()
				for probe in response['results']:
					if probe["status"]['name'] == "Connected":
						if probe['address_v4'] is None: continue
						probe['active'] = True
						self.all_active_probes.append(probe)
				if response["next"] is not None:
					next_path = response["next"][22:]
				else:
					break
			if probe_pkl_fn != []:
				# delete the old set of active probes
				call("del {}".format(probe_pkl_fn), shell=True)
			probe_pkl_fn = os.path.join(CACHE_DIR, 'active_probes', "active_probes_{}_{}_{}.pkl".format(
				today[0], today[1], today[2]))

			# there may be probes that used to be active (from which we've executed measurements)
			# but are no longer active
			# add in these inactive probes, but mark them as inactive
			active_prb_ids = set([p['id'] for p in self.all_active_probes])
			old_prb_ids = set([p['id'] for p in old_active_probes])
			now_inactive = get_difference(old_prb_ids, active_prb_ids)
			for in_id in now_inactive:
				prb_obj = [p for p in old_active_probes if p['id'] == in_id][0]
				prb_obj['active'] = False
				self.all_active_probes.append(prb_obj)

			# cache the active probes for later
			pickle.dump(self.all_active_probes, open(probe_pkl_fn, 'wb'))
		else:
			probe_pkl_fn = youngest[1]
			# load active probes from the cache
			self.all_active_probes = pickle.load(open(probe_pkl_fn, 'rb'))

		# see if we need to update IP to asn mapping
		probe_networks = self.get_active_probe_networks()
		self.lookup_asns_if_needed(probe_networks)

	def get_active_probe_networks(self):
		"""Gets active probe /24 and writes to a txt file."""
		self.get_all_active_ripe_probes()
		self.prune_probes()
		ntwrks = {}
		for p in self.useable_probes['v4']:
			prefix = p['prefix_v4']
			if prefix is None: continue
			ntwrks[prefix] = None
		ntwrs=list(ntwrks.keys())
		with open(os.path.join(DATA_DIR, "active_probe_prefixes.txt"), 'w') as f:
			# For use in Kusto queries
			f.write("(")
			for i,ntwrk in enumerate(ntwrs):
				f.write("\"{}\"".format(ntwrk))
				if i < len(ntwrs) - 1:
					f.write(",")
			f.write(")")
		return ntwrs

	def get_probes_within(self, loc, dist):
		"""Gets active probes within dist km of loc."""
		ret = []
		best_so_far = None
		min_dist = np.inf
		for p in self.useable_probes['v4']:
			if not p['active']: continue
			probe_loc = p['geometry']['coordinates']
			lon,lat = probe_loc
			probe_loc = (lat,lon)
			d_prob_loc = geopy.distance.geodesic(loc,probe_loc).km
			if geopy.distance.geodesic(loc,probe_loc).km <= dist:
				ret.append(p)
			if d_prob_loc < min_dist:
				best_so_far = p
				min_dist = d_prob_loc
		if ret == []:
			return [best_so_far]
		return ret

	def get_probe_by_id(self, prb_id, ip_v='v4'):
		"""Each RIPE Atlas probe has an ID. Retrieve the probe object from the class object
			useable_probes with the corresponding ID prb_id, if it exists."""

		try:
			self.pid_to_obj_cache
		except AttributeError:
			self.pid_to_obj_cache = {}
		try:
			return self.pid_to_obj_cache[prb_id,ip_v]
		except KeyError:
			pass
		self.get_all_active_ripe_probes()
		self.prune_probes()
		try:
			obj = [p for p in self.useable_probes[ip_v] if p['id'] == prb_id][0]
		except IndexError:
			# we couldn't find this probe, which likely means its gone inactive
			obj = None

		self.pid_to_obj_cache[prb_id,ip_v] = obj

		return obj

	def parse_ripe_trace_result(self, result, ret_ips=False):
		"""Parse traceroute measurement result from RIPE Atlas into a nice form."""
		if result == "error": return None
		src,dst = result['from'], result['dst_addr']
		if src == "" or dst == "": return None

		ret = {}
		try:
			l_path = len(result['result'][0]['result'])
		except KeyError:
			# error in measuring
			return None
		raw_ripe_paths = []
		hop_rtts = []
		# Extract hops and RTTs
		for el in result['result']:
			this_arr = []
			this_rtt_arr = []
			for i in range(len(el['result'])):
				try:
					if 'rtt' not in el['result'][i]: continue
					these_hops = [el['result'][i]['from']]
					these_hops = [el for el in these_hops if not is_bad_ip(el)]
					if len(these_hops) == 0:
						continue
					this_arr.append(these_hops[0])
					this_rtt_arr.append(el['result'][i]['rtt'])
				except KeyError:
					# unresponsive hop on this measurement
					this_arr.append("*")
			if len(this_arr) == 0:
				continue
			hop_rtts.append(this_rtt_arr)
			raw_ripe_paths.append(this_arr)
		if len(raw_ripe_paths) == 0:
			# Bad measurement
			return None
		# prepend/append the source/dest if we haven't
		# we know if the traceroute didn't reach the destination if rtts[-1] == []
		if src not in set(raw_ripe_paths[0]):
			raw_ripe_paths = [[src]] + raw_ripe_paths
			hop_rtts = [[0]] + hop_rtts
		if dst not in set(raw_ripe_paths[-1]):
			raw_ripe_paths = raw_ripe_paths + [[dst]]
			hop_rtts = hop_rtts + [[]]
			ret['reached_dst'] = False
		else:
			ret['reached_dst'] = True
		every_ip = list(set([ip for hop_set in raw_ripe_paths for ip in hop_set]))
		if ret_ips: return every_ip
		# Calculate the AS path
		# uses the Traceroute class utility function which does things like remove ASes associated with IXPs
		try:
			asn_path = self.tr.ip_to_asn(raw_ripe_paths)
		except KeyError:
			still_need = get_difference(every_ip, self.ip_to_asn)
			self.lookup_asns_if_needed(still_need)
			asn_path = self.tr.ip_to_asn(raw_ripe_paths)

		ret["ip_paths"] = raw_ripe_paths
		ret["rtts"] = hop_rtts
		for i in range(len(asn_path)):
			if asn_path[i] is None:
				asn_path[i] = "None"
			else:
				asn_path[i] = str(asn_path[i])
		ret["as_paths"] = asn_path
		ret['time'] = result['timestamp']
		
		return ret

	def prune_probes(self):
		"""Looks through active probes, and limits probes to 
			ones we can use for our measurements."""

		# right now this is just making sure we have a valid location

		# look for valid location, separate into v4 and/or v6 capable
		if self.useable_probes is not None: return
		self.useable_probes = {
			"v4": [], "v6": []
		}

		# if a probe is inactive, we still mark that as 'useable'
		# but don't print stats for it
		for probe in self.all_active_probes:
			if probe["geometry"] is None: continue # need to know where the probe is
			if probe['address_v4'] is not None or probe['prefix_v4'] is not None:
				# Make prefix and address the same, preferring to set address to prefix since
				# address is more likely to be useless (private IP space, etc.)
				self.useable_probes['v4'].append(probe)
				if probe['prefix_v4'] is not None and probe['address_v4'] is None:
					probe['address_v4'] = probe['prefix_v4'].split('/')[0]
				elif probe['prefix_v4'] is None and probe['address_v4'] is not None:
					probe['prefix_v4'] = ip32_to_24(probe['address_v4']) + "/24"
			if probe['address_v6'] is not None or probe['prefix_v6'] is not None:
				self.useable_probes['v6'].append(probe)
				if probe['prefix_v6'] is not None and probe['address_v6'] is None:
					probe['address_v6'] = probe['prefix_v6'].split('/')[0]
				elif probe['prefix_v6'] is None and probe['address_v6'] is not None:
					probe['prefix_v6'] = ip32_to_24(probe['address_v6']) + "/24"
		for k in self.useable_probes:
			these_prbs_active = [p for p in self.useable_probes[k] if p['active']]
			n_networks = len(set([p['asn_v4'] for p in these_prbs_active]))
			n_countries = len(set([p['country_code'] for p in these_prbs_active]))
			print("{} - {} total probes in {} networks and {} countries".format(
				k, len(these_prbs_active), n_networks, n_countries
			))

		self.check_load_probe_to_front_end()


if __name__ == "__main__":
	aw = Atlas_Wrapper()
