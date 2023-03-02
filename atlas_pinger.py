### This file contains the following lower-level utilities
# a) executing measurements with RIPE ATlas
# b) pulling results of RIPE Atlas measurements
# c) data loading



from ripe.atlas.cousteau import (AtlasResultsRequest, ProbeRequest, AtlasRequest, 
	Traceroute as RipeTraceroute, AtlasSource, AtlasCreateRequest, MeasurementRequest, Probe, Ping)
import glob, os, pickle, re, geopy.distance, json, numpy as np, copy, tqdm, time
# plt.rcParams.update({"text.usetex": True})
# import matplotlib.pyplot as plt # Takes a while, so only do it if you really need to


from subprocess import call
from constants import *
from helpers import *
from measurement_cache import Measurement_Cache
from painter_helper_classes import Atlas_Wrapper


ATLAS_API_KEY = "b4530677-a25b-437e-b3dc-774066c9110d"
ATLAS_STOP_KEY = "53c5ed1e-24eb-457f-bb62-ba91f161987b"

# Constraints we have to work with imposed by RIPE Atlas
MAX_PROBES_PER_MEAS = 100
MAX_MEAS_PER_DST = 20
MAX_ACTIVE_MEAS = 95


def get_relevant_measurements(description_uids, oldness=None):
	"""Gets measurement IDs associated with measurements whose descriptions contain one of 
		description_uids."""
	relevant_measurements = []
	path = "/api/v2/measurements/my?key={}&start_time__gt={}&page_size=500".format(ATLAS_API_KEY, int(time.time() - oldness))
	request = AtlasRequest(**{"url_path": path})
	(is_success, response) = request.get()
	if not is_success: print(response); exit(0)
	while True:
		for el in response["results"]:
			if oldness is not None:
				if el['start_time'] < time.time() - oldness: continue
			for description_uid in description_uids:
				if description_uid in el["description"]:
					relevant_measurements.append(el['id'])
		if response["next"] is None: break
		path = response["next"][len("https://atlas.ripe.net"):]
		request = AtlasRequest(**{"url_path": path})
		(is_success, response) = request.get()
		if not is_success: print(response); exit(0)
	return list(set(relevant_measurements))

class Atlas_Pinger(Atlas_Wrapper):
	"""Class with utility functions for executing measurements, pulling measurement results, 
		and executing larger "campaigns" of measurements."""
	def __init__(self):
		super().__init__()

	def check_wait_execute(self, cfg):
		""".
			Tries to execute the measurement set by cfg. Waits until RIPE Atlas will allow the measurement
			given various rate limits.
			cfg needs all the keys required by execute_case_study_measurments.
		"""
		# Assume one target, since RIPE Atlas measurements are executed by target
		# RIPE Atlas annoyingly takes a while to provide updated numbers of active measurements, so we sleep
		# to give it time to do so
		dst = cfg['targets_v4'][0]
		# Check to see if we can afford to execute this measurement
		active_meas = self.get_active_measurements()
		# CHECKS
		# are there already MAX_MEAS_PER_DST to this dst running?
		while len([meas for meas in active_meas if meas['target'] == dst]) >= MAX_MEAS_PER_DST:
			time.sleep(30)
			active_meas = self.get_active_measurements()
		# are there already MAX_ACTIVE_MEAS running?
		while len(active_meas) >= MAX_ACTIVE_MEAS:
			time.sleep(30)
			active_meas = self.get_active_measurements()
		try:
			self.execute_case_study_measurements(cfg, verb=False)
		except: # Likely a credits issue --- exit to be safe
			import traceback
			traceback.print_exc()
			exit(0)

	def load_measurement_configs(self, config_name):
		"""Loads measurement campaign configs specified in the 'configs' directory."""
		self.measurement_configs = []
		if config_name is None:
			for cfg_fn in glob.glob(os.path.join(CONFIG_DIR, "*.json")):
				self.measurement_configs.append(json.load(open(cfg_fn, 'r')))
		else:
			config_path = os.path.join(CONFIG_DIR, "{}.json".format(config_name))
			self.measurement_configs.append(json.load(open(config_path, 'r')))

	def execute_longitudinal_painter_assessement(self):
		"""For each RIPE Atlas probe, launch measurement to the anycast address and 
			best peering interfaces for that probe."""
		# Put this call in a cron-job to make things easier
		

		ret = self.get_microsoft_peering_addresses(tolerance=TOLERANCE)
		pop_peer_to_dst = ret['pop_peer_to_dst']
		MSMT_FREQUENCY = 1*24*3600

		self.pull_results(recent=7*24*3600, long_load=True)
		# max number of destinations we can measure to per probe
		# this number is limited by our credit spending limit per day
		# limit ~ 1e6 / (NUM_PROBES * (ping per meas)) - 1 (minus 1 for traceroute)
		MAX_N_PER_PROBE = 10 

		from painter_adv_orchestrator import Painter_Adv_Orchestrator, mas_to_probes_fn
		real_mas_latencies_fn = Painter_Adv_Orchestrator.get_real_meas_fn(ignore_anycast=False)
		
		mas_perfs = {}
		for row in csv.DictReader(open(real_mas_latencies_fn,'r'),delimiter='\t'):
			metro,asn = row['metro'], self.parse_asn(row['asn'])
			intf = row['intf']
			if 'anycast' in intf: 
				intf = 'anycast'
			else:
				intf = intf.split('-')
				if intf[1] in ['hybridcast','regional']:
					continue # ignore these cases for now
				intf[1] = self.parse_asn(intf[1])
				intf = tuple(intf)
			l = float(row['l'])
			try:
				mas_perfs[metro,asn][intf] = l
			except:
				mas_perfs[metro,asn] = {intf: l}
		best_pop_peer_by_mas = {}
		for mas in mas_perfs:
			intfs = list(mas_perfs[mas])
			perfs = [mas_perfs[mas][intf] for intf in intfs]
			best_perfs = list(sorted(zip(intfs,perfs), key = lambda el : el[1]))
			best_pop_peer_by_mas[mas] = [el[0] for el in best_perfs[0:MAX_N_PER_PROBE]]

		mas_to_probes = csv.DictReader(open(mas_to_probes_fn,'r'),delimiter='\t')
		mas_to_probes = {(row['metro'], self.parse_asn(row['asn'])): list(map(int, row['probes'].split('--'))) for row in mas_to_probes}

		cfg = {
			"campaign_type": "case-study",
			"meas_type": "ping",
			"targets_v4": [],
			"probes": [],
			"n_packet_per_meas": 7,
		}
		dst_to_probe_map = {}
		for mas in mas_to_probes:
			try:
				best_pop_peers = best_pop_peer_by_mas[mas]
			except KeyError:
				continue
			for best_pop_peer in best_pop_peers:
				if best_pop_peer == 'anycast': continue # we measure here anyway
				# Convert optimal interface (pop,peer) to a destination IP Address
				addr = pop_peer_to_dst[best_pop_peer]
				probes = mas_to_probes[mas]
				np.random.shuffle(probes) # I shuffle them in case some are unresponsive at some times
				for probe in probes:
					try:
						dst_to_probe_map[addr].append(probe)
					except KeyError:
						dst_to_probe_map[addr] = [probe]
					break # only do one per MAS
		all_probes = [p for probes in mas_to_probes.values() for p in probes]
		anycast_address = GET_ANYCAST_ADDRESSES()[0]
		dst_to_probe_map[anycast_address] = []
		for probe in list(set(all_probes)):
			dst_to_probe_map[anycast_address].append(probe)

		dsts_to_query = sorted(dst_to_probe_map, key = lambda dst : -1 * len(dst_to_probe_map[dst]))
		n_to_execute = sum(1 for dst in dst_to_probe_map for prb in dst_to_probe_map[dst])
		print("{} measurements to execute pre prune.".format(n_to_execute))
		for i,dst in tqdm.tqdm(enumerate(dst_to_probe_map),desc="Checking to make sure measurements don't already exist"):
			to_del = []
			for j,prb in enumerate(dst_to_probe_map[dst]):
				prb_obj = self.get_probe_by_id(prb)
				if prb_obj is None: 
					to_del.append(j)
					continue # inactive
				mas = self.get_probe_metro_as(prb)[0]
				if mas[0] is None: 
					# no point measuring if we don't know MAS
					to_del.append(j)
					continue
				all_meas = self.mc.get_meas(prb, dst, _type='ping',match_on='probe')
				if len(all_meas) == 0: continue # haven't measured here yet
				most_recent_meas = all_meas[np.argmax(np.array([meas['timestamp'] for meas in all_meas]))]

				delta = time.time() - most_recent_meas['timestamp']
				if delta < MSMT_FREQUENCY:
					# delete measurements that occurred too recently
					to_del.append(j)
			for j in reversed(to_del):
				del dst_to_probe_map[dst][j]
		for dst in list(dst_to_probe_map.keys()):
			if dst_to_probe_map[dst] == []:
				del dst_to_probe_map[dst]


		dsts_to_query = sorted(dst_to_probe_map, key = lambda dst : -1 * len(dst_to_probe_map[dst]))
		n_to_execute = sum(1 for dst in dst_to_probe_map for prb in dst_to_probe_map[dst])
		print("{} measurements to execute after pruning.".format(n_to_execute))
		upd = tqdm.tqdm(desc="Executing measurements...", total=n_to_execute)

		for dst in dsts_to_query:
			cfg['targets_v4'] = [dst]
			probes = dst_to_probe_map[dst]
			if len(probes) > MAX_PROBES_PER_MEAS:
				probes_chunks = split_seq(probes, int(np.ceil(len(probes) / MAX_PROBES_PER_MEAS)))
			else:
				probes_chunks = [probes]
			for probes_chunk in probes_chunks:
				cfg['probes'] = probes_chunk
				self.check_wait_execute(cfg)
			upd.update(len(probes))

	def execute_oneoff_painter_assessment(self, **kwargs):
		"""Gets latency from probes to peers at all PoPs."""
		np.random.seed(31415) # seed this so we can start and stop
		self.check_load_ip_to_asn()
		self.check_load_as_rel()
		self.get_fe_to_pop()
		self.load_peering_data()

		CAMPAIGN_NAME = ['unicast_pop_heuristic', 'regional', 'hybridcast'][0]
		print("DOING CAMPAIGN : {}".format(CAMPAIGN_NAME))
		DO_VALID_PATH_CHECK = (CAMPAIGN_NAME == 'unicast_pop_heuristic')

		self.pull_results(recent=24*2*3600, long_load=True)
		dst_to_pop = {}

		# Get set of dsts -> probes
		# Remove existing measurements
		dst_to_probe_map = {}
		for pop in self.pops:
			json_fn = "{}_{}.json".format(pop, CAMPAIGN_NAME)
			if not os.path.exists(os.path.join(CONFIG_DIR, json_fn)): continue
			cfg = json.load(open(os.path.join(CONFIG_DIR, json_fn),'r'))
			cfg_copy = copy.deepcopy(cfg)
			for dst in cfg['targets_v4']:
				dst_to_pop[dst] = pop
				try:
					dst_to_probe_map[dst]
				except KeyError:
					dst_to_probe_map[dst] = []
				dst_to_probe_map[dst] += cfg['probes']
		for dst in dst_to_probe_map:
			dst_to_probe_map[dst] = list(set(dst_to_probe_map[dst]))
		
		if DO_VALID_PATH_CHECK:
			src_to_dst = {}
			src_to_mas = {}
			for i,dst in enumerate(dst_to_probe_map):
				for i,prb in enumerate(dst_to_probe_map[dst]):
					probe_mas = self.get_probe_metro_as(prb)
					if probe_mas[1] is None: continue
					prb_obj = self.get_probe_by_id(prb)
					prb_src_addr = prb_obj['prefix_v4'].split('/')[0]
					src_to_mas[prb_src_addr] = probe_mas[0]
					try:
						src_to_dst[prb_src_addr].append(dst)
					except KeyError:
						src_to_dst[prb_src_addr] = [dst]
			valid_src_dst_pairs = self.precheck_valid_bgp_path(src_to_dst, src_to_mas, dst_to_pop)

		## TODO -- this won't work for hybrid cast etc..
		wanted_to = []
		for i,dst in tqdm.tqdm(enumerate(dst_to_probe_map),
			desc="Checking to make sure measurements don't already exist"):
			to_del = []
			for j,prb in enumerate(dst_to_probe_map[dst]):
				prb_obj = self.get_probe_by_id(prb)
				prb_mas = self.get_probe_metro_as(prb)[0]

				if prb_mas[0] is None: 
					to_del.append(j)
					continue
				prb_src_addr = prb_obj['prefix_v4'].split('/')[0]
				try:
					if DO_VALID_PATH_CHECK:
						if dst not in valid_src_dst_pairs[prb_src_addr]: 
							raise KeyError
						else:
							# We want to execute a measurement here, unless its been executed recently
							wanted_to.append((prb, dst))
					if self.mc.meas_exists(prb_mas,dst,_type='ping',match_on='mas'):
						# delete pre-existing measurements
						to_del.append(j)
				except KeyError:
					# delete cases for which no VF path exists
					to_del.append(j)
			for j in reversed(to_del):
				del dst_to_probe_map[dst][j]
		for dst in list(dst_to_probe_map.keys()):
			if dst_to_probe_map[dst] == []:
				del dst_to_probe_map[dst]
		if kwargs.get('ret_wanted_to', False):
			# Only calculate what we wanted to probe
			return wanted_to


		# Execute largest number of probes first, since there are a lot of unresponsive straggler probes
		#  that take up budget in the measurement queue
		dsts_to_query = sorted(dst_to_probe_map, key = lambda dst : -1 * len(dst_to_probe_map[dst]))
		# Now execute ping measurements dst by dst
		n_to_execute = sum(1 for dst in dst_to_probe_map for prb in dst_to_probe_map[dst])
		print("Starting meas, {} to execute, {} dsts".format(n_to_execute, len(dsts_to_query)))
		# do the traceroute to the anycast address
		all_probes = list(set([prb for dst in dst_to_probe_map for prb in dst_to_probe_map[dst]]))
		cfg = json.load(open(os.path.join(CONFIG_DIR, 'anycast_tr_all_{}_probes.json'.format(CAMPAIGN_NAME)),'r'))
		probes = get_intersection(cfg['probes'], all_probes)
		anycast_targ = cfg['targets_v4'][0]
		to_del = []
		for i,prb in enumerate(probes):
			prb_mas = self.get_probe_metro_as(prb)[0]
			if prb_mas[0] is None: continue
			if self.mc.meas_exists(prb_mas,anycast_targ, 
				match_on='mas'): to_del.append(i)
		for i in reversed(to_del):
			del probes[i]
		if len(probes) > 0:
			cfg['probes'] = probes
			probes_chunks = split_seq(cfg['probes'], int(np.ceil(len(probes) / MAX_PROBES_PER_MEAS)))
			for probes_chunk in probes_chunks:
				cfg['probes'] = probes_chunk
				self.check_wait_execute(cfg)
		
		dsts_to_query = sorted(dst_to_probe_map, key = lambda dst : -1 * len(dst_to_probe_map[dst]))
		n_to_execute = sum(1 for dst in dst_to_probe_map for prb in dst_to_probe_map[dst])
		print("{} measurements to execute after pruning.".format(n_to_execute))
		upd = tqdm.tqdm(desc="Executing measurements...", total=n_to_execute)

		for dst in dsts_to_query:
			cfg_copy['targets_v4'] = [dst]
			probes = dst_to_probe_map[dst]
			if len(probes) > MAX_PROBES_PER_MEAS:
				probes_chunks = split_seq(probes, int(np.ceil(len(probes) / MAX_PROBES_PER_MEAS)))
			else:
				probes_chunks = [probes]
			for probes_chunk in probes_chunks:
				cfg_copy['probes'] = probes_chunk
				self.check_wait_execute(cfg_copy)
				n_to_execute -= len(probes_chunk)
			upd.update(len(probes))

	def get_cfg_case_study_from_loc(self, cfg):
		if len(cfg.get('probes',[])) > 0:
			# already populated
			return cfg
		loc = cfg['loc']
		dst = cfg['dist']
		probes_of_interest = self.get_probes_within(loc, dst)
		cfg['probes'] = [p['id'] for p in probes_of_interest]
		if cfg['probes'] == []: 
			print("No probes found matching config spec.")
			return False
		return cfg

	def execute_case_study_from_loc(self, cfg, check_exists=False):
		cfg = self.get_cfg_case_study_from_loc(cfg)
		if not cfg:
			exit(0)
		if check_exists:
			# remove probes for which we've measured to all dsts
			to_del = []
			for i,probe in enumerate(cfg['probes']):
				if sum(self.mc.meas_exists(probe, dst, _type=cfg['meas_type']) for dst in cfg['targets_v4']) == len(cfg['targets_v4']):
					to_del.append(i)
			for p in reversed(to_del):
				del cfg['probes'][i]
		if 'max_n_probes' in cfg:
			np.random.shuffle(cfg['probes'])
			cfg['probes'] = cfg['probes'][0:cfg['max_n_probes']]
		self.execute_case_study_measurements(cfg)
		return cfg

	def execute_case_study_measurements(self, config, verb=True): 
		"""Execute small-scale one-off measurements of a specific type."""
		# useful for case studies of a specific probe
		prb_ids = config['probes']
		sources = [AtlasSource(type="probes", value=_id, requested=5) for 
			_id in prb_ids]
		n_rounds_required = int(np.ceil(len(sources) / 100))
		if n_rounds_required == 1:
			source_chunks = [sources]
		else:
			source_chunks = split_seq(sources, n_rounds_required)
		targets = config['targets_v4']
		for source in source_chunks:
			if config['meas_type'] == 'ping':
				for t in targets:
					is_success = False
					while not is_success:
						if verb:
							print("Launching ping one-off to address: {}.".format(t))
						png = Ping(
							af=4, 
							target=t, 
							description="io-case-study -- Ping towards address, {}.".format(t),
							packets=config['n_packet_per_meas'],
						)
						atlas_request = AtlasCreateRequest(
							key=ATLAS_API_KEY,
							measurements=[png],
							is_oneoff=True,
							sources=source,
						)
						is_success, response = atlas_request.create()
						if not is_success:
							if "start time in future" in response['error']['errors'][0]['detail']:
								continue
							else:
								print("Did not succesfully execute measurement : {}".format(response))
								exit(0)
			elif config['meas_type'] == 'traceroute':
				for t in targets:
					if verb:
						print("Launching traceroute one-off to address: {}.".format(t))
					trcrt = RipeTraceroute(
						af=4,
						target=t,
						protocol="ICMP",
						description="io-case-study -- Traceroute towards address, {}".format(t)
					)
					atlas_request = AtlasCreateRequest(
						key=ATLAS_API_KEY,
						measurements=[trcrt],
						is_oneoff=True,
						sources=source,
					)
					is_success, response = atlas_request.create()
					if not is_success:
						print("Did not succesfully execute measurement : {}".format(response))
						exit(0)
			else:
				raise ValueError("Measurement type {} not yet implemented.".format(config['type']))
	
	def get_active_measurements(self):
		"""Gets currently running measurements."""
		results = []
		path = "/api/v2/measurements/my?key={}&status=2&page_size=500".format(ATLAS_API_KEY)
		request = AtlasRequest(**{"url_path": path})
		(is_success, response) = request.get()
		if not is_success: print(response); exit(0)
		while True:
			for el in response['results']:
				results.append(el)
			if response["next"] is None: break
			path = response["next"][len("https://atlas.ripe.net"):]
			request = AtlasRequest(**{"url_path": path})
			(is_success, response) = request.get()
			if not is_success: print(response); exit(0)
		return results

	def pull_results(self, recent=24*3600*7, **kwargs):
		"""Look for recent measurements with description containing one of description_uids. Pull
			them and add them to our RIPE Atlas measurement cache."""

		self.check_load_measurement_cache(**kwargs)

		relevant_measurements = []
		# status = 4 means it is a stopped measurement
		path = "/api/v2/measurements/my?key={}&start_time__gt={}&status=4&page_size=500".format(ATLAS_API_KEY,int(time.time()-recent))
		request = AtlasRequest(**{"url_path": path})
		(is_success, response) = request.get()
		if not is_success: print(response); exit(0)
		# a relevant measurement is a measurement (a) we don't have cached and (b) whose
		# description contains one of the description_uids
		description_uids = ["Check Traceroute towards anycast address", 
			"io -- Ping towards FE,", "io-case-study", "io -- all_fe PI Check Traceroute"]
		msmt_id_to_desc = {}
		print("Checking if we need to pull results.")
		while True:
			for el in response["results"]:
				for description_uid in description_uids:
					if description_uid in el["description"]:
						# check if this measurement is in our cache
						if self.mc.exists(el): continue
						# not in the cache, indicate we need to pull it
						relevant_measurements.append(el['id'])
						msmt_id_to_desc[el['id']] = el['description'] # save description
			if response["next"] is None: break
			path = response["next"][len("https://atlas.ripe.net"):]
			request = AtlasRequest(**{"url_path": path})
			(is_success, response) = request.get()
			if not is_success: print(response); exit(0)
		# pull all the measurements we dont have yet
		for i, msmt_id in tqdm.tqdm(enumerate(set(relevant_measurements)), desc="Pulling results..."):
			path = "/api/v2/measurements/{}/results".format(msmt_id)
			request = AtlasRequest(**{"url_path": path})
			(is_success, results) = request.get()
			if not is_success: 
				print("Failed fetching {}".format(msmt_id))
				continue
			if len(results) == 0:
				# failed measurement of some sort
				self.mc.add_msmt({
					"msm_id": msmt_id,
					"prb_id": -1,
					"type": None,
				})
			else:
				for result in results:
					try:
						result['description'] = msmt_id_to_desc[msmt_id] # save description
						# Save this measurement in our cache, so we don't have to pull it again
						self.mc.add_msmt(result)
					except TypeError:
						print("Failed adding the following result in pull_results: \n{}".format(result))
		self.mc.update_data_structures()
		self.mc.mas_to_meas_id = {}
		for mid in self.mc.data["meas"]:
			for p in self.mc.data["meas"][mid]:
				mas = self.get_probe_metro_as(p)[0]
				if mas[0] is not None:
					try:
						self.mc.mas_to_meas_id[mas].append((mid, p))
					except KeyError:
						self.mc.mas_to_meas_id[mas] = [(mid,p)]
		self.mc.save_cache()

	def pull_specific_measurements(self, msmt_type, prb_id=None, dst=None, recent=7*3600*24):
		"""Pulls all traceroutes from prb_id to dst of a specific measurement type. Returns a string summarizing 
			what happened with the traceroutes."""
		# ignore measurements older than a week by default
		out_str = ""
		parsers = {
			"traceroute": self.parse_ripe_trace_result,
			"ping": None,
		}

		assert not (prb_id is None and dst is None)

		relevant_measurements = []
		# The RIPE Atlas API exposes some basic filters so we don't have to pull tons
		# of measurements
		if dst is None and prb_id is not None:
			path = "/api/v2/measurements/my?key={}&current_probes={}&type={}&page_size=500".format(
				ATLAS_API_KEY,prb_id,msmt_type)
		elif prb_id is None and dst is not None:
			path = "/api/v2/measurements/my?key={}&target_ip={}&type={}&page_size=500".format(
				ATLAS_API_KEY, dst,msmt_type)
		else:
			path = "/api/v2/measurements/my?key={}&target_ip={}&current_probes={}&type={}&page_size=500".format(
				ATLAS_API_KEY, dst, prb_id,msmt_type)
		request = AtlasRequest(**{"url_path": path})
		(is_success, response) = request.get()
		if not is_success: print(response); exit(0)
		while True:
			for el in response["results"]:
				if el["start_time"] < time.time() - recent: continue
				relevant_measurements.append(el['id'])
			if response["next"] is None: break
			path = response["next"][len("https://atlas.ripe.net"):]
			request = AtlasRequest(**{"url_path": path})
			(is_success, response) = request.get()
			if not is_success: print(response); exit(0)
		# pull all the measurements we dont have yet
		all_results = []
		# unfortunately we have to pull them one by one
		# (there's probably some multi-processing solution but I didn't see it as worth it to implement)
		for msmt_id in set(relevant_measurements):
			if np.random.random() > .95: 
				print(f"pulling results {np.random.random()}")
			path = "/api/v2/measurements/{}/results".format(msmt_id)
			request = AtlasRequest(**{"url_path": path})
			(is_success, results) = request.get()
			if not is_success: print(results); exit(0)
			for result in results:
				if prb_id is not None:
					if prb_id != result['prb_id']: continue
				all_results.append((result['prb_id'], parsers[msmt_type](result)))
		self.check_load_siblings()
		# summarizes what we've pulled, which may be interesting to look at, or maybe not
		for prb_id, r in all_results:
			if r is None: continue
			dst = r['ip_paths'][-1][0]
			high_level_features = self.get_high_level_path_features(r)
			#if not high_level_features["is_vf"] and not dst in anycast_addresses: continue
			out_str += "PRB {} measuring to {}\n".format(prb_id, dst)
			for i, ip_hop in enumerate(r['ip_paths']):
				if r['rtts'][i] == []:
					min_rtt = "?"
				else:
					min_rtt = np.min(r['rtts'][i])
				all_ips_this_hop = set(ip_hop)
				ip_hop_str = ""
				for ip_addr in all_ips_this_hop:
					ip_hop_str += ip_addr 
					try:
						peering_interf_loc = high_level_features['peering_interfs'][ip_addr]
						ip_hop_str += " (Peer {})".format(peering_interf_loc)
					except KeyError:
						pass
					try:
						tor_interf_loc  = high_level_features['tor_interfs'][ip_addr]
						ip_hop_str += " (TOR {})".format(tor_interf_loc)
					except KeyError:
						pass
					ip_hop_str += "; "
				out_str += "Hop {} -- {} ({})\n".format(i, ip_hop_str, min_rtt)
			out_str += (",".join([el['actual'] for el in high_level_features['as_path']]) + "\n\n")
		return out_str + "\n\n--------------------------\n\n"

	def spin_up_measurements(self, config_name=None):
		# execute campaigns associated with each config
		self.load_measurement_configs(config_name)
		for msmt_config in self.measurement_configs:
			if msmt_config['campaign_type'] == "large-scale":
				self.execute_large_scale_measurements(msmt_config)
			elif msmt_config['campaign_type'] == "case-study":
				self.execute_case_study_measurements(msmt_config)
			elif msmt_config['campaign_type'] == "exhaustive":
				self.execute_exhaustive_campaign(msmt_config)
			elif msmt_config['campaign_type'] == "case-study-loc":
				self.execute_case_study_from_loc(msmt_config)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="pull_results")
	parser.add_argument("--config", default='exhaustive')
	args = parser.parse_args()
	ap = Atlas_Pinger()
	if args.mode == "measurement_campaign":
		ap.spin_up_measurements(config_name=args.config)
	elif args.mode == "pull_results":
		ap.pull_results()
	elif args.mode == 'execute_oneoff_painter':
		ap.execute_oneoff_painter_assessment()
	elif args.mode == 'execute_longitudinal_painter':
		ap.execute_longitudinal_painter_assessement()
	

	