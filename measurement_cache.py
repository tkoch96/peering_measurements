## Stores RIPE Atlas measurement results in a Python pickle, so we don't have
## to pull them every time we need them
## We store the measurements essentially in their raw form, to enable rapid iteration

import os, pickle, numpy as np, datetime, time, ipaddress, tqdm
from subprocess import call
from peering_measurements.config import *

class Measurement_Cache:
	def __init__(self,long_load=False):
		self.long_load = long_load
		self.msmt_cache_dir = os.path.join(CACHE_DIR,"atlas_measurements")
		self.data = {"meas": {}, "meta": {}}
		for msmt_file in os.listdir(self.msmt_cache_dir):
			if msmt_file.startswith("."): continue
			print("Loading measurement cache (might take a while), long load: {}".format(self.long_load))
			d = pickle.load(open(os.path.join(self.msmt_cache_dir, msmt_file),'rb'))
			print("Done")
			# d has metadata and measurements
			for msmt_id in d["meas"]:
				self.data["meas"][msmt_id] = d["meas"][msmt_id]
			for k in d["meta"]:
				if k == "last_updated":
					# take most recent
					try:
						self.data["meta"][k] = np.maximum(self.data["meta"][k],
							d["meta"][k])
					except KeyError:
						self.data["meta"][k] = d['meta'][k]
				else:
					raise ValueError("Metadata key {} is not supported.".format(k))
		self.update_data_structures()

	def update_data_structures(self):
		# get prb to meas id mapping
		self.prb_to_meas_id = {} # measurement IDs by probe
		self.meas_by_prbid_dst = {} # whether or not measurement of probe,dst exists
		self.min_lat_by_dst = {} # minimum latency to dst from all probes
		self.min_lat_to_probe = {}
		for mid in tqdm.tqdm(self.data["meas"], desc="Organizing measurements from cache."):
			for p in self.data["meas"][mid]:
				try:
					self.prb_to_meas_id[p].append(mid)
				except KeyError:
					self.prb_to_meas_id[p] = [mid]
				if self.long_load:
					for measurement in self.data['meas'][mid][p]:
						try:
							self.meas_by_prbid_dst[p,measurement['dst_name']] = None
						except KeyError:
							# bad measurement
							continue
						try:
							ping_lats = [el['rtt'] for el in measurement['result']]
						except KeyError:
							# bad measurement
							continue
						mpl = np.min(ping_lats)
						try:
							if mpl < self.min_lat_by_dst[measurement['dst_name']]:
								self.min_lat_by_dst[measurement['dst_name']] = mpl
								self.min_lat_to_probe[measurement['dst_name']] = p
						except KeyError:
							self.min_lat_by_dst[measurement['dst_name']] = mpl
							self.min_lat_to_probe[measurement['dst_name']] = p

	def add_msmt(self, msmt):
		try:
			self.data["meas"][msmt['msm_id']]
		except KeyError:
			self.data["meas"][msmt['msm_id']] = {}
		try:
			self.data["meas"][msmt['msm_id']][msmt['prb_id']].append(msmt)
		except KeyError:
			self.data["meas"][msmt['msm_id']][msmt['prb_id']] = [msmt]

	def exists(self, msmt):
		msmt_id = msmt['id']
		try:
			# if this line works, we have at least some data
			# if it's an ongoing campaign, its up to you to make sure you're not adding results
			# to the cache too early
			these_msmts = self.data["meas"][msmt_id] 
			return True
		except KeyError:
			return False

	def get_items(self):
		for k,v in self.data["meas"].items():
			yield k,v

	def save_cache(self):
		td = datetime.datetime.today()
		# indicate we've updated the measurement cache
		self.data["meta"]["last_updated"] = time.time()
		cache_fn = "measurements_{}_{}_{}.pkl".format(td.year,td.month,td.day)
		cache_fn = os.path.join(self.msmt_cache_dir, cache_fn)
		# remove old files to avoid bloat
		call("rm {}".format(os.path.join(self.msmt_cache_dir, "*")), shell=True)
		pickle.dump(self.data, open(cache_fn, "wb"))

	def meas_exists(self, uid, dst, _type='traceroute', match_on='network'):
		"""Checks to see if we've executed a msmt from a probe to a destination."""
		# UID is a probe ID or <metro, AS> (whichever makes sense)
		if match_on == 'network':
			# match on probe network
			if "/" not in dst:
				dst += "/32"
			try:
				self.prb_to_meas_id[uid]
			except KeyError:
				return False
			dst_network = ipaddress.ip_network(dst)
			for mid in self.prb_to_meas_id[uid]:
				for measurement in self.data["meas"][mid][uid]:
					if measurement['type'] == _type and ipaddress.ip_address(measurement["dst_name"]) in dst_network:
						return True
		elif match_on == 'mas':
			# match on probe Metro, AS
			for mid, _pid in self.mas_to_meas_id[uid]:
				for measurement in self.data['meas'][mid][_pid]:
					if measurement['type'] == _type and measurement['dst_name'] == dst:
						return True
		elif match_on == 'probe':
			# match on probe ID
			try:
				self.meas_by_prbid_dst[uid,dst]
				return True
			except KeyError:
				return False
		else:
			raise ValueError("match_on invalid {}".format(match_on))
		
		return False

	def get_meas(self, uid, dst, _type='traceroute', match_on='network'):
		"""Checks to see if we've executed a msmt from a probe to a destination."""
		# UID is a probe ID or <metro, AS> (whichever makes sense)
		ret = []
		if match_on == 'network':
			# match on <probe ID, destination network>
			if "/" not in dst:
				dst += "/32"
			dst_network = ipaddress.ip_network(dst)
			for mid in self.prb_to_meas_id[uid]:
				for measurement in self.data["meas"][mid][uid]:
					if measurement['type'] == _type and ipaddress.ip_address(measurement["dst_name"]) in dst_network:
						ret.append(measurement)
		elif match_on == 'mas':
			# match on probe Metro, AS
			for mid, _pid in self.mas_to_meas_id[uid]:
				for measurement in self.data['meas'][mid][_pid]:
					if measurement['type'] == _type and measurement['dst_name'] == dst:
						ret.append(measurement)
		elif match_on == 'probe':
			# match on probe ID
			for mid in self.prb_to_meas_id[uid]:
				for measurement in self.data['meas'][mid][uid]:
					if measurement['type'] == _type and measurement['dst_name'] == dst:
						ret.append(measurement)
		else:
			raise ValueError("match_on invalid {}".format(match_on))

		return ret

if __name__ == "__main__":
	# Makes a figure showing number of measurmenets over time

	t_s = 1642705000 # start of all painter-specific measurements
	n_secs = time.time() - t_s
	bin_freq = 3 * 24 * 3600 # 3 days
	all_time_meas = np.zeros((int(np.ceil(n_secs / bin_freq))))

	m = Measurement_Cache()
	for meas_id in m.data['meas']:
		for pid in m.data['meas'][meas_id]:
			for meas in m.data['meas'][meas_id][pid]:
				try:
					if meas['timestamp'] < t_s: continue
				except KeyError:
					continue
				bin_i = (meas['timestamp'] - t_s) // bin_freq
				all_time_meas[bin_i] += 1
	import matplotlib.pyplot as plt
	print(all_time_meas)
	plt.plot(np.arange(len(all_time_meas)), all_time_meas)
	plt.xlabel("Time (3 day bins)")
	plt.ylabel("Number of Measurements")
	plt.savefig('figures/number_ripe_atlas_measurements_over_time.pdf')


