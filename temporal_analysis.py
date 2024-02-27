from analyze_measurements import Measurement_Analyzer
from sklearn.cluster import Birch
import pandas as pd, numpy as np, tqdm, os, time,copy
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import scipy.stats as scistat
from helpers import *
from config import *

FIG_DIR = os.path.join(FIG_DIR, 'temporal_analysis')


class Temporal_Analyzer(Measurement_Analyzer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.N_move_average = 4
		self.cluster_temporally = True

	def read_lats_over_time(self):
		ts = -np.inf
		te = np.inf
		self.i_to_popp = {}
		self.i_to_targ = {}

		print("Analyzing from times {} to {}".format(ts,te))
		lat_fn = os.path.join(CACHE_DIR, "vultr_latency_over_time_case_study.csv")
		self.lats_by_targ = {}
		rowi=0
		for row in tqdm.tqdm(open(lat_fn, 'r'),desc="reading latency measurements from VULTR."):
			fields = row.strip().split(',')
			if len(fields) == 3:
				pop,peer,i = fields
				self.i_to_popp[int(i)] = (pop,peer)
				continue
			elif len(fields) == 2:
				targ,i = fields
				self.i_to_targ[int(i)] = targ
				continue
			t_meas,client_dsti,poppi,lat = fields
			if float(t_meas) < ts or float(t_meas) > te: continue 
			try:
				self.lats_by_targ[int(client_dsti)]
			except KeyError:
				self.lats_by_targ[int(client_dsti)] = {}
			try:
				self.lats_by_targ[int(client_dsti)][int(poppi)]
			except KeyError:
				self.lats_by_targ[int(client_dsti)][int(poppi)] = []
			lat = int(np.ceil(float(lat) * 1000))
			self.lats_by_targ[int(client_dsti)][int(poppi)].append((int(t_meas), lat))
			# if rowi == 50000000:
			# 	break
			rowi+=1

		print("Read {} lines".format(rowi))

	def get_probing_targets(self):
		import pytricia
		probe_target_fn = os.path.join(CACHE_DIR , 'interesting_targets_to_probe.csv')
		user_pref_tri = pytricia.PyTricia()
		for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, 'yunfan_prefixes_with_users.csv')), desc="Loading yunfan targets..."):
			if row.strip() == 'prefix': continue
			user_pref_tri[row.strip()] = None
		print("Loaded {} yunfan prefixes".format(len(user_pref_tri)))
		targs_in_user_prefs = {}

		pops_of_interest = ['newyork','toronto','atlanta']
		provider_popps = []
		n_each_pop = 2
		pop_ctr = {pop:0 for pop in pops_of_interest}
		for row in open(os.path.join(CACHE_DIR, '{}_provider_popps.csv'.format(self.system))):
			pop,peer = row.strip().split(',')
			if pop not in pops_of_interest:
				continue
			if pop_ctr[pop] == n_each_pop:
				continue
			pop_ctr[pop] += 1
			provider_popps.append((pop,peer))
		print(provider_popps)

		# ignore the clouds
		ignore_ases = [self.parse_asn(asn) for asn in [699,8075,15169,792,37963,36351]]

		anycast_latencies = parse_anycast_latency_file(os.path.join(CACHE_DIR, '{}_anycast_latency_withoutrpki.csv'.format(self.system)))
		print("Loaded {} anycast latencies".format(len(anycast_latencies)))
		n_valid_latency = 0
		all_asns = {}
		# self.lookup_asns_if_needed(list(set([ip32_to_24(ip) for ip,pop in anycast_latencies])))
		for (ip,pop), lats in anycast_latencies.items():
			if pop not in pops_of_interest: continue
			parent_pref = user_pref_tri.get_key(ip)
			lat = np.min(lats)
			if lat < 5 or lat > 80: continue
			n_valid_latency += 1

			ip_asn = self.parse_asn(ip)
			all_asns[ip_asn] = None

			if parent_pref is not None:
				parent_key = self.parse_asn(parent_pref)
				if parent_key in ignore_ases: continue
				try:
					targs_in_user_prefs[parent_key].append((ip,float(lat)*1000,pop))
				except KeyError:
					targs_in_user_prefs[parent_key] = [(ip,float(lat)*1000,pop)]
		print("{} targets with valid latency, {} ASNs".format(n_valid_latency, len(all_asns)))
		print("There are {} ASNs with pingable targets matching our criteria".format(len(targs_in_user_prefs)))
		used_prefs = {}
		with open(probe_target_fn,'w') as f:
			popps = list(provider_popps)
			popps = [str(popp[0]) + "|" + str(popp[1]) for popp in popps]
			popps_str = "-".join(popps)
			for parent_asn, targslatspop in targs_in_user_prefs.items():
				sorted_targs = sorted(targslatspop, key = lambda el : np.abs(el[1]-20))
				for t,l,pop in sorted_targs[0:20]:
					parent_pref = user_pref_tri.get_key(t)
					try:
						used_prefs[parent_pref]
						continue
					except KeyError:
						used_prefs[parent_pref] = None
					# org = self.org_to_as.get(parent_asn, [parent_asn])[0]
					# f.write("{},{},{}\n".format(org,t,l))
					f.write("{},{},{}\n".format(t,popps_str,pop))

		# interesting_prefs = list(targs_in_user_prefs)
		# self.lookup_asns_if_needed(interesting_prefs)
		# asn_ct = {}
		# for pref in interesting_prefs:
		# 	asn = self.parse_asn(pref)
		# 	if asn is not None:
		# 		try:
		# 			asn_ct[asn] += 1
		# 		except KeyError:
		# 			asn_ct[asn] = 1
		# for asn,ct in sorted(asn_ct.items(), key = lambda el : -1 * el[1]):
		# 	print("{} {} {}".format(asn,self.org_to_as.get(asn,[asn]),ct))

	def characterize_time_series(self,lat_bp,var_bp,loss_bp,verb=False):
		typical_lats_by_popp = {popp: np.median(lats) for popp,lats in lat_bp.items()}
		ranked_lbp = sorted(typical_lats_by_popp.items(), key = lambda el : el[1])
		best_popp, second_best_popp = ranked_lbp[0][0],ranked_lbp[1][0]

		# check for path changes
		# path changes skew variance estimates
		# means = cluster
		# if more than one (n_pts at mean > 10% of n_total_pts): path change
		has_path_change = False
		for popp in [best_popp, second_best_popp]:
			if var_bp[popp] > 1:
				brc = Birch(threshold=4,n_clusters=None)
				labels = brc.fit_predict(lat_bp[popp].reshape(-1,1))
				u,c = np.unique(labels,return_counts=True)
				total_n = len(lat_bp[popp])
				frac = c / total_n
				if sum(frac > .1 ) > 1:
					has_path_change = True
					break

		total_loss = sum([sum(v) for v in loss_bp.values()])
		has_loss =  (total_loss > 10)


		## check for persistent congestion
		congested_popps = {}
		for popp,lats in lat_bp.items():
			pdlat = pd.DataFrame(lats)
			avged_lat = np.array(pdlat.rolling(self.N_move_average,min_periods=1).median()).flatten()
			mlat = np.percentile(lats,1) # propagation delay
			lat_mask = (avged_lat > (mlat + 5)).astype(np.int32)
			## fill in holes via voting
			hole_filled_lat_mask = lat_mask.copy()
			for i in range(len(lats)):
				if not hole_filled_lat_mask[i]:
					hole_filled_lat_mask[i] = int(np.round(np.average(lat_mask[np.maximum(0,i-self.N_move_average):np.maximum(i+1,len(lats))])))
			## look for blocks of "ones" greater than a certain length
			congested_blocks, n_congest, congested, was_ever_uncongested = [], 0, False, False
			# if verb:
			# 	print("{} {} {}".format(avged_lat, lat_mask, hole_filled_lat_mask))
			for i, l in enumerate(hole_filled_lat_mask):
				if l and was_ever_uncongested:
					n_congest += 1
					if n_congest == self.N_move_average:
						congested = True
						# start of congestion is N_move_average places ago
						congest_start = np.maximum(i - self.N_move_average,0)
				elif not l:
					was_ever_uncongested = True
					if congested:
						# end of congestion
						congest_end = i
						### Make sure there's some loss
						if sum(loss_bp[popp][congest_start:congest_end]) > 0:
							congested_blocks.append((congest_start, congest_end))
						congested = False
					n_congest = 0
			congested_popps[popp] = congested_blocks
		any_congestion = any(len(blocks) > 0 for popp, blocks in congested_popps.items())
		return {
			'best_popp': best_popp,
			'has_loss' : has_loss,
			'second_best_popp': second_best_popp,
			'has_path_change': has_path_change,
			'has_congestion': any_congestion,
			'congested_popps': congested_popps,
		}

	def check_parse_load_latency_data(self):
		try:
			self.lats_by_targ
			return
		except AttributeError:
			pass
		parsed_latencies_cache_fn = os.path.join(CACHE_DIR, 'jitter_metrics.pkl')
		### Parse the original measurements
		if not os.path.exists(parsed_latencies_cache_fn):
			print("Parsing raw latency measurements...")
			self.read_lats_over_time()
			brc = None
			avg_over = 5 # seconds, clustering time
			self.loss_by_targ = {}
			self.ranks_by_targ = {}
			all_popps = sorted(list(self.i_to_popp))
			rand_noise_popp = {poppi:.0001*np.random.random() for poppi in all_popps} ## arbitrarily breaks ties
			rand_noise_popp_np = np.array([rand_noise_popp[poppi] for poppi in all_popps])
			for targ in tqdm.tqdm(self.lats_by_targ,desc="Post-processing latency data"):
				self.loss_by_targ[targ] = {}
				lats_np = None
				for popp in self.lats_by_targ[targ]:
					## Since the probing isn't exactly periodic, we find the measurement points by clustering measurements in time
					if self.cluster_temporally:
						if brc is None:
							ts = np.array([el[0] for el in self.lats_by_targ[targ][popp]])
							brc = Birch(threshold=avg_over,n_clusters=None)
							labels = brc.fit_predict(ts.reshape(-1,1))
						new_lats = {}
						for t,lat in self.lats_by_targ[targ][popp]:
							lat += rand_noise_popp[popp]
							t_map = brc.subcluster_centers_[np.argmin(np.abs(brc.subcluster_centers_ - t))][0]
							try:
								new_lats[t_map].append(lat)
							except KeyError:
								new_lats[t_map] = [lat]
						new_lats_avg = {t_map: np.mean(new_lats[t_map]) for t_map in new_lats}
						self.lats_by_targ[targ][popp] = sorted(new_lats_avg.items(), key = lambda el : el[0])
						self.loss_by_targ[targ][popp] = [np.maximum(4 - len(new_lats[t_map]),0) for t_map,_ in self.lats_by_targ[targ][popp]]
					else:
						self.loss_by_targ[targ][popp] = np.zeros((len(self.lats_by_targ[targ]), ))


					## reject outliers (temporary spikes, likely due to measurement error)
					lats = np.array(self.lats_by_targ[targ][popp])[:,1]
					if lats_np is None:
						lats_np = np.zeros((len(self.i_to_popp), len(lats))) ### Default is large since we only use this for ranking paths
					else:
						if lats_np.shape[1] < len(lats):
							lats_np = np.concatenate([lats_np,np.zeros((len(self.i_to_popp), len(lats) - lats_np.shape[1]))], axis=1)
					
					mlat = np.median(lats)
					devlat = np.sqrt(np.var(lats))
					rmvs = (lats > (mlat + 11 * devlat)) # outliers
					notboring = False
					needs_plugging = np.where(rmvs)[0]
					for rmv in needs_plugging:
						self.lats_by_targ[targ][popp][rmv] = list(self.lats_by_targ[targ][popp][rmv])
						if rmv == 0:
							self.lats_by_targ[targ][popp][rmv][1] = lats[rmv+1]
						elif rmv == len(lats) - 1:
							self.lats_by_targ[targ][popp][rmv][1] = lats[rmv-1]
						else:
							self.lats_by_targ[targ][popp][rmv][1] = np.mean([lats[rmv-1],lats[rmv+1]])
						self.lats_by_targ[targ][popp][rmv] = tuple(self.lats_by_targ[targ][popp][rmv])

					## Smooth with EWMA
					alpha = .5
					smoothed_data = [self.lats_by_targ[targ][popp][0]]
					for i in range(1, len(self.lats_by_targ[targ][popp])):
						new_lat = alpha * smoothed_data[i-1][1] + (1 - alpha) * self.lats_by_targ[targ][popp][i][1]
						smoothed_data.append((self.lats_by_targ[targ][popp][i][0], new_lat))
					self.lats_by_targ[targ][popp] = smoothed_data

					lats_np[popp,0:len(self.lats_by_targ[targ][popp])] = np.array([el[1] for el in self.lats_by_targ[targ][popp]])
				if lats_np.shape[1] > 1:
					for poppi in range(lats_np.shape[0]): ## Fill in holes caused by not measuring everything every time
						if lats_np[poppi,-1] == 0:
							lats_np[poppi,-1] = lats_np[poppi,-2]

				## Rank each option over time
				#### Why is last column fked, also fix zeros
				self.ranks_by_targ[targ] = scistat.rankdata(lats_np,axis=0)-1

			for targ in list(self.lats_by_targ):
				if len(self.lats_by_targ[targ]) == 1:
					del self.lats_by_targ[targ]
					del self.loss_by_targ[targ]
					del self.ranks_by_targ[targ]
				

			pickle.dump({	
				'lats_by_targ':self.lats_by_targ, 
				'loss_by_targ': self.loss_by_targ, 
				'ranks_by_targ': self.ranks_by_targ,
				'i_to_popp': self.i_to_popp, 
				'i_to_targ': self.i_to_targ
			}, open(parsed_latencies_cache_fn,'wb'))
		else:
			print("Loading parsed raw measurements")
			d = pickle.load(open(parsed_latencies_cache_fn,'rb'))
			self.lats_by_targ, self.loss_by_targ, self.ranks_by_targ = d['lats_by_targ'], d['loss_by_targ'], d['ranks_by_targ']
			self.i_to_popp, self.i_to_targ = d['i_to_popp'], d['i_to_targ']


	def check_parse_load_highlevel_data(self):
		try:
			self.classified_congestions
			return
		except AttributeError:
			pass
		parsed_high_level_metrics_cache_fn = os.path.join(CACHE_DIR, 'high_level_temporal_dynamics.pkl')
		if not os.path.exists(parsed_high_level_metrics_cache_fn):
			self.check_parse_load_latency_data()
			self.classified_congestions = {}
			for targ in tqdm.tqdm(self.lats_by_targ,desc="Identifying interesting targets to plot."):
				# targ = targ_to_i['198.166.29.134']
				lats_by_popp = {popp: np.array([el[1] for el in self.lats_by_targ[targ][popp]]) for popp in self.lats_by_targ[targ]}
				loss_by_popp = self.loss_by_targ[targ]
				var_by_popp = {popp: np.var(lats_by_popp[popp]) for popp in lats_by_popp}
				self.classified_congestions[targ] = self.characterize_time_series(lats_by_popp, var_by_popp, loss_by_popp)
			pickle.dump({
				'classified_congestions': self.classified_congestions,
				'i_to_popp': self.i_to_popp, 
				'i_to_targ': self.i_to_targ,
			}, open(parsed_high_level_metrics_cache_fn,'wb'))
		else:
			print("Loading parsed high level measurements")
			d = pickle.load(open(parsed_high_level_metrics_cache_fn,'rb'))
			self.i_to_popp,self.i_to_targ,self.classified_congestions = d['i_to_popp'], d['i_to_targ'], d['classified_congestions']

	def general_analysis(self):
		### high level question : how often do we see cases where a path decision function would be non-trivial?

		## intuitively, if the "best path" is unstable, and the second-best path is close to it,
		## it could be interesting
		## most interesting would be cases where the paths have predictable "noises", 
		## instead of something like random shot noise


		## are there enough cases where I could make an interesting decision?
		## straw-man : assume 10s flows, place each flow on lowest latency path at each point in time,
		## average/max latency of flow - best possible over time and over dests
		## for the cases where I could make an interesting decision, is it feasible to estimate that noise?

		def get_intersection(intv1,intv2):
			## Intersection of two real intervals
			if intv1[0] > intv2[0]:
				tmp = copy.copy(intv2)
				intv2 = copy.copy(intv1)
				intv1 = tmp

			if intv1[1] >= intv2[0] and intv1[0] <= intv2[1]:
				## nonempty intersection
				return (intv2[0], intv1[1])
			return None

		def calc_iou(intv1, intv2):
			## Intersection over the union of two real intervals
			# Order them
			if intv1[0] > intv2[0]:
				tmp = copy.copy(intv2)
				intv2 = copy.copy(intv1)
				intv1 = tmp

			if intv1[1] >= intv2[0] and intv1[0] <= intv2[1]:
				## nonempty intersection
				i = intv1[1] - intv2[0]
				if intv2[1] >= intv1[1]:
					u = intv2[1] - intv1[0]
				else:
					u = intv1[1] - intv1[0]
				return i / u
			else: 
				return 0



		self.check_parse_load_highlevel_data()
		targs = list(self.classified_congestions)
		all_match_info = []
		for targ in targs:
			if not self.classified_congestions[targ]['has_congestion']: continue
			info = self.classified_congestions[targ]
			## Find max IOU for each congestive event
			for poppi,eventsi in info['congested_popps'].items():
				for eventi in eventsi:
					matches = {}
					for poppj,eventsj in info['congested_popps'].items():
						if poppi == poppj: continue
						matches[poppj] = {"max_iou": 0, "max_event": None}
						for eventj in eventsj:
							iou = calc_iou(eventi,eventj)
							if iou > matches[poppj]['max_iou']:
								matches[poppj]['max_iou'] = iou
								matches[poppj]['max_event'] = eventj
								if iou == 1:break
					all_match_info.append((targ,poppi,eventi,matches))

		for IOU_THRESHOLD in [.3,.5,1]:
			stats = {'n_popps_common_to': [], 'n_pops_common_to': []}
			for targ,poppi,eventi,matches in all_match_info:
				### how many paths was this event common to
				stats['n_popps_common_to'].append(sum(1 for m in matches if matches[m]['max_iou'] >= IOU_THRESHOLD))
				stats['n_pops_common_to'].append(len(set(self.i_to_popp[m][0] for m in matches if matches[m]['max_iou'] >= IOU_THRESHOLD)))
			x,cdf_x = get_cdf_xy(stats['n_popps_common_to'])
			plt.plot(x,cdf_x,label="PoPPs THRESH: {}".format(IOU_THRESHOLD))
			x,cdf_x = get_cdf_xy(stats['n_pops_common_to'])
			plt.plot(x,cdf_x,label="PoPs THRESH: {}".format(IOU_THRESHOLD), marker='x')
		plt.xlabel("Number of Things Common To")
		plt.ylabel("CDF of Congestive Events")
		plt.grid(True)
		plt.ylim([0,1])
		plt.legend()
		self.save_fig('n_paths_common_to.pdf')


		IOU_THRESHOLD = .5
		cases = {0:[],1:[]}
		interesting_targs = []
		self.check_parse_load_latency_data()
		for targ,poppi,eventi,matches in all_match_info:
			just_before_event_t = eventi[0] - 1
			poppi_rank = self.ranks_by_targ[targ][poppi,just_before_event_t]
			if poppi_rank != 0: continue # only focus on cases where poppi was the best before congestion
			measi = np.array([el[1] for el in self.lats_by_targ[targ][poppi]])
			for poppj in matches:
				if matches[poppj]['max_iou'] < IOU_THRESHOLD: continue
				i = get_intersection(eventi,matches[poppj]['max_event'])
				midpoint_event = i[0] + (i[1] - i[0])//2
				if self.ranks_by_targ[targ][poppj,midpoint_event] not in [0,1]: continue
				measj = np.array([el[1] for el in self.lats_by_targ[targ][poppj]])

				## We have an interesting match
				delta_before = np.mean(measj[just_before_event_t-1:just_before_event_t+1] - measi[just_before_event_t-1:just_before_event_t+1])
				delta_after = np.mean(measj[midpoint_event-1:midpoint_event+1] - measi[midpoint_event-1:midpoint_event+1])

				# print("{} {}\n{}\n{}\n{}\n{}\n{} {}".format(just_before_event_t,midpoint_event,
				# 	self.ranks_by_targ[targ][poppi,:],self.ranks_by_targ[targ][poppj,:],
				# 	measi.round(),measj.round(),delta_before,delta_after))
				cases[0].append(delta_after - delta_before)
				cases[1].append(delta_after)
				if delta_after < -5:
					print(matches[poppj])
					print(eventi)
					print("{} tbf:{} tmid:{} i: {} j: {} Before {}ms {}ms After {}ms {}ms".format(self.i_to_targ[targ],just_before_event_t,midpoint_event,
						self.i_to_popp[poppi], self.i_to_popp[poppj], measi[just_before_event_t], measj[just_before_event_t],
						measi[midpoint_event], measj[midpoint_event]))
					interesting_targs.append((targ, midpoint_event * 60))

		plt.scatter(cases[0],cases[1])
		plt.xlabel("Difference in Deltas")
		plt.ylabel("Delta After")
		plt.grid(True)
		self.save_fig('delta_latency_top_2.pdf')


		popps = list(set(popp for targ in self.lats_by_targ for popp in self.lats_by_targ[targ]))
		print("{} time steps".format(len(self.lats_by_targ[targs[0]][popps[0]])))
		cols = ['red','black','royalblue','magenta','darkorange','forestgreen','tan']
		popp_to_col = {popp:c for popp,c in zip(popps,cols)}
		popp_to_ind = {popp:i for i,popp in enumerate(popps)}


		plt.rcParams["figure.figsize"] = (40,90)
		nrows,ncols = 25,8
		plt_every = 1
		f,ax = plt.subplots(nrows,ncols)
		for targi, (targ, t_event) in enumerate(sorted(interesting_targs)[0:200]):
			axrow = targi // ncols
			axcol = targi % ncols
			ax2 = ax[axrow,axcol].twinx()
			for popp in sorted(self.lats_by_targ[targ]):
				measx = np.array([el[0] for el in self.lats_by_targ[targ][popp]])
				measx = measx - measx[0]
				measy = np.array([el[1] for el in self.lats_by_targ[targ][popp]])
				lab = self.i_to_popp[popp]
				ax[axrow,axcol].plot(measx[::plt_every],measy[::plt_every],label=lab,color=popp_to_col[popp])

				lossy = self.loss_by_targ[targ][popp]
				ax2.scatter(measx,lossy,color=popp_to_col[popp])
			ax2.set_ylim([0,4])
			# ax[axrow,axcol].set_ylim([10,300])
			ax[axrow,axcol].set_title(self.i_to_targ[targ] + " ({})".format(t_event))
			ax[axrow,axcol].legend(fontsize=5)
		self.save_fig('latency_over_time_interesting_targs.pdf')
		exit(0)

		# assessed_delta_perfs = {}

		# all_delta_perfs = [el for targ in assessed_delta_perfs for el in assessed_delta_perfs[targ]]
		# max_by_targ = [np.max(perfs) for targ,perfs in assessed_delta_perfs.items()]
		# med_by_targ = [np.median(perfs) for targ,perfs in assessed_delta_perfs.items()]

		# ## q: fraction of time we are within X ms of the best?
		# fracs_within_best = {nms: [sum(1 for p in perfs if p > nms)/len(perfs) for targ,perfs in assessed_delta_perfs.items()] \
		# 	for nms in [1,3,5,10,15,50,100]}

		# inter_decision_t = 75 # seconds
		# for targ in tqdm.tqdm(lats_by_targ,desc="Assessing strawman"):
		# 	boring = True
		# 	best_overall = None
		# 	range_by_popp = {}
		# 	for popp in lats_by_targ[targ]:
		# 		lat_vals = list([lat for t,lat in lats_by_targ[targ][popp]])
		# 		range_by_popp[popp] = {
		# 			'min': np.min(lat_vals),
		# 			'max': np.max(lat_vals)
		# 		}
		# 		if best_overall is None:
		# 			best_overall = popp
		# 		elif range_by_popp[best_overall]['min'] > range_by_popp[popp]['min']:
		# 			best_overall = popp
		# 	for poppi in lats_by_targ[targ]:
		# 		if poppi == best_overall: continue
		# 		if range_by_popp[best_overall]['max'] > range_by_popp[poppi]['min'] + 5:
		# 			boring = False
		# 			break

		# 	all_meas = []
		# 	for popp,vals in lats_by_targ[targ].items():
		# 		for v in vals:
		# 			all_meas.append((popp,v))
		# 	sorted_meas = [(popp_to_ind[popp],t,lat) for popp, (t,lat) in sorted(all_meas, 
		# 		key = lambda el : el[1][0])]
		# 	t_start = sorted_meas[0][1]
		# 	curr_popp_lats = np.zeros((6))
		# 	t_last_decision = None
		# 	curr_popp_decision = None
		# 	delta_perfs = []
		# 	for popp,t,lat in sorted_meas:
		# 		curr_popp_lats[popp] = lat
		# 		if t - t_start > 60:
		# 			## start making decisions
		# 			best_popp = np.argmin(curr_popp_lats)
		# 			if t_last_decision is None:
		# 				curr_popp_decision = best_popp
		# 				t_last_decision = t
		# 			elif t - t_last_decision > inter_decision_t:
		# 				curr_popp_decision = best_popp
		# 				t_last_decision = t

		# 			delta_perf = curr_popp_lats[curr_popp_decision] - curr_popp_lats[best_popp]
		# 			delta_perfs.append((t,delta_perf))

		# 			# if not boring:
		# 			# 	print("{} {} {} {}".format(best_popp, curr_popp_lats[best_popp], curr_popp_decision,
		# 			# 		curr_popp_lats[curr_popp_decision]))

		# 	delta_perfs = np.array(delta_perfs)
		# 	ts,te = delta_perfs[0][0], delta_perfs[-1][0]
		# 	ns_period = 1.3 # how often we assess performance
		# 	n_periods = int(np.ceil((te-ts)/ns_period))
		# 	assessed_delta_perfs[targ] = []
		# 	for i in range(n_periods):
		# 		tnow = ts + ns_period * i
		# 		instancei = np.where(delta_perfs[:,0] - tnow >= 0 )[0][0]
		# 		delta_perf_now = delta_perfs[instancei,1]
		# 		assessed_delta_perfs[targ].append(delta_perf_now)
				
			

			# if not boring:
			# 	print("Best overall is {}".format(best_overall))
			# 	print(np.sum(delta_perfs[:,1]))
			# 	f,ax = plt.subplots(1)
			# 	for popp in lats_by_targ[targ]:
			# 		measx = np.array([el[0] for el in lats_by_targ[targ][popp]])
			# 		measx = measx - measx[0]
			# 		measy = np.array([el[1] for el in lats_by_targ[targ][popp]])
			# 		ax.plot(measx,measy,label=popp)
			# 	ax.legend()
			# 	plt.savefig('figures/tmptarglatovertime.pdf')
			# 	plt.clf(); plt.close()
			# 	exit(0)


		plt.rcParams["figure.figsize"] = (5,10)
		f,ax = plt.subplots(2)
		for arr,lab in zip([all_delta_perfs, max_by_targ,med_by_targ], ['all','max','median']):
			x,cdf_x = get_cdf_xy(arr)
			ax[0].plot(x,cdf_x,label=lab)
		for nms in fracs_within_best:
			x,cdf_x = get_cdf_xy(list(fracs_within_best[nms]))
			ax[1].plot(x,cdf_x,label="{} ms".format(nms))
		ax[0].legend()
		ax[1].legend()
		ax[0].grid(True)
		ax[1].grid(True)
		ax[0].set_xlabel('Achieved - Optimal (ms)')
		ax[0].set_ylabel('CDF of Targets/Time-Targets')
		ax[0].set_xlim([0,50])
		ax[1].set_xlabel('Fraction of Time Performance Requirement Not Met')
		ax[1].set_ylabel('CDF of Targets')
		ax[1].set_xlim([0,1])
		ax[1].set_ylim([0,1])
		self.save_fig('deltaperfs-strawman.pdf')
		plt.clf(); plt.close()



		# 1. for best and second best mean/median latency path, plot mu_1/mu_2 and sigma_1 / sigma_2
		# 2. for mu_1 - mu_2 vs mu_1 - best_overall_measured (when one path degrades, does the 2nd best degrade)
		metrics = {
			'compare_two': {'x':[], 'y': []}, 
			'degrade_corr': {'x': [], 'y': []}
		}

		metrics_by_targ = {}
		for targ in lats_by_targ:
			if len(lats_by_targ[targ]) < 2 : continue
			lats_by_popp = {popp: np.array([el[1] for el in lats_by_targ[targ][popp]]) for popp in lats_by_targ[targ]}
			var_by_popp = {popp: np.var(lats_by_popp[popp]) for popp in lats_by_popp}
			if np.min(list(var_by_popp.values())) > 10:
				continue
			
			ret = check_path_change(lats_by_popp, var_by_popp)
			best_popp, second_best_popp, has_path_change = ret['best_popp'], ret['second_best_popp'], ret['has_path_change']

			if has_path_change: continue

			mean_1, mean_2 = np.mean(lats_by_popp[best_popp]), np.mean(lats_by_popp[second_best_popp])
			var_1, var_2 = var_by_popp[best_popp], var_by_popp[second_best_popp]
			best_overall = np.min(list(el for popp,els in lats_by_popp.items() for el in els))

			metrics['compare_two']['x'].append(mean_1 - mean_2)
			metrics['compare_two']['y'].append(var_1 - var_2)
			metrics['degrade_corr']['x'].append(mean_1 - mean_2)
			metrics['degrade_corr']['y'].append(mean_1 - best_overall)

			metrics_by_targ[targ] = {'compare_two': (mean_1 - mean_2, var_1 - var_2),
			'degrade_corr': (mean_1 - mean_2, mean_1 - best_overall)}

		f,ax = plt.subplots(2)
		for i, (mtrc, lab,xl,yl) in enumerate(zip(['compare_two', 'degrade_corr'], 
			['Compare', 'Degradation'], ['mu1 - mu2','mu1 - mu2'], ['var1 - var2','mu1 - prop'])):
			ax[i].scatter(metrics[mtrc]['x'], metrics[mtrc]['y'])
			ax[i].set_title(lab)
			ax[i].set_xlabel(xl)
			ax[i].set_ylabel(yl)
		ax[1].set_ylim([-10,100])
		ax[0].set_xlim([-50,50])
		ax[0].set_ylim([-10,10])
		ax[1].set_xlim([-50,50])
		self.save_fig('jitter_metrics_investigation.pdf')


		# sorted_targs = sorted(metrics_by_targ.items(), 
		# 	key = lambda el : -1 * np.abs(el[1]['compare_two'][0] * el[1]['compare_two'][1]))
		# plt.rcParams["figure.figsize"] = (20,60)
		# nrows,ncols = 25,4
		# f,ax = plt.subplots(nrows,ncols)
		# for targi, (targ,ml) in enumerate(sorted_targs[0:100]):
		# 	axrow = targi // ncols
		# 	axcol = targi % ncols
		# 	for popp in lats_by_targ[targ]:
		# 		measx = np.array([el[0] for el in lats_by_targ[targ][popp]])
		# 		measx = measx - measx[0]
		# 		measy = np.array([el[1] for el in lats_by_targ[targ][popp]])
		# 		ax[axrow,axcol].plot(measx,measy)
		# plt.savefig('figures/latency_over_time_random_targs.pdf')
		# plt.clf(); plt.close()

	def save_fig(self, fig_file_name):
		# helper function to save to specific figure directory
		# plt.grid(True)
		plt.savefig(os.path.join(FIG_DIR, fig_file_name),bbox_inches='tight')
		plt.clf()
		plt.close()


if __name__ == "__main__":
	np.random.seed(31415)
	ta = Temporal_Analyzer()
	ta.general_analysis()