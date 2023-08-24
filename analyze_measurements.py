import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os, glob, pickle
import matplotlib.pyplot as plt, numpy as np,tqdm

from config import *
from helpers import *
from generic_measurement_utils import AS_Utils_Wrapper


def read_lats_over_time():
	ts = -np.inf
	te = np.inf
	i_to_popp = {}
	i_to_targ = {}

	print("Analyzing from times {} to {}".format(ts,te))
	lat_fn = os.path.join(CACHE_DIR, "vultr_latency_over_time_case_study.csv")
	lats_by_targ = {}
	rowi=0
	for row in tqdm.tqdm(open(lat_fn, 'r'),desc="reading latency measurements from VULTR."):
		fields = row.strip().split(',')
		if len(fields) == 3:
			pop,peer,i = fields
			i_to_popp[int(i)] = (pop,peer)
			continue
		elif len(fields) == 2:
			targ,i = fields
			i_to_targ[int(i)] = targ
			continue
		t_meas,client_dsti,poppi,lat = fields
		if float(t_meas) < ts or float(t_meas) > te: continue 
		try:
			lats_by_targ[int(client_dsti)]
		except KeyError:
			lats_by_targ[int(client_dsti)] = {}
		try:
			lats_by_targ[int(client_dsti)][int(poppi)]
		except KeyError:
			lats_by_targ[int(client_dsti)][int(poppi)] = []
		lat = int(np.ceil(float(lat) * 1000))
		lats_by_targ[int(client_dsti)][int(poppi)].append((int(t_meas), lat))
		if rowi == 10000000:
			break
		rowi+=1

	print("Read {} lines".format(rowi))
	return {
		'lats_by_targ': lats_by_targ,
		'i_to_popp': i_to_popp,
		'i_to_targ': i_to_targ,
	}


class Measurement_Analyzer(AS_Utils_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.update_cc_cache()
		self.systems = ['vultr','peering']
		self.system = kwargs.get('system','vultr')

		self.provider_popps = {system:{} for system in self.systems}
		for system in self.systems:
			for row in open(os.path.join(CACHE_DIR, '{}_provider_popps.csv'.format(system)),'r'):
				if system == 'vultr':
					pop,peer = row.strip().split(',')
					peer = self.parse_asn(peer)
					popp = pop,peer
					self.provider_popps[system][popp] = None
				elif system == 'peering':
					pop, providers = row.strip().split('\t')
					providers = providers.split("|")
					for peer in providers:
						peer = self.parse_asn(peer)
						popp = pop,peer
						self.provider_popps[system][popp] = None

	def why_missing(self):
		unreachable_clients_by_popp = {}
		unreachable_clients_by_pop = {}
		for fn in tqdm.tqdm(glob.glob(os.path.join(CACHE_DIR, 'unreachable_dsts', '*')),
			desc='reading unreachable dsts'):
			pop,peer = fn.split("/")[-1].split("-")
			peer = peer[0:-4]
			try:
				unreachable_clients_by_popp[pop,peer]
			except KeyError:
				unreachable_clients_by_popp[pop,peer] = 0
			for row in open(fn,'r'):
				unreachable_clients_by_popp[pop,peer] += 1
				try:
					unreachable_clients_by_pop[pop]
				except KeyError:
					unreachable_clients_by_pop[pop] = {}
				try:
					unreachable_clients_by_pop[pop][peer] += 1
				except KeyError:
					unreachable_clients_by_pop[pop][peer] = 1


		n_by_popp = [v for v in unreachable_clients_by_popp.values()]
		n_by_pop = {pop:[v for v in unreachable_clients_by_pop[pop].values()] for pop in unreachable_clients_by_pop}
		x,cdf_x = get_cdf_xy(n_by_popp,logx=True)
		plt.semilogx(x,cdf_x,label="All PoPs",**get_col_marker(0))
		i=1
		for pop in n_by_pop:
			x,cdf_x = get_cdf_xy(n_by_pop[pop], logx=True)
			plt.semilogx(x,cdf_x,label=pop,**get_col_marker(i))
			i+=1
		plt.legend(fontsize=8)
		plt.xlabel('Number of unreachable targets')
		plt.ylabel("CDF of PoPPs")
		plt.savefig('figures/n_unreachable_targets_by_popp.pdf')

	def customer_cone_comparison(self):
		### Compare number of ASes in ingress peer/provider customer cones for available systems
		ccs = {s:{} for s in self.systems}
		for system in self.systems: # Need to pre-update the CC cache with all peers, in case some are missing
			for row in open(os.path.join(DATA_DIR, '{}_peers.csv'.format(system)),'r'):
				if system == 'vultr':
					pop,peer,nh = row.strip().split(',')
					if peer == 'peer': continue
					peer = self.parse_asn(peer)
					self.get_cc(peer)
				elif system == 'peering':
					pop,peers = row.strip().split('\t')
					peers = peers.split('|')
					for peer in peers:
						peer = self.parse_asn(peer)
						self.get_cc(peer)
		every_asn = list(set(self.ip_to_asn.values()).union(self.cc_cache))
		for system in self.systems:
			for row in open(os.path.join(DATA_DIR, '{}_peers.csv'.format(system)),'r'):
				if system == 'vultr':
					pop,peer,nh = row.strip().split(',')
					if peer == 'peer': continue
					peer = self.parse_asn(peer)
					try:
						self.provider_popps[system][pop,peer]
						ccs[system][pop,peer] = every_asn
					except KeyError:
						ccs[system][pop,peer] = self.get_cc(peer)
				elif system == 'peering':
					pop,peers = row.strip().split('\t')
					peers = peers.split('|')
					for peer in peers:
						peer = self.parse_asn(peer)
						try:
							self.provider_popps[system][pop,peer]
							ccs[system][pop,peer] = every_asn
						except KeyError:
							ccs[system][pop,peer] = self.get_cc(peer)

			all_n_ccs = list([len(ccs[system][k]) for k in ccs[system]])
			x,cdf_x = get_cdf_xy(all_n_ccs,logx=True)
			plt.semilogx(x,cdf_x,label=system)
		plt.xlabel("Number of ASes in Customer Cone")
		plt.ylabel("CDF of Ingresses")
		plt.grid(True)
		plt.legend()
		self.save_fig('system_cc_size_comparison.pdf')

		plt.rcParams.update({'font.size': 16})
		f,ax = plt.subplots(1,1)
		f.set_size_inches(12,7)
		every_asn = {asn:None for asn in every_asn}
		for system,lab in zip(['peering','vultr'],['PEERING','VULTR']):
			n_ingresses_by_asn = {}
			for popp,cc in ccs[system].items():
				for asn in cc:
					try:
						every_asn[asn]
					except KeyError:
						print("Nothing found for {}".format(asn))
						continue
					try:
						n_ingresses_by_asn[asn] += 1
					except KeyError:
						n_ingresses_by_asn[asn] = 1
			for asn, n in n_ingresses_by_asn.items():
				if n > 100 and system == 'peering':
					print(self.org_to_as.get(asn,asn))
			x,cdf_x = get_cdf_xy(list(n_ingresses_by_asn.values()))
			ax.plot(x,cdf_x,label=lab)
		ax.set_xlabel("Number of ingresses with paths from AS",fontsize=16)
		ax.set_ylabel("CDF of ASes",fontsize=16)
		ax.legend(fontsize=16)
		ax.set_ylim([0,1.0])
		ax.grid(True)
		self.save_fig("number_reachable_ingresses_by_as_comparison.pdf")

	def summarize_need_meas(self, peers):
		cust_cone = pickle.load(open(os.path.join(CACHE_DIR, 
			'{}_cust_cone.pkl'.format(self.system)),'rb'))
		with open(os.path.join(CACHE_DIR, 'possible_pref_targets_by_peer.csv'),'w') as f:
			for peer in peers:
				all_asns = cust_cone.get(peer,[])
				all_prefs = {}
				for asn in all_asns:
					prefs = self.routeviews_asn_to_pref.get(asn)
					if prefs is not None:
						for pref in prefs:
							all_prefs[pref] = None
				all_prefs = list(all_prefs)
				if len(all_prefs) > 0:
					prefs_str = "-".join(all_prefs)	
				else:
					prefs_str = ""
				f.write("{},{}\n".format(peer, prefs_str))
		print("Done summarizing need meas")

	def prune_per_ingress_measurements(self):
		by_ingress_fn = os.path.join(CACHE_DIR, '{}_ingress_latencies_by_dst.csv'.format(
			self.system))
			
		res = parse_ingress_latencies_mp(by_ingress_fn)
		meas_by_popp, meas_by_ip = res['meas_by_popp'], res['meas_by_ip']

		anycast_latencies = {}
		for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(
			self.system)),'r'),desc='loading anycast latencies'):
				_,ip,lat,pop = row.strip().split(',')
				if lat == '-1': continue
				anycast_latencies[ip] = np.maximum(float(lat) * 1000,1)
		
		in_both = get_intersection(list(anycast_latencies), list(meas_by_ip))
		meas_by_ip = {ip:meas_by_ip[ip] for ip in in_both}	

		n_targs_per_popp = {popp:len(meas_by_popp[popp]) for popp in meas_by_popp}
		sorted_n_targs_per_popp = sorted(n_targs_per_popp.items(), key = lambda el : -1 * el[1])
		# Pop stockholm  ASN 6939 has 320158 targets google
		# Pop london  ASN 9002 has 204318 targets RETN (transit it seems)
		# Pop amsterdam  ASN 9002 has 197231 targets
		# Pop newyork  ASN 9002 has 174009 targets
		# Pop losangelas  ASN 4766 has 162511 targets Korea Telecom
		# Pop seattle  ASN 4766 has 161705 targets
		# Pop newyork  ASN 15169 has 44741 targets Google
		# Pop newyork  ASN 22773 has 44065 targets Cox
		# Pop sydney  ASN 15169 has 43730 targets Google
		# Pop seattle  ASN 15169 has 43658 targets 
		# Pop miami  ASN 15169 has 43509 targets
		# Pop melbourne  ASN 15169 has 43243 targets
		# Pop tokyo  ASN 15169 has 40825 targets
		# Pop amsterdam  ASN 33891 has 40574 targets core backbone, german transit
		# Pop losangelas  ASN 15169 has 39507 targets google
		# Pop warsaw  ASN 9009 has 31035 targets unlimited telecom, they do network infrastructure
		# Pop newyork  ASN 577 has 17925 targets canadian backbone
		# Pop melbourne  ASN 8075 has 14485 targets microsoft
		# Pop sydney  ASN 8075 has 14478 targets 
		# Pop newyork  ASN 4230 has 14412 targets IT provider of sorts
		# Pop newyork  ASN 812 has 13087 targets janky ISP of sorts
		# Pop seattle  ASN 812 has 11398 targets
		for popp,n in sorted_n_targs_per_popp:
			pop,peer = popp
			if n >= 10e3:
				print("Pop {}  ASN {} has {} targets".format(pop,peer,n))

		with open(os.path.join(TMP_DIR, 'keep_clients.csv'),'w') as f:
			for ip in meas_by_ip:
				f.write("{}\n".format(ip))

		## maybe limit those big boys like korea telecom even more, then output to a smaller performances
		## file and work with that locally

		valid_clients = {ip:None for ip in meas_by_ip}
		with open(by_ingress_fn, 'r') as f:
			with open(os.path.join(CACHE_DIR, 
					'filtered_{}_ingress_latencies_by_dst.csv'.format(self.system)),'w') as f2:
				for row in f:
					_,ip,pop,peer,lat = row.strip().split(',')
					if int(peer) in EXCLUDE_PEERS:
						continue
					try:
						valid_clients[ip]
					except KeyError:
						continue
					f2.write(row)


	def characterize_per_ingress_measurements(self):
		### Look at what we measured to for each ingress, some basic properties
		by_ingress_fn = os.path.join(CACHE_DIR, '{}_ingress_latencies_by_dst.csv'.format(
			self.system))
			
		res = parse_ingress_latencies_mp(by_ingress_fn)
		meas_by_popp, meas_by_ip = res['meas_by_popp'], res['meas_by_ip']
		anycast_latencies = {}
		anycast_pop = {}
		n_by_pop = {pop:0 for pop in POP_TO_LOC[self.system]}
		for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(
			self.system)),'r'),desc='loading anycast latencies'):
				_,ip,lat,pop = row.strip().split(',')
				if lat == '-1': continue
				anycast_latencies[ip] = np.maximum(float(lat) * 1000,1)
				anycast_pop[ip] = pop
				n_by_pop[pop] += 1
		print("Number of clients by PoP:\n{}".format(n_by_pop))

		from advertisement_experiments import Advertisement_Experiments
		ae = Advertisement_Experiments('vultr','measure_prepainter')
		all_peers = list(set([popp[1] for popp in ae.popps]))
		meas_peers = list(set([popp[1] for popp in meas_by_popp]))

		print("{} targets, {} ingresses with a measurement, {} peers".format(
			len(meas_by_ip), len(meas_by_popp), len(set(meas_peers))))

		measured_to_already = [tuple(row.strip().split(',')) for row 
			in open(os.path.join(CACHE_DIR, 'already_completed_popps.csv'),'r')]
		measured_to_peers = set(peer for pop,peer in measured_to_already)
		no_meas_peers = get_difference(all_peers, meas_peers)
		no_meas_peers = get_intersection(no_meas_peers, measured_to_peers)
		print("{} peers without a measurement".format(len(no_meas_peers)))
		self.summarize_need_meas(no_meas_peers)

		x,cdf_x = get_cdf_xy(list([len(meas_by_popp[popp]) for popp in meas_by_popp]),logx=True)
		plt.semilogx(x,cdf_x)
		plt.xlabel("Number of Reachable Targets by Ingress")
		plt.ylabel("CDF of Ingresses")
		plt.grid(True)
		self.save_fig('{}_n_targets_by_ingress.pdf'.format(self.system))

		x,cdf_x = get_cdf_xy(list([len(meas_by_ip[ip]) for ip in meas_by_ip]))
		plt.plot(x,cdf_x)
		plt.xlabel("Number of Reachable Ingresses by Target")
		plt.ylabel("CDF of Targets")
		plt.grid(True)
		self.save_fig('{}_n_ingresses_by_target.pdf'.format(self.system))


		ips_in_both = get_intersection(list(anycast_latencies), list(meas_by_ip))
		print("{} targets for anycast, {} for lat by dst, {} in both".format(
			len(anycast_latencies), len(meas_by_ip), len(ips_in_both)))

		n_saw_improve = {ip:0 for ip in ips_in_both}
		best_improve = {ip:0 for ip in ips_in_both}
		best_improve_intf = {ip:None for ip in ips_in_both}
		for ip in ips_in_both:
			for popp,lat in meas_by_ip[ip].items():
				if lat < anycast_latencies[ip]:
					delta = anycast_latencies[ip] - lat
					best_improve[ip] = np.maximum(delta, best_improve[ip])
					if delta == best_improve[ip]:
						best_improve_intf[ip] = popp
					best_improve[ip] = np.minimum(best_improve[ip],300)
					
					n_saw_improve[ip] += 1
		n_each_popp_best, n_each_pop_best = {},{}
		with open(os.path.join(TMP_DIR, 'best_improve_over_anycast_by_ip.csv'),'w') as f:
			f.write("ip,best_pop,best_peer,improve,anycast_lat,anycast_pop,2nd,2nd_lat,3rd,3rd_lat\n")
			for ip,imprv in best_improve.items():
				popp = best_improve_intf[ip]
				if popp is None:
					continue
				try:
					n_each_popp_best[popp] += 1
				except KeyError:
					n_each_popp_best[popp] = 1
				try:
					n_each_pop_best[popp[0]] += 1
				except KeyError:
					n_each_pop_best[popp[0]] = 1
				if imprv > 50:
					### approx location of this guy by getting lowest latencies
					closest_popps = sorted(meas_by_ip[ip].items(), key = lambda el : el[1])
					
					try:
						closest_second, closest_second_lat = closest_popps[1]
					except IndexError:
						closest_second, closest_second_lat = None,0
					try:
						closest_third, closest_third_lat = closest_popps[2]
					except IndexError:
						closest_third, closest_third_lat = None,0

					f.write("{},{},{},{},{},{},{},{},{},{}\n".format(ip,popp[0],popp[1],int(imprv),
						int(anycast_latencies[ip]), anycast_pop[ip], 
						closest_second, int(closest_second_lat), closest_third, int(closest_third_lat)))
		print(sorted(n_each_popp_best.items(), key = lambda el : el[1]))
		print(sorted(n_each_pop_best.items(), key = lambda el : el[1]))
		x,cdf_x = get_cdf_xy(list(n_saw_improve.values()))
		plt.plot(x,cdf_x)
		plt.grid(True)
		plt.xlabel("Number of Ingresses Giving Improvement Over Anycast")
		plt.ylabel("CDF of Targets")
		self.save_fig("{}_num_ingresses_giving_improve.pdf".format(self.system))

		x,cdf_x = get_cdf_xy(list(best_improve.values()))
		plt.plot(x,cdf_x)
		plt.xlabel("Best Improvement over Anycast (ms)")
		plt.ylabel("CDF of Targets")
		plt.grid(True)
		self.save_fig("{}_improvements_over_anycast.pdf".format(self.system))

	def compare_with_without_rpki(self):
		all_rpki_anycast_latencies = parse_anycast_latency_file(os.path.join(CACHE_DIR, '{}_anycast_latency_withrpki.csv'.format(
			self.system)),ignore_invalid=True)
		all_nonrpki_anycast_latencies = parse_anycast_latency_file(os.path.join(CACHE_DIR, '{}_anycast_latency_withoutrpki.csv'.format(
			self.system)), ignore_invalid=True)
		limit_to = get_intersection(all_rpki_anycast_latencies, all_nonrpki_anycast_latencies)

		rpki_anycast_latencies = parse_anycast_latency_file(os.path.join(CACHE_DIR, '{}_anycast_latency_withrpki.csv'.format(
			self.system)))
		nonrpki_anycast_latencies = parse_anycast_latency_file(os.path.join(CACHE_DIR, '{}_anycast_latency_withoutrpki.csv'.format(
			self.system)))

		in_rpki_not_without = get_difference(rpki_anycast_latencies, nonrpki_anycast_latencies)
		in_without_not_rpki = get_difference(nonrpki_anycast_latencies, rpki_anycast_latencies)
		in_both = get_intersection(rpki_anycast_latencies, nonrpki_anycast_latencies)

		in_rpki_not_without = get_intersection(in_rpki_not_without, limit_to)
		in_without_not_rpki = get_intersection(in_without_not_rpki, limit_to)
		print("IWNR: {} ".format(in_without_not_rpki[0:10]))
		in_both = get_intersection(in_both, limit_to)

		print("Total: {} ,Only in RPKI : {}, In Both : {}, Only without RPKI : {}".format(len(limit_to), len(in_rpki_not_without),
			len(in_both), len(in_without_not_rpki)))

		by_asn = {'rpki_block': {}, 'total': {}}
		for targ,pop in limit_to:
			asn = self.parse_asn(targ)
			try:
				by_asn['total'][asn].append(targ)
			except KeyError:
				by_asn['total'][asn] = [targ]
		## do they hit the same pop / can pop explain the split behavior
		## 
		print("Measured {} ASes total".format(len(by_asn['total'])))
		for targ,pop in in_rpki_not_without:
			asn = self.parse_asn(targ)
			try:
				by_asn['rpki_block'][asn].append(targ)
			except KeyError:
				by_asn['rpki_block'][asn] = [targ]
		for asn, targs in sorted(by_asn['rpki_block'].items(), key = lambda el : -1 * len(el[1]))[0:100]:
			print("{} ({}) -- {} targs, {} frac".format(asn,self.org_to_as.get(asn,None),len(targs),len(targs)/len(by_asn['total'][asn])))

		all_fracs = list([len(by_asn['rpki_block'].get(asn,[])) / len(by_asn['total'][asn]) for 
			asn in by_asn['total']])
		x,cdf_x = get_cdf_xy(all_fracs)
		plt.plot(x,cdf_x)
		plt.xlabel("Fraction of Unresponsive Targets When not RPKI")
		plt.ylabel("CDF of ASes")
		plt.grid(True)
		plt.savefig('figures/rpki_unresponsiveness.pdf')

	def compare_mine_jc_anycast(self):
		my_anycast_latencies = parse_anycast_latency_file(os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(
			self.system)))
		jc_ip_catchments = {}
		for row in open(os.path.join(CACHE_DIR, 'jc_{}_anycast_catchment.csv'.format(self.system)),'r'):
			ip,catchment = row.strip().split('\t')
			jc_ip_catchments[ip] = catchment
		jc_anycast_latencies = {}
		for row in open(os.path.join(CACHE_DIR, 'jc_{}_anycast_latency.csv'.format(self.system)),'r'):
			ip, lat = row.strip().split('\t')
			try:
				jc_ip_catchments[ip]
			except KeyError:
				continue
			lat = float(lat) * 1000
			if lat > 2000:continue
			jc_anycast_latencies[ip,jc_ip_catchments[ip]] = lat
		jc_anycast_latencies_new = {}
		for row in open(os.path.join(CACHE_DIR, 'jc_{}_anycast_latency_new.csv'.format(self.system)),'r'):
			ip, lat = row.strip().split('\t')
			try:
				jc_ip_catchments[ip]
			except KeyError:
				continue
			lat = float(lat)
			if lat > 2000:continue
			jc_anycast_latencies_new[ip,jc_ip_catchments[ip]] = lat
		

		my_anycast_latencies = {k:v for k,v in my_anycast_latencies.items() if len(v) == 2}
		ips_in_both = get_intersection(my_anycast_latencies,jc_anycast_latencies)
		print("Comparing {} IPs".format(len(ips_in_both)))
		ips_in_both = get_intersection(ips_in_both, jc_anycast_latencies_new)
		print("{} IPs in jc new".format(len(jc_anycast_latencies_new)))
		print("Comparing {} IPs".format(len(ips_in_both)))


		diffs = list([min(my_anycast_latencies[ip]) - jc_anycast_latencies[ip] for ip in ips_in_both])
		x,cdf_x = get_cdf_xy(diffs)
		plt.plot(x,cdf_x,label="Mine - JC Old")

		diffs = list([min(my_anycast_latencies[ip]) - jc_anycast_latencies_new[ip] for ip in ips_in_both])
		x,cdf_x = get_cdf_xy(diffs)
		plt.plot(x,cdf_x,label="Mine - JC New")

		diffs = list([jc_anycast_latencies[ip] - jc_anycast_latencies_new[ip] for ip in ips_in_both])
		x,cdf_x = get_cdf_xy(diffs)
		plt.plot(x,cdf_x,label="JC Old - JC New")

		# Compare mine at two separate time slices
		diffs = []
		for ip in my_anycast_latencies:
			if len(my_anycast_latencies[ip]) == 2:
				diffs.append(my_anycast_latencies[ip][1] - my_anycast_latencies[ip][0])
		x,cdf_x = get_cdf_xy(diffs)
		plt.plot(x,cdf_x, label="Later - Earlier")

		plt.xlim([-40,40])
		plt.xlabel("Difference in Anycast Lat (ms)")
		plt.ylabel("CDF of Targets")
		plt.grid(True)
		plt.legend()
		self.save_fig("jc_me_anycast_comparison.pdf")

	def plot_lat_over_time(self):
		### high level question : how often do we see cases where a path decision function would be non-trivial?

		## intuitively, if the "best path" is unstable, and the second-best path is close to it,
		## it could be interesting
		## most interesting would be cases where the paths have predictable "noises", 
		## instead of something like random shot noise


		## are there enough cases where I could make an interesting decision?
		## straw-man : assume 10s flows, place each flow on lowest latency path at each point in time,
		## average/max latency of flow - best possible over time and over dests
		## for the cases where I could make an interesting decision, is it feasible to estimate that noise?

		np.random.seed(31415)
		cache_fn = os.path.join(CACHE_DIR, 'jitter_metrics.pkl')
		from scipy.ndimage import uniform_filter1d
		N_move_average = 4
		from sklearn.cluster import Birch
		import pandas as pd

		def characterize_time_series(lbp,vbp,verb=False):
			typical_lats_by_popp = {popp: np.median(lats) for popp,lats in lbp.items()}
			ranked_lbp = sorted(typical_lats_by_popp.items(), key = lambda el : el[1])
			best_popp, second_best_popp = ranked_lbp[0][0],ranked_lbp[1][0]

			# check for path changes
			# path changes skew variance estimates
			# means = cluster
			# if more than one (n_pts at mean > 10% of n_total_pts): path change
			has_path_change = False
			for popp in [best_popp, second_best_popp]:
				if vbp[popp] > 1:
					brc = Birch(threshold=4,n_clusters=None)
					labels = brc.fit_predict(lbp[popp].reshape(-1,1))
					u,c = np.unique(labels,return_counts=True)
					total_n = len(lbp[popp])
					frac = c / total_n
					if sum(frac > .1 ) > 1:
						has_path_change = True
						break

			## check for persistent congestion
			congested_popps = {}
			for popp,lats in lbp.items():
				pdlat = pd.DataFrame(lats)
				avged_lat = np.array(pdlat.rolling(N_move_average,min_periods=1).median()).flatten()
				mlat = np.percentile(lats,1) # propagation delay
				lat_mask = (avged_lat > (mlat + 5)).astype(np.int32)
				## fill in holes via voting
				hole_filled_lat_mask = np.zeros(len(lat_mask))
				for i in range(len(lats)):
					hole_filled_lat_mask[i] = int(np.round(np.average(lat_mask[np.maximum(0,i-N_move_average):np.maximum(i+1,len(lats))])))
				## look for blocks of "ones" greater than a certain length
				congested_blocks, n_congest, congested, was_ever_uncongested = [], 0, False, False
				# if verb:
				# 	print("{} {} {}".format(avged_lat, lat_mask, hole_filled_lat_mask))
				for i, l in enumerate(hole_filled_lat_mask):
					if l and was_ever_uncongested:
						n_congest += 1
						if n_congest == N_move_average:
							congested = True
							# start of congestion is N_move_average places ago
							congest_start = np.maximum(i - N_move_average,0)
					elif not l:
						was_ever_uncongested = True
						if congested:
							# end of congestion
							congest_end = i
							congested_blocks.append((congest_start, congest_end))
							congested = False
						n_congest = 0
				congested_popps[popp] = congested_blocks
			any_congestion = any(len(blocks) > 0 for popp, blocks in congested_popps.items())
			return {
				'best_popp': best_popp,
				'second_best_popp': second_best_popp,
				'has_path_change': has_path_change,
				'has_congestion': any_congestion,
				'congested_popps': congested_popps,
			}

		if not os.path.exists(cache_fn):
			

			
			mean_lats_by_targ = {}


			ret = read_lats_over_time()
			lats_by_targ = ret['lats_by_targ']
			i_to_popp = ret['i_to_popp']
			i_to_targ = ret['i_to_targ']

			cluster = False
			brc = None
			avg_over = 3000 # seconds, clustering time
			loss_by_targ = {}
			for targ in tqdm.tqdm(lats_by_targ,desc="Post-processing latency data"):
				n_pts = 0
				loss_by_targ[targ] = {}
				mean_lats_by_targ[targ] = 0
				for popp in lats_by_targ[targ]:
					if cluster:
						if brc is None:
							ts = np.array([el[0] for el in lats_by_targ[targ][popp]])
							brc = Birch(threshold=avg_over,n_clusters=None)
							labels = brc.fit_predict(ts.reshape(-1,1))
						new_lats = {}
						for t,lat in lats_by_targ[targ][popp]:
							t_map = brc.subcluster_centers_[np.argmin(np.abs(brc.subcluster_centers_ - t))][0]
							try:
								new_lats[t_map].append(lat)
							except KeyError:
								new_lats[t_map] = [lat]
						new_lats_avg = {t_map: np.mean(new_lats[t_map]) for t_map in new_lats}
						lats_by_targ[targ][popp] = sorted(new_lats_avg.items(), key = lambda el : el[0])
						loss_by_targ[targ][popp] = [np.maximum(4 - len(new_lats[t_map]),0) for t_map,_ in lats_by_targ[targ][popp]]
					else:
						loss_by_targ[targ][popp] = np.zeros((len(lats_by_targ[targ]), ))
					for t,l in lats_by_targ[targ][popp]:
						mean_lats_by_targ[targ] += l
						n_pts += 1

					## reject outliers (temporary spikes)
					lats = np.array(lats_by_targ[targ][popp])[:,1]
					mlat = np.median(lats)
					devlat = np.sqrt(np.var(lats))
					rmvs = (lats > (mlat + 11 * devlat))
					notboring = False
					needs_plugging = np.where(rmvs)[0]
					# if len(needs_plugging) > 0:
					# 	plt.plot(lats)
					# 	plt.savefig('figures/tmp{}.pdf'.format(targ))
					# 	plt.clf()
					# 	plt.close()
					for rmv in needs_plugging:
						lats_by_targ[targ][popp][rmv] = list(lats_by_targ[targ][popp][rmv])
						if rmv == 0:
							lats_by_targ[targ][popp][rmv][1] = lats[rmv+1]
						elif rmv == len(lats) - 1:
							lats_by_targ[targ][popp][rmv][1] = lats[rmv-1]
						else:
							lats_by_targ[targ][popp][rmv][1] = np.mean([lats[rmv-1],lats[rmv+1]])
						lats_by_targ[targ][popp][rmv] = tuple(lats_by_targ[targ][popp][rmv])

				mean_lats_by_targ[targ] = mean_lats_by_targ[targ] / n_pts
			for targ in list(lats_by_targ):
				if len(lats_by_targ[targ]) == 1:
					del lats_by_targ[targ]
					del mean_lats_by_targ[targ]
					del loss_by_targ[targ]
				

			pickle.dump([mean_lats_by_targ, lats_by_targ, loss_by_targ, i_to_popp, i_to_targ], open(cache_fn,'wb'))
		else:
			mean_lats_by_targ, lats_by_targ, loss_by_targ, i_to_popp, i_to_targ = pickle.load(open(cache_fn,'rb'))
			pass
		targs = list(lats_by_targ)
		np.random.shuffle(targs)

		## straw man
		popps = list(set(popp for targ in lats_by_targ for popp in lats_by_targ[targ]))
		cols = ['red','black','royalblue','magenta','darkorange','forestgreen','tan']
		popp_to_col = {popp:c for popp,c in zip(popps,cols)}

		popp_to_ind = {popp:i for i,popp in enumerate(popps)}


		# interesting_targs = [targ for targ,perfs in assessed_delta_perfs.items() if any(p > 15 for p in perfs)]
		interesting_targs = []
		for targ in tqdm.tqdm(lats_by_targ,desc="Identifying interesting targets to plot."):
			lats_by_popp = {popp: np.array([el[1] for el in lats_by_targ[targ][popp]]) for popp in lats_by_targ[targ]}
			var_by_popp = {popp: np.var(lats_by_popp[popp]) for popp in lats_by_popp}
			ret = characterize_time_series(lats_by_popp, var_by_popp)
			if ret['has_congestion']:
				if len(ret['congested_popps'][ret['best_popp']]) > 0:
					interesting_targs.append((targ,ret))

		plt.rcParams["figure.figsize"] = (40,90)
		nrows,ncols = 25,8
		plt_every = 1
		f,ax = plt.subplots(nrows,ncols)
		for targi, (targ,ret) in enumerate(sorted(interesting_targs)[0:200]):
			axrow = targi // ncols
			axcol = targi % ncols
			ax2 = ax[axrow,axcol].twinx()
			for popp in sorted(lats_by_targ[targ]):
				measx = np.array([el[0] for el in lats_by_targ[targ][popp]])
				measx = measx - measx[0]
				measy = np.array([el[1] for el in lats_by_targ[targ][popp]])
				if len(ret['congested_popps'][popp]) > 0:
					cbstr = "--".join(["{}-{}".format(int(measx[el[0]]),
						int(measx[el[1]])) for el in ret['congested_popps'][popp]])
					lab = "{}-{}".format(i_to_popp[popp],cbstr)
				else:
					lab = i_to_popp[popp]
				ax[axrow,axcol].plot(measx[::plt_every],measy[::plt_every],label=lab,color=popp_to_col[popp])

				lossy = loss_by_targ[targ][popp]
				ax2.scatter(measx,lossy,color=popp_to_col[popp])
			ax2.set_ylim([0,4])
			# ax[axrow,axcol].set_ylim([10,300])
			ax[axrow,axcol].set_title(i_to_targ[targ])
			ax[axrow,axcol].legend(fontsize=5)
		plt.savefig('figures/latency_over_time_interesting_targs.pdf')
		plt.clf(); plt.close()





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
		plt.savefig('figures/deltaperfs-strawman.pdf')
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
		plt.savefig('figures/jitter_metrics_investigation.pdf')
		plt.clf(); plt.close()


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

	def user_prefix_impact(self):
		perfs = {
			'with_users': {},
			'without_users': {},
		}

		for fn, k in zip([os.path.join(CACHE_DIR, '{}_anycast_latency_yunfan.csv'.format(self.system)),
			os.path.join(CACHE_DIR, '{}_anycast_latency_noyunfan.csv'.format(self.system))], ['with_users', 
			'without_users']):
			pops = {}
			for row in open(fn,'r'):
				t,dst,l,pop = row.strip().split(',')
				if l == '-1': continue
				l = np.minimum(np.maximum(1,float(l) * 1000),500)
				pops[pop] = None
				pref = self.routeviews_pref_to_asn.get_key(dst)
				if pref is None: continue
				try:
					perfs[k][pref].append(l)
				except KeyError:
					perfs[k][pref] = [l]
			print("{} -- {} responsive dsts at {} pops".format(k,len(perfs[k]), len(pops)))
			print(pops.keys())
			for pref in list(perfs[k]):
				perfs[k][pref] = np.median(perfs[k][pref])
		x,cdf_x = get_cdf_xy(list(perfs['with_users'].values()))
		plt.plot(x,cdf_x,label="With Users")
		x,cdf_x = get_cdf_xy(list(perfs['without_users'].values()))
		plt.plot(x,cdf_x,label="Without Users")
		plt.xlabel("Anycast Latency (ms)")
		plt.ylabel("CDF of Destinations")
		plt.legend()
		plt.grid(True)
		plt.xlim([0,300])
		plt.ylim([0,1])
		plt.savefig('figures/with_without_users_anycast_performance.pdf')

	def get_probing_targets(self):
		import pytricia
		probe_target_fn = os.path.join(CACHE_DIR , 'interesting_targets_to_probe.csv')
		user_pref_tri = pytricia.PyTricia()
		for row in open(os.path.join(CACHE_DIR, 'yunfan_prefixes_with_users.csv')):
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
		self.lookup_asns_if_needed(list(set([ip32_to_24(ip) for ip,pop in anycast_latencies])))
		for (ip,pop), lats in anycast_latencies.items():
			parent_pref = user_pref_tri.get_key(ip)
			lat = np.min(lats)
			if lat < 2 or lat > 2000: continue
			n_valid_latency += 1

			ip_asn = self.parse_asn(ip)
			all_asns[ip_asn] = None

			if parent_pref is not None:
				parent_key = self.parse_asn(parent_pref)
				if parent_key in ignore_ases: continue
				try:
					targs_in_user_prefs[parent_key].append((ip,float(lat)*1000))
				except KeyError:
					targs_in_user_prefs[parent_key] = [(ip,float(lat)*1000)]
		print("{} targets with valid latency, {} ASNs".format(n_valid_latency, len(all_asns)))
		print("There are {} ASNs with pingable targets matching our criteria".format(len(targs_in_user_prefs)))
		used_prefs = {}
		with open(probe_target_fn,'w') as f:
			popps = list(provider_popps)
			popps = [str(popp[0]) + "|" + str(popp[1]) for popp in popps]
			popps_str = "-".join(popps)
			for parent_asn, targslats in targs_in_user_prefs.items():
				sorted_targs = sorted(targslats, key = lambda el : np.abs(el[1]-20))
				for t,l in sorted_targs[0:40]:
					parent_pref = user_pref_tri.get_key(t)
					try:
						used_prefs[parent_pref]
						continue
					except KeyError:
						used_prefs[parent_pref] = None
					# org = self.org_to_as.get(parent_asn, [parent_asn])[0]
					# f.write("{},{},{}\n".format(org,t,l))
					f.write("{},{}\n".format(t,popps_str))

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

	def unicast_anycast_withdrawal_experiment(self):
		plot_cache_fn = os.path.join(CACHE_DIR, 'anycast_unicast_withdrawal_quickplotcache.pkl')
		if not os.path.exists(plot_cache_fn):
			raw_results = pickle.load(open(os.path.join(CACHE_DIR,
				'unicast_anycast_experiment_results.pkl'),'rb'))
			parsed_lats = {}
			parsed_losses = {}
			parsed_ts = {}
			t_start_overall = np.inf
			t_end_overall = -1 * np.inf
			for dst in raw_results:
				for target in raw_results[dst]:
					try:
						parsed_lats[target]
					except KeyError:
						parsed_lats[target] = {}
						parsed_losses[target] = {}
						parsed_ts[target] = {}
					try:
						parsed_lats[target][dst]
					except KeyError:
						parsed_lats[target][dst] = []
						parsed_losses[target][dst] = []
						parsed_ts[target][dst] = []
					for res in raw_results[dst][target]:
						if res['t_start'] is None: continue
						res['t_start'] += TIME_OFFSET
						parsed_ts[target][dst].append(res['t_start'])
						if res['rtt'] is None: 
							parsed_losses[target][dst].append(res['t_start'])
							continue
						if res['t_start'] < t_start_overall:
							t_start_overall = res['t_start']
						if res['t_start'] > t_end_overall:
							t_end_overall = res['t_start']
						parsed_lats[target][dst].append((res['t_start'], res['rtt']))
			do_plot=True
			if do_plot:
				plt.rcParams["figure.figsize"] = (40,90)
				nrows,ncols = 25,8
				f,ax = plt.subplots(nrows,ncols)
			lats_np = {}
			losses_np = {}
			loss_aggr_period = 1 # second
			n_bins_loss = int(np.ceil((t_end_overall - t_start_overall) / loss_aggr_period)) + 1
			for targi,targ in enumerate(parsed_lats):
				lats_np[targ] = {}
				losses_np[targ] = {}
				for dst in parsed_lats[targ]:
					if len(parsed_lats[targ][dst]) == 0:continue
					these_lats = np.array(parsed_lats[targ][dst])
					t = these_lats[:,0] - t_start_overall
					losses_np[targ][dst] = np.zeros((2,n_bins_loss))
					for _t in parsed_ts[targ][dst]:
						_t = _t - t_start_overall
						bini = int(_t/loss_aggr_period)
						losses_np[targ][dst][0,bini] += 1
					for tloss in parsed_losses[targ][dst]:
						tloss = tloss - t_start_overall
						bini = int(tloss/loss_aggr_period)
						losses_np[targ][dst][1,bini] += 1
					losses_np[targ][dst] = losses_np[targ][dst][1,:] / (losses_np[targ][dst][0,:] + .00001)

					lats = 1000 * these_lats[:,1]
					lats_np[targ][dst] = np.array([t,lats])
					if do_plot:
						axrow = targi // ncols
						axcol = targi % ncols
						if axrow >= nrows or axcol >= ncols:
							continue

						ax[axrow,axcol].plot(t,lats,label=dst)
						ax[axrow,axcol].set_title(targ)
						ax[axrow,axcol].legend(fontsize=5)
						
			if do_plot:
				plt.savefig('figures/unicast_anycast_withdrawal_all_targs.pdf')
				plt.clf(); plt.close()

			update_times = pickle.load(open('cache/prefix_update_times.pkl','rb'))
			update_times = np.array(update_times) - t_start_overall
			bin_freq = 1 # seconds
			min_ut, max_ut = np.min(update_times), np.max(update_times)
			n_bins = int(np.ceil((max_ut - min_ut) / bin_freq)) + 1
			update_ct = np.zeros((n_bins))
			for ut in update_times:
				_bin = int((ut - min_ut) // bin_freq)
				update_ct[_bin] += 1
			update_times_arr = np.array([np.linspace(min_ut, max_ut, num=len(update_ct)), update_ct])
			lats_np['ris_updates'] = update_times_arr

			pickle.dump([lats_np,losses_np], open(plot_cache_fn,'wb'))
		else:
			try:
				[lats_np,losses_np] = pickle.load(open(plot_cache_fn,'rb'))
			except:
				lats_np = pickle.load(open(plot_cache_fn,'rb')) # backwards compatability
		targ_of_interest = "138.124.187.148"
		those_lats = lats_np[targ_of_interest]
		ris_updates = lats_np['ris_updates']


		# Manually finding the blackhole time
		tms = those_lats['184.164.239.2'][0,:]
		lts = those_lats['184.164.239.2'][1,:]

		# for t,d in zip(tms[0:-1], np.diff(tms)):
		# 	if d > .3:
		# 		print("{} {}".format(t,d))
		# for t,l,d in zip(tms[0:-1],lts[0:-1], np.diff(lts)):
		# 	if np.abs(d) > 2.5:
		# 		print("{} {} {}".format(t,l,d))

		# blackhole_time = 101.28439426422119
		blackhole_time = 106.79071998596191
		blackhole_time_i = np.where(tms==blackhole_time)[0][0]


		anycast_path_pre = those_lats['184.164.239.1'][:,0:blackhole_time_i]
		anycast_path_pre[1,:] += 3
		anycast_path_post = those_lats['184.164.239.2'][:,blackhole_time_i+1:]

		those_lats['anycast'] = np.concatenate([anycast_path_pre,anycast_path_post],axis=1)
		those_lats['anycastpre'] = anycast_path_pre
		those_lats['anycastpost'] = anycast_path_post

		self.plot_withdrawal_time_series(those_lats, ris_updates, blackhole_time)
		for plotnumi in range(6):
			self.plot_withdrawal_time_series_for_ppt(those_lats, ris_updates, blackhole_time, blackhole_time_i, plotnumi)

	def plot_withdrawal_time_series(self, latency_over_time, ris_updates,
			blackhole_time):
		## Time series showing path selection and path latencies
		start_t = 0
		blackhole_times = [blackhole_time]
		chosen_path_times = [(0,'optimal'),(blackhole_time-.2,'suboptimal'),(blackhole_time+1000,'suboptimal')]
		intf_to_label = { # TODO
			"184.164.238.1": "Unicast Path",
			"anycastpre": "Anycast Path",
			"anycastpost": "Anycast Path",
			"184.164.240.1": "Alternate Path 1",
			"184.164.242.1": "Alternate Path 2",
			"184.164.243.1": "Alternate Path 3",
		}
		name_intf_mapping = {
			'optimal': '184.164.238.1',
			'suboptimal': '184.164.240.1',
		}
		intfs = sorted(list(intf_to_label))
		f,ax = self.get_figure(fs=21,fw=13)
		max_y = 95
		rng=[blackhole_time - 60, blackhole_time + 100]
		intf_colors = {intf: ['red','blue','brown','tan','salmon','salmon'][i] for i,intf in enumerate(intfs)}
		painter_color = 'forestgreen'
		# latency over time
		for intf in intfs:
			t_arr = latency_over_time[intf][0,:] - start_t
			lat_arr = latency_over_time[intf][1,:]
			ax.plot(t_arr, lat_arr, c=intf_colors[intf])

		anycast_unavailable_length = 1.2#recover_time - blackhole_time


		# blackhole events
		this_blackhole = [t for t in blackhole_times if t>=rng[0] and t<=rng[1]][0]
		ax.vlines(x=this_blackhole-.3, ymin=0, ymax=max_y, color='k', linestyle='dashed')
		ax.vlines(x=this_blackhole+anycast_unavailable_length,ymin=0, ymax=max_y, color='r', linestyle='dashed')
		ax.vlines(x=this_blackhole+60, ymin=0, ymax=max_y, color='y', linestyle='dashed')
		# ax.annotate("Network\nBlackhole", (this_blackhole+1, 50),fontweight='bold')
		ax.annotate("   PoP\nFailure", (this_blackhole-10, 84),fontweight='bold', color='white', backgroundcolor='black')
		# ax.annotate("PAINTER\nPath Choice", (this_blackhole+1, 295), c='forestgreen',fontweight='bold')
		
		ax.annotate("   PAINTER\nPath Choices", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
		ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
			arrowprops=dict(lw=.4,color=painter_color))
		ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole+55,43),
			arrowprops=dict(lw=.4,color=painter_color))
		
		
		# ax.annotate("DNS\nReaction", (this_blackhole+61, 295), c='r', fontweight='bold')
		ax.annotate("  Normal\nOperation", (this_blackhole-50, 63), c='blue',fontweight='bold',
			rotation=30)
		ax.annotate("Unavailable\n Using DNS", (this_blackhole+28, 55), c='y', fontweight='bold',
			rotation=30)
		
		ax.annotate("Unavailable\n Using Anycast", (this_blackhole+4, 58), c='r', 
			fontweight='bold', rotation=30)
		ax.annotate("",xytext=(this_blackhole+6,62),xy=(this_blackhole+.5,58),
			arrowprops=dict(lw=.4,color='red'))
		

		ax.fill_between(np.linspace(this_blackhole-.3,this_blackhole+anycast_unavailable_length),0,max_y, color='red', alpha=.1)
		ax.fill_between(np.linspace(this_blackhole+anycast_unavailable_length,this_blackhole+60),0,max_y, color='yellow', alpha=.1)
		
		# highlight chosen path
		for i in range(len(chosen_path_times) - 1):
			this_t = chosen_path_times[i][0] - start_t
			next_t = chosen_path_times[i+1][0] - start_t
			this_path = chosen_path_times[i][1]
			intf = name_intf_mapping[this_path]
			data_this_path = latency_over_time[intf]
			t_arr = data_this_path[0,:] -  start_t
			inds = [i for i,_t in enumerate(t_arr) if _t > this_t and _t <= next_t]
			if inds == []: continue
			# inds = inds + [max(inds) + 1]
			lat_arr = data_this_path[1,inds]
			t_arr = t_arr[inds]
			ax.plot(t_arr,lat_arr,color=painter_color,linewidth=10,alpha=.4)
			# ax.plot(t_arr,lat_arr,color=intf_colors[intf],linewidth=10,alpha=.4)

		# Label lines by annotation
		ax.annotate("Anycast Path", (this_blackhole-55, 27), color='salmon')
		ax.annotate("Unicast Path", (this_blackhole-50, 17), color='chocolate')
		ax.annotate("Alternate Unicast\n         Paths", (this_blackhole - 55, 52))

		ax.set_ylim([0,max_y])
		ax.set_xlabel("Time (s)")
		ax.set_ylabel("RTT Latency (ms)")
		ax.set_xlim(rng)
		# set the x labels to be between zero and range length, for presentation purposes
		labels = [item.get_text() for item in ax.get_xticklabels()]
		n_labels = len(labels)
		dlta = (rng[1] - rng[0]) * 1.0 / n_labels
		for i in range(n_labels):
			labels[i] = round(dlta * i)
		ax.set_xticklabels(labels)


		ax2col = 'fuchsia'
		ax2 = ax.twinx()
		ax2.plot(ris_updates[0,:], ris_updates[1,:], marker='.',color=ax2col, linewidth=2)
		ax2.set_ylabel("# Anycast Prefix BGP Updates")
		ax2.yaxis.label.set_color(ax2col)
		ax2.tick_params(axis='y', colors=ax2col)  
		ax2.annotate("# BGP Updates", (this_blackhole-45,8), color=ax2col, fontsize=16)  


		# axzoom = f.add_axes([.75,.58,.13,.2])
		# for intf in intfs:
		# 	t_arr = latency_over_time[intf][0,:] - start_t
		# 	lat_arr = latency_over_time[intf][1,:]
		# 	axzoom.plot(t_arr, lat_arr, c=intf_colors[intf])

		# anycast_unavailable_length = 1.2#recover_time - blackhole_time


		# # blackhole events
		# this_blackhole = [t for t in blackhole_times if t>=rng[0] and t<=rng[1]][0]
		# axzoom.vlines(x=this_blackhole-.3, ymin=0, ymax=max_y, color='k', linestyle='dashed')
		# axzoom.vlines(x=this_blackhole+anycast_unavailable_length,ymin=0, ymax=max_y, color='r', linestyle='dashed')
		# axzoom.vlines(x=this_blackhole+60, ymin=0, ymax=max_y, color='y', linestyle='dashed')
		
		# axzoom.fill_between(np.linspace(this_blackhole-.3,this_blackhole+anycast_unavailable_length),0,max_y, color='red', alpha=.1)
		# axzoom.fill_between(np.linspace(this_blackhole+anycast_unavailable_length,this_blackhole+60),0,max_y, color='yellow', alpha=.1)

		# axzoom.annotate("Available Using PAINTER", (blackhole_time-1.3,36),
		# 	fontweight='bold', fontsize=8, backgroundcolor='white', 
		# 	color='forestgreen')

		# ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole+77,57),
		# 	arrowprops=dict(lw=.4,color=painter_color))

		# # highlight chosen path
		# for i in range(len(chosen_path_times) - 1):
		# 	this_t = chosen_path_times[i][0] - start_t
		# 	next_t = chosen_path_times[i+1][0] - start_t
		# 	this_path = chosen_path_times[i][1]
		# 	intf = name_intf_mapping[this_path]
		# 	data_this_path = latency_over_time[intf]
		# 	t_arr = data_this_path[0,:] -  start_t
		# 	inds = [i for i,_t in enumerate(t_arr) if _t > this_t and _t <= next_t]
		# 	if inds == []: continue
		# 	# inds = inds + [max(inds) + 1]
		# 	lat_arr = data_this_path[1,inds]
		# 	t_arr = t_arr[inds]
		# 	axzoom.plot(t_arr,lat_arr,color=painter_color,linewidth=10,alpha=.4)
		# axzoom.set_xlim([blackhole_time-1.5,blackhole_time+2.5])
		# axzoom.set_xticks([])
		# axzoom.set_yticks([])
		# axzoom.set_ylim([20,57])

		self.save_fig("system_demonstration_time_series.pdf")

	def plot_withdrawal_time_series_for_ppt(self, latency_over_time, ris_updates,
			blackhole_time, blackhole_time_i, plotnumi):
		## Time series showing path selection and path latencies
		start_t = 0
		blackhole_times = [blackhole_time]
		chosen_path_times = [(0,'optimal'),(blackhole_time-.2,'suboptimal'),(blackhole_time+1000,'suboptimal')]
		intf_to_label = { # TODO
			"184.164.238.1": "Unicast Path",
			"anycastpre": "Anycast Path",
			"anycastpost": "Anycast Path",
			"184.164.240.1": "Alternate Path 1",
			"184.164.242.1": "Alternate Path 2",
			"184.164.243.1": "Alternate Path 3",
		}
		name_intf_mapping = {
			'optimal': '184.164.238.1',
			'suboptimal': '184.164.240.1',
		}
		intfs = sorted(list(intf_to_label))
		f,ax = self.get_figure(fs=21,fw=13)
		max_y = 95
		rng=[blackhole_time - 60, blackhole_time + 100]
		this_blackhole = [t for t in blackhole_times if t>=rng[0] and t<=rng[1]][0]
		intf_colors = {intf: ['red','blue','brown','tan','salmon','salmon'][i] for i,intf in enumerate(intfs)}
		painter_color = 'forestgreen'
		# latency over time
		
		
		ax.set_ylim([0,max_y])
		ax.set_xlabel("Time (s)")
		ax.set_ylabel("RTT Latency (ms)")
		ax.set_xlim(rng)
		# set the x labels to be between zero and range length, for presentation purposes
		labels = [item.get_text() for item in ax.get_xticklabels()]
		n_labels = len(labels)
		dlta = (rng[1] - rng[0]) * 1.0 / n_labels
		for i in range(n_labels):
			labels[i] = round(dlta * i)
		ax.set_xticklabels(labels)
		if plotnumi == 0:
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return

		ax.annotate("Anycast Path", (this_blackhole-55, 27), color='salmon')
		ax.annotate("PAINTER Advertised\n Paths", (this_blackhole - 55, 57))
		for intf in intfs:
			t_arr = latency_over_time[intf][0,:] - start_t
			lat_arr = latency_over_time[intf][1,:]
			if t_arr[0] > blackhole_time -5: continue
			ax.plot(t_arr[0:blackhole_time_i], lat_arr[0:blackhole_time_i], c=intf_colors[intf])


		ax.annotate("  Normal\nOperation", (this_blackhole-50, 73), c='blue',fontweight='bold',
			rotation=30)
		anycast_unavailable_length = 1.2#recover_time - blackhole_time
		if plotnumi == 1:
			ax.annotate("   PAINTER\nPath Choice", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
			ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
				arrowprops=dict(lw=.4,color=painter_color))
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return


		# blackhole events
		ax.vlines(x=this_blackhole-.3, ymin=0, ymax=max_y, color='k', linestyle='dashed')
		ax.annotate("   PoP\nFailure", (this_blackhole-10, 84),fontweight='bold', color='white', backgroundcolor='black')
		if plotnumi == 2:
			ax.annotate("   PAINTER\nPath Choice", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
			ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
				arrowprops=dict(lw=.4,color=painter_color))
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return

		for intf in intfs:
			t_arr = latency_over_time[intf][0,:] - start_t
			lat_arr = latency_over_time[intf][1,:]
			ax.plot(t_arr, lat_arr, c=intf_colors[intf])
		if plotnumi == 3:
			ax.annotate("   PAINTER\nPath Choice", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
			ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
				arrowprops=dict(lw=.4,color=painter_color))
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return
		

		ax.vlines(x=this_blackhole+anycast_unavailable_length,ymin=0, ymax=max_y, color='r', linestyle='dashed')
		ax.annotate("Unavailable\n Using Anycast", (this_blackhole+4, 58), c='r', 
			fontweight='bold', rotation=30)
		ax.annotate("",xytext=(this_blackhole+6,62),xy=(this_blackhole+.5,58),
			arrowprops=dict(lw=.4,color='red'))
		ax.fill_between(np.linspace(this_blackhole-.3,this_blackhole+anycast_unavailable_length),0,max_y, color='red', alpha=.1)
		if plotnumi == 4:
			ax.annotate("   PAINTER\nPath Choice", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
			ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
				arrowprops=dict(lw=.4,color=painter_color))
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return


		ax.vlines(x=this_blackhole+60, ymin=0, ymax=max_y, color='y', linestyle='dashed')
		ax.annotate("Unavailable\n Using DNS", (this_blackhole+28, 55), c='y', fontweight='bold',
		rotation=30)
		# ax.annotate("Network\nBlackhole", (this_blackhole+1, 50),fontweight='bold')
		# ax.annotate("PAINTER\nPath Choice", (this_blackhole+1, 295), c='forestgreen',fontweight='bold')
		ax.fill_between(np.linspace(this_blackhole+anycast_unavailable_length,this_blackhole+60),0,max_y, color='yellow', alpha=.1)
		if plotnumi == 5:
			ax.annotate("   PAINTER\nPath Choice", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
			ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
				arrowprops=dict(lw=.4,color=painter_color))
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return
		

		ax.annotate("   PAINTER\nPath Choices", (this_blackhole+61, 8), c='forestgreen',fontweight='bold')
		ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole-5,21),
			arrowprops=dict(lw=.4,color=painter_color))
		ax.annotate("",xytext=(this_blackhole+60,13),xy=(this_blackhole+55,43),
			arrowprops=dict(lw=.4,color=painter_color))
		# highlight chosen path
		for i in range(len(chosen_path_times) - 1):
			this_t = chosen_path_times[i][0] - start_t
			next_t = chosen_path_times[i+1][0] - start_t
			this_path = chosen_path_times[i][1]
			intf = name_intf_mapping[this_path]
			data_this_path = latency_over_time[intf]
			t_arr = data_this_path[0,:] -  start_t
			inds = [i for i,_t in enumerate(t_arr) if _t > this_t and _t <= next_t]
			if inds == []: continue
			# inds = inds + [max(inds) + 1]
			lat_arr = data_this_path[1,inds]
			t_arr = t_arr[inds]
			ax.plot(t_arr,lat_arr,color=painter_color,linewidth=10,alpha=.4)
			# ax.plot(t_arr,lat_arr,color=intf_colors[intf],linewidth=10,alpha=.4)
		if plotnumi == 6:
			self.save_fig("system_demonstration_time_series_for_ppt_{}.png".format(plotnumi))
			return

	def save_fig(self, fig_file_name):
		# helper function to save to specific figure directory
		# plt.grid(True)
		plt.savefig(os.path.join('figures', fig_file_name),bbox_inches='tight')
		plt.clf()
		plt.close()

	def find_targs_unicast_anycast_withdrawal_experiment(self):
		ret = read_lats_over_time()
		lats_by_targ = ret['lats_by_targ']
		i_to_popp = ret['i_to_popp']
		i_to_targ = ret['i_to_targ']


		targ_to_i = {targ:i for i,targ in i_to_targ.items()}
		popp_to_i = {popp:i for i,popp in i_to_popp.items()}

		atlanta_popps = list([popp for popp in popp_to_i if popp[0] == 'atlanta'])


		anycast_lats = parse_anycast_latency_file(os.path.join(CACHE_DIR, 'vultr_anycast_latency.csv'))

		valid_targs = []
		for targ, pop in anycast_lats:
			if pop != 'newyork':
				continue
			if np.min(anycast_lats[targ,pop]) > 100: continue
			try:
				lats = lats_by_targ[targ_to_i[targ]]
			except KeyError:
				continue
			new_lats = {}
			for poppi,latsseries in lats.items():
				popp = i_to_popp[poppi]
				new_lats[popp] = np.min([el[1] for el in latsseries])
			## 1 - best provider latency is better than anycast
			## 2 - differences in atlanta latencies by a few ms
			sorted_lats = sorted(new_lats.items(), key = lambda el : el[1])
			if sorted_lats[0][0][0] == 'newyork' and sorted_lats[0][1] < anycast_lats[targ,pop]:
				atlanta_lats = list([new_lats[atlanta_popp] for atlanta_popp in new_lats if atlanta_popp[0] == 'atlanta'])
				if len(atlanta_lats) == 0: continue
				if np.abs(np.min(atlanta_lats) - np.max(atlanta_lats)) > 4:
					valid_targs.append(targ)

		for targ in valid_targs:
			lats = lats_by_targ[targ_to_i[targ]]
			new_lats = {}
			for poppi,latsseries in lats.items():
				popp = i_to_popp[poppi]
				new_lats[popp] = np.min([el[0] for el in latsseries])
			print("{} -- {}".format(targ,new_lats))

		with open('cache/unicast_anycast_withdrawal_targs.csv', 'w') as f:
			for targ in valid_targs:
				f.write(targ + "\n")


if __name__ == "__main__":
	ma = Measurement_Analyzer()
	# ma.compare_with_without_rpki()
	# ma.get_probing_targets()
	# ma.user_prefix_impact()
	# ma.get_probing_targets()
	# ma.plot_lat_over_time()
	# ma.characterize_per_ingress_measurements()
	# ma.prune_per_ingress_measurements()
	# ma.find_targs_unicast_anycast_withdrawal_experiment()
	ma.unicast_anycast_withdrawal_experiment()
