import os
import matplotlib.pyplot as plt, numpy as np,tqdm

from config import *
from helpers import *
from generic_measurement_utils import AS_Utils_Wrapper
from measure_peering import Peering_Pinger



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
		with open(os.path.join(CACHE_DIR, 'possible_pref_targets_by_peer.csv'),'w') as f:
			for peer in peers:
				print("Summarizing {}".format(peer))
				peer = self.parse_asn(peer)
				all_asns = self.get_cc(peer)
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

	def characterize_per_ingress_measurements(self):
		### Look at what we measured to for each ingress, some basic properties
		meas_by_popp = {}
		meas_by_ip = {}
		by_ingress_fn = os.path.join(CACHE_DIR, '{}_ingress_latencies_by_dst_small.csv'.format(
			self.system))
			
		res = parse_ingress_latencies_mp(by_ingress_fn)
		meas_by_popp, meas_by_ip = res['meas_by_popp'], res['meas_by_ip']
		anycast_latencies = {}
		anycast_pop = {}
		for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(
			self.system)),'r'),desc='loading anycast latencies'):
				_,ip,lat,pop = row.strip().split(',')
				if lat == '-1': continue
				anycast_latencies[ip] = np.maximum(float(lat) * 1000,1)
				anycast_pop[ip] = pop

		pp = Peering_Pinger('vultr','measure_prepainter')
		all_peers = [popp[1] for popp in pp.popps]
		meas_peers = [popp[1] for popp in meas_by_popp]

		print("{} targets, {} ingresses with a measurement".format(
			len(meas_by_ip), len(meas_by_popp)))

		no_meas_peers = get_difference(all_peers, meas_peers)
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

	def compare_mine_jc_anycast(self):
		my_anycast_latencies = {}
		for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, '{}_anycast_latency.csv'.format(
			self.system))),'r'):
				t,ip,lat,pop = row.strip().split(',')
				if lat == '-1': continue
				if float(lat) > 2:
					continue
				try:
					my_anycast_latencies[ip,pop].append(np.maximum(float(lat) * 1000,1))
				except KeyError:
					my_anycast_latencies[ip,pop] = [np.maximum(float(lat) * 1000,1)]
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



if __name__ == "__main__":
	ma = Measurement_Analyzer()
	ma.characterize_per_ingress_measurements()
