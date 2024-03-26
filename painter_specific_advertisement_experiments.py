import os, re, socket, csv, time, json, numpy as np, pickle, geopy.distance, copy, glob, tqdm
from config import *
from helpers import *
from subprocess import call, check_output
import subprocess

from advertisement_experiments import Advertisement_Experiments
from pinger_wrapper import Pinger_Wrapper

from delete_table_rules import delete_table_rules
delete_table_rules()

np.random.seed(31415)


class Painter_Specific_Advertisement_Experiments(Advertisement_Experiments):
	def __init__(self, system, mode, **kwargs):
		self.system = system

		### modes for which we don't need to load all the data
		qinit_modes	= ['anycast_unicast_withdrawal_experiment','diagnose_negative_latency']
		if mode in qinit_modes:
			kwargs['quickinit'] = True

		kwargs['childoverride'] = True
		super().__init__(system,mode,**kwargs)
		self.run = {
			'diagnose_negative_latency': self.diagnose_negative_latency,
			'simple_anycast_test': self.simple_anycast_test,
			'measure_prepainter': self.measure_prepainter, # measures performances to calculate painter
			'conduct_painter': self.conduct_painter, # conducts painter-calculated advertisement
			'conduct_oneperpop': self.conduct_oneperpop, # conducts one advertisement per pop
			'conduct_oneperpop_reuse': self.conduct_oneperpop_reuse, # conducts one advertisement per pop, but reuses far away
			'find_needed_pingable_targets': self.find_needed_pingable_targets, # pings everything to see if we can find responsive addresses
			'anycast_unicast_withdrawal_experiment': self.anycast_unicast_withdrawal_experiment,
		}[mode]

	def measure_prepainter(self):
		""" Conducts anycast measurements and per-ingress measurements, for input into a PAINTER calculation."""

		popp_lat_fn = os.path.join(CACHE_DIR, "{}_ingress_latencies_by_dst.csv".format(self.system))
		meas_by_popp, meas_by_ip = self.load_per_popp_meas(popp_lat_fn, exclude_providers=True)
		meas_peers = [popp[1] for popp in meas_by_popp]
		all_peers = [popp[1] for popp in self.popps]

		already_completed_popps = [tuple(row.strip().split(',')) for row in open(self.already_completed_popps_fn,'r')]
		need_meas_peers = get_difference(all_peers, meas_peers)
		need_meas_peers = get_difference(all_peers, list([popp[1] for popp in already_completed_popps]))
		print("Still need meas for {} peers".format(len(need_meas_peers)))
		if len(need_meas_peers) > 0 and False:
			### TODO -- this doesn't really work
			# from analyze_measurements import Measurement_Analyzer
			# ma = Measurement_Analyzer()
			# ma.summarize_need_meas(need_meas_peers)
			# self.find_needed_pingable_targets()
			self.check_construct_client_to_peer_mapping(forceparse=True) # recompute customer cone
		else:
			self.check_construct_client_to_peer_mapping()
		
		## Given we've found pingable targets, measure anycast
		every_client_of_interest = self.check_measure_anycast()

		### Next, get latency from all users to all peers
		prefix_popps = self.get_advertisements_prioritized(exclude_providers=True)

		## all clients that respond to our probes
		have_anycast = [dst for dst,lat in self.anycast_latencies.items() if lat != -1]
		every_client_of_interest = get_intersection(have_anycast, every_client_of_interest)
		if len(prefix_popps) > 0:
			### Conduct measurements and save latencies
			self.conduct_measurements_to_prefix_popps(prefix_popps, every_client_of_interest, popp_lat_fn)
		
		## pick clients who have a measurement to at least one non-provider
		limited_every_client_of_interest = self.limit_to_interesting_clients(every_client_of_interest)
		print("{} interesting clients. Getting latency to all the providers for these clients".format(
			len(limited_every_client_of_interest)))
		
		## Get provider latency
		prefix_popps = self.get_advertisements_prioritized(exclude_providers=False)
		# free up memory
		del every_client_of_interest
		del meas_by_popp
		del meas_by_ip
		del have_anycast

		self.conduct_measurements_to_prefix_popps(prefix_popps, limited_every_client_of_interest, popp_lat_fn,
			only_providers=True, propagate_time=20)#20*60)

	def conduct_painter(self):

		# load advertisement
		# for prefix in budget (rate-limited)
		# popp in advertisemnet
		# conduct the advertisement
		# measure from every user
		# save perfs
		self.check_construct_client_to_peer_mapping()

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

		ranked_advs_fn = os.path.join(CACHE_DIR, 'ranked_advs_vultr.pkl')
		all_ranked_advs = pickle.load(open(ranked_advs_fn,'rb'))
		print(all_ranked_advs.keys())
		MI = self.maximum_inflation
		print("Conductng advertisement solution for MI = {} km".format(int(MI)))
		time.sleep(5)
		adv_soln = all_ranked_advs['all_time-{}-painter_v7-dont_ignore_anycast-vultr'.format(int(MI))]['painter_v7']

		advs_by_budget = get_advs_by_budget(adv_soln)
		print(advs_by_budget)

		popp_lat_fn = os.path.join(CACHE_DIR, "{}_adv_latencies.csv".format(self.system))
		anycast_latencies = self.load_anycast_latencies()
		# every_client_of_interest = [dst for dst,lat in anycast_latencies.items() if lat != -1]
		every_client_of_interest = self.get_reachable_clients(limit=True)
		every_client_of_interest = self.get_condensed_targets(every_client_of_interest)

		budgets = sorted(list(advs_by_budget))
		n_prefs = len(self.available_prefixes)
		n_adv_rounds = int(np.ceil(len(budgets) / n_prefs))
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		pw.n_rounds = 7

		advs_by_budget = [advs_by_budget[budget] for budget in budgets]
		for adv_round_i in range(n_adv_rounds):
			try:
				adv_sets = advs_by_budget[n_prefs * adv_round_i: n_prefs * (adv_round_i+1)]
			except IndexError:
				adv_sets = advs_by_budget[n_prefs * adv_round_i:]
			if sum(len(adv) for adv in adv_sets) == 0: continue
			self.measure_vpn_lats()
			srcs = []
			pref_set = []
			popps_set = []
			pops_set = {}
			adv_set_to_taps = {}
			src_to_budget = {}
			for i, adv_set in enumerate(adv_sets):
				pref = self.get_most_viable_prefix()
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
				src_to_budget[measurement_src_ip] = budgets[n_prefs * adv_round_i + i]
				popps = [(pop,peer) for pop,peer in adv_set]
				popps = [(pop,self.convert_org_to_peer(peer,pop)) for pop,peer in popps]
				popps_set.append(popps)
				self.advertise_to_popps(popps, pref)

				pops = list(set([popp[0] for popp in popps]))
				pops_set[i] = pops
				adv_set_to_taps[i] = [self.pop_to_intf[pop] for pop in pops]
			max_n_pops = max(len(v) for v in adv_set_to_taps.values())	
			for pop_iter in range(max_n_pops):
				srcs_set = []
				taps_set = []
				this_pops_set = []
				clients_set = []
				for adv_set_i in range(len(adv_sets)):
					try:
						# different prefixes have different numbers of PoPs involved
						# measurements are done per PoP, so this is a bit awkward
						taps_set.append(adv_set_to_taps[adv_set_i][pop_iter])
						srcs_set.append(srcs[adv_set_i])
						this_pop = pops_set[adv_set_i][pop_iter]
						close_clients = self.pop_to_close_clients(this_pop)
						clients_set.append(get_intersection(every_client_of_interest, close_clients))
						this_pops_set.append(this_pop)
					except IndexError:
						pass
				print("PoP iter {}".format(pop_iter))
				print(srcs_set)
				print(taps_set)
				print(this_pops_set)
				print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))
				lats_results = pw.run(srcs_set, taps_set, clients_set)
				for src,pop in zip(srcs_set, this_pops_set):
					budget = src_to_budget[src]
					with open(popp_lat_fn,'a') as f:
						for client_dst, meas in lats_results[src].items():
							rtts = []
							for m in meas:
								startpop = m.get('startpop')
								endpop = m.get('endpop')
								rtt = m['rtt']
								if rtt is not None:
									f.write("{},{},{},{},{}\n".format(budget,client_dst,
										startpop,endpop,rtt - self.pop_vpn_lats[pop]))
								else:
									f.write("{},{},{},{},{}\n".format(budget,client_dst,
										startpop,endpop,None))
			with open(popp_lat_fn,'a') as f:
				for src,popps in zip(srcs, popps_set):
					budget = src_to_budget[src]
					# record which popps we're talking about
					popps_str = ",".join(["-".join(popp) for popp in popps])
					f.write("newadv,{},{}\n".format(budget, popps_str))
			for pref, popps in zip(pref_set, popps_set):
				for pop in set([popp[0] for popp in popps]):
					self.withdraw_from_pop(pop, pref)

	def conduct_oneperpop(self):
		"""Conducts advertisement with one prefix per PoP, assesses latency."""
		self.check_construct_client_to_peer_mapping()
		
		lat_fn = os.path.join(CACHE_DIR, "{}_oneperpop_adv_latencies.csv".format(self.system))
		anycast_latencies = self.load_anycast_latencies()
		every_client_of_interest = self.get_reachable_clients()
		every_client_of_interest = self.get_condensed_targets(every_client_of_interest)

		n_prefs = len(self.available_prefixes)
		pops = list(self.pops)
		n_adv_rounds = int(np.ceil(len(pops) / n_prefs))
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		pw.n_rounds = 4

		for adv_round_i in range(n_adv_rounds):
			try:
				these_pops = pops[n_prefs * adv_round_i: n_prefs * (adv_round_i+1)]
			except IndexError:
				these_pops = pops[n_prefs * adv_round_i:] # the rest
			self.measure_vpn_lats()
			srcs = []
			pref_set = []
			taps_set = []
			clients_set = []
			for i, pop in enumerate(these_pops):
				pref = self.get_most_viable_prefix()
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
				self.advertise_to_pop(pop, pref)
				taps_set.append(self.pop_to_intf[pop])
				clients_set.append(every_client_of_interest, close_clients)

			print("PoP iter {}".format(adv_round_i))
			print(srcs)
			print(taps_set)
			print(these_pops)
			print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))
			lats_results = pw.run(srcs, taps_set, clients_set)
			for src,pop in zip(srcs, these_pops):
				with open(lat_fn,'a') as f:
					for client_dst, meas in lats_results[src].items():
						rtts = []
						for m in meas:
							startpop = m.get('startpop')
							endpop = m.get('endpop')
							rtt = m['rtt']
							if rtt is not None:
								f.write("{},{},{},{}\n".format(client_dst,
									startpop,endpop,rtt - self.pop_vpn_lats[pop]))
							else:
								f.write("{},{},{},{}\n".format(client_dst,
									startpop,endpop,None))
			for pop,pref in zip(these_pops,pref_set):
				self.withdraw_from_pop(pop, pref)

	def conduct_oneperpop_reuse(self):
		"""Conducts advertisement with one prefix per PoP with reuse across distant PoPs, assesses latency."""
		self.check_construct_client_to_peer_mapping()

		MI = int(self.maximum_inflation)
		lat_fn = os.path.join(CACHE_DIR, "{}_oneperpop_reuse_{}_adv_latencies.csv".format(self.system, MI))
		anycast_latencies = self.load_anycast_latencies()
		every_client_of_interest = self.get_reachable_clients()
		every_client_of_interest = self.get_condensed_targets(every_client_of_interest)

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

		ranked_advs_fn = os.path.join(CACHE_DIR, 'ranked_advs_vultr.pkl')
		all_ranked_advs = pickle.load(open(ranked_advs_fn,'rb'))
		print(all_ranked_advs.keys())
		MI = self.maximum_inflation
		print("Conductng one per PoP with reuse for MI = {} km".format(int(MI)))
		time.sleep(5)
		adv_soln = all_ranked_advs['all_time-{}-hybridcast_abstract_with_reuse-dont_ignore_anycast-vultr'.format(int(MI))]['hybridcast_abstract_with_reuse']

		### advs by budget is per popp, convert to per pop
		advs_by_budget = get_advs_by_budget(adv_soln)
		pops_by_budget = {b:list(set([el[0] for el in advs_by_budget[b]])) for b in advs_by_budget}
		pops_by_budget = [pops_by_budget[b] for b in sorted(list(pops_by_budget)) if len(pops_by_budget[b]) > 0]
		print(pops_by_budget)

		n_prefs = len(self.available_prefixes)
		n_adv_rounds = int(np.ceil(len(pops_by_budget) / n_prefs))
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		pw.n_rounds = 4

		for adv_round_i in range(n_adv_rounds):
			try:
				these_pop_sets = pops_by_budget[n_prefs * adv_round_i: n_prefs * (adv_round_i+1)]
			except IndexError:
				these_pop_sets = pops_by_budget[n_prefs * adv_round_i:] # the rest
			self.measure_vpn_lats()
			srcs = []
			pref_set = []
			adv_set_to_taps = {}
			for i, pop_set in enumerate(these_pop_sets):
				pref = self.get_most_viable_prefix()
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
				for popi, pop in enumerate(pop_set):
					self.advertise_to_pop(pop, pref, override_rfd=(popi>0))
				adv_set_to_taps[i] = [self.pop_to_intf[pop] for pop in pop_set]

			max_n_pops = max(len(v) for v in adv_set_to_taps.values())	
			for pop_iter in range(max_n_pops):
				srcs_set = []
				taps_set = []
				this_msmt_iter_pops_set = []
				clients_set = []
				for adv_set_i in range(len(these_pop_sets)):
					try:
						# different prefixes have different numbers of PoPs involved
						# measurements are done per PoP, so this is a bit awkward
						taps_set.append(adv_set_to_taps[adv_set_i][pop_iter])
						srcs_set.append(srcs[adv_set_i])
						this_pop = these_pop_sets[adv_set_i][pop_iter]
						clients_set.append(every_client_of_interest, close_clients)
						this_msmt_iter_pops_set.append(this_pop)
					except IndexError:
						pass
				print("PoP iter {}".format(pop_iter))
				print(srcs_set)
				print(taps_set)
				print(this_msmt_iter_pops_set)
				print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))
				lats_results = pw.run(srcs_set, taps_set, clients_set)
				for src,pop in zip(srcs_set, this_msmt_iter_pops_set):
					with open(lat_fn,'a') as f:
						for client_dst, meas in lats_results[src].items():
							rtts = []
							for m in meas:
								startpop = m.get('startpop')
								endpop = m.get('endpop')
								rtt = m['rtt']
								if rtt is not None:
									f.write("{},{},{},{}\n".format(client_dst,
										startpop,endpop,rtt - self.pop_vpn_lats[pop]))
								else:
									f.write("{},{},{},{}\n".format(client_dst,
										startpop,endpop,None))
			for pop_set,pref in zip(these_pop_sets,pref_set):
				for pop in pop_set:
					self.withdraw_from_pop(pop, pref)


	def anycast_unicast_withdrawal_experiment(self):
		### MAKE SURE CONFIG IS GO, BGP ON, NOTHING ADVERTISED

		## set up 4 forward rules for 2 pops, 4 prefixes
		## 1: unicast at pop A
		## 2: anycast at PoPs A and B
		## 3: unicast at PoP B on anycast route
		## 4: unicast at PoP B off anycast route, higher than anycast

		dsts = []
		for row in open(os.path.join(CACHE_DIR, 'unicast_anycast_withdrawal_targs.csv'),'r'):
			dsts.append(row.strip())
		# dsts = ['76.8.178.179','104.255.177.1','23.226.164.21','216.235.230.1',
		# '64.246.232.1','188.227.58.201','64.222.154.1']
		dsts = ['138.124.187.148']
		dst_sets = [dsts for _ in range(7)]
		## adv to corresponding folks at new york and atlanta
		## 1 -> new york,3257
		## 2 -> all new york and atlanta
		## 3 -> atlanta, 3257
		## 4 -> atlanta, 1299
		## call run on [srcs, taps]
		## withdraw all from newyork at some point in time
		## measure latency always

		prefs = []

		pref = '184.164.238.0/24'
		self.advertise_to_popps([('newyork','2914')],pref)
		prefs.append(pref)

		pref = '184.164.239.0/24'
		self.advertise_to_pops(['newyork','atlanta'],pref)
		prefs.append(pref)
	
		pref = '184.164.240.0/24'
		self.advertise_to_popps([('atlanta','3257')],pref)		
		prefs.append(pref)

		pref = '184.164.241.0/24'
		self.advertise_to_popps([('newyork','3257')],pref)		
		prefs.append(pref)

		pref = '184.164.242.0/24'
		self.advertise_to_popps([('atlanta','1299')],pref)
		prefs.append(pref)

		pref = '184.164.243.0/24'
		self.advertise_to_popps([('atlanta','174')],pref)		
		prefs.append(pref)

		taps = list([self.pop_to_intf[pop] for pop in ['newyork', 'newyork', 'atlanta', 'newyork', 'atlanta', 'atlanta']])
		srcs = list([pref_to_ip(pref) for pref in prefs])

		## add a secondary address from which to measure backup anycast path
		taps.append(self.pop_to_intf['atlanta'])
		srcs.append('184.164.239.2')

		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		pw.n_rounds = 1500*20
		print("Starting measurements")
		addrs = []
		lats_results = pw.run(srcs, taps, dst_sets, sleep_period=.01)
		pickle.dump(lats_results, open(os.path.join(CACHE_DIR, 
			'unicast_anycast_experiment_results.pkl'),'wb'))

		print("Withdrawing prefixes...")
		self.withdraw_from_pop('newyork',prefs[0])
		self.withdraw_from_pop('atlanta',prefs[1])
		self.withdraw_from_pop('atlanta',prefs[2])
		self.withdraw_from_pop('newyork',prefs[3])
		self.withdraw_from_pop('atlanta',prefs[4])
		self.withdraw_from_pop('atlanta',prefs[5])
		print("Done withdrawing prefixes.")
		exit(0)

	def diagnose_negative_latency(self):
		popp_lat_fn = os.path.join(CACHE_DIR, "{}_ingress_latencies_by_dst.csv".format(self.system))
		meas_by_popp, meas_by_ip = self.load_per_popp_meas(popp_lat_fn)
		bad_ip_popps = []
		for ip in meas_by_ip:
			for popp, lat in meas_by_ip[ip].items():
				if lat < 0:
					bad_ip_popps.append((ip,popp,lat))
		n_bad_count = {'pop':{}, 'popp': {}, 'ip': {}}
		for ip,popp,lat in bad_ip_popps:
			pop,peer = popp
			try:
				n_bad_count['pop'][pop] += lat
			except KeyError:
				n_bad_count['pop'][pop] = lat
			try:
				n_bad_count['popp'][popp] += lat
			except KeyError:
				n_bad_count['popp'][popp] = lat
			try:
				n_bad_count['ip'][ip] += lat
			except KeyError:
				n_bad_count['ip'][ip] = lat
		for k,vs in n_bad_count.items():
			sorted_vs = sorted(vs.items(), key = lambda el : el[1])
			# print("{} --- {}".format(k,sorted_vs[0:100]))
		popp_of_interest = ('tokyo', '13335')
		clients_of_interest = [list(set(ip for ip,popp,lat in bad_ip_popps if popp == popp_of_interest))]
		
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		popps = [popp_of_interest]
		self.measure_vpn_lats()
		exit(0)
		srcs = []
		adv_set_to_taps = {}
		pref = '184.164.239.0/24'#self.get_most_viable_prefix()
		measurement_src_ip = pref_to_ip(pref)
		srcs.append(measurement_src_ip)

		pops = list(set([popp[0] for popp in popps])) ### hmmmm
		taps_set = list([self.pop_to_intf[pop] for pop in pops])
		# self.advertise_to_popps(popps, pref)
		lats_results = pw.run(srcs, taps_set, clients_of_interest)
		pickle.dump(lats_results, open('tmp.pkl','wb'))


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="measure_prepainter",
		choices=["simple_anycast_test","measure_anycast", 'measure_prepainter', 
		'conduct_painter', 'conduct_oneperpop',
		'conduct_oneperpop_reuse', 'find_needed_pingable_targets',
		'latency_over_time_casestudy', 'find_yunfan_targets',
		'anycast_unicast_withdrawal_experiment',
		'diagnose_negative_latency'])
	parser.add_argument('--system', required=True, 
		choices=['peering','vultr'],help="Are you using PEERING or VULTR?")
	parser.add_argument('--maximum_inflation', required=False,
		help='Maximum inflation to use for painter conduction', default=3000, type=int)
	args = parser.parse_args()

	ae = Painter_Specific_Advertisement_Experiments(args.system,args.mode,maximum_inflation=float(args.maximum_inflation),
		quickinit=False)
	ae.run()
