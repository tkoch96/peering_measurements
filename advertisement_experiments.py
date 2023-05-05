import os, re, socket, csv, time, json, numpy as np, pickle, geopy.distance, copy, glob, tqdm
from config import *
from helpers import *
from subprocess import call, check_output
import subprocess

from deployment_measure_wrapper import Deployment_Measure_Wrapper
from pinger_wrapper import Pinger_Wrapper

from delete_table_rules import delete_table_rules
delete_table_rules()

np.random.seed(31415)


class Advertisement_Experiments(Deployment_Measure_Wrapper):
	def __init__(self, system, mode, **kwargs):
		self.system = system
		super().__init__(**kwargs)
		self.run = {
			'measure_prepainter': self.measure_prepainter, # measures performances to calculate painter
			'conduct_painter': self.conduct_painter, # conducts painter-calculated advertisement
			'conduct_oneperpop': self.conduct_oneperpop, # conducts one advertisement per pop
			'conduct_oneperpop_reuse': self.conduct_oneperpop_reuse, # conducts one advertisement per pop, but reuses far away
			'find_needed_pingable_targets': self.find_needed_pingable_targets, # pings everything to see if we can find responsive addresses
			'latency_over_time_casestudy': self.latency_over_time_casestudy, # measure latency to different targets over long period of time
		}[mode]

	def check_measure_anycast(self):
		self.anycast_latencies = self.load_anycast_latencies()
		every_client_of_interest = list(set([client for popp in self.popp_to_clients for client
			in self.popp_to_clients[popp]]))
		print("Every client of interest includes {} addresses".format(len(every_client_of_interest)))
		still_need_anycast = get_difference(every_client_of_interest, self.anycast_latencies)
		print("Could get anycast latency for {} pct of addresses".format(
			len(still_need_anycast) / len(every_client_of_interest) * 100.0))
		if len(still_need_anycast) / len(every_client_of_interest) > 1:
			print("GETTING ANYCAST LATENCY!!")
			self.measure_vpn_lats()
			### First, get anycast pop
			client_to_pop = {client_ntwrk:pop for pop,ntwrks in self.pop_to_clients.items()\
				for client_ntwrk in ntwrks}
			in_file = []
			for row in open(os.path.join(DATA_DIR, 'client_to_pop.csv'), 'r'):
				in_file.append(row.strip().split(',')[0])
			have_pop = {client:None for client in client_to_pop}
			still_need_pop = get_difference(still_need_anycast, client_to_pop)
			still_need_pop = get_difference(still_need_pop, in_file)
			print("getting client to pop ")
			
			dsts = get_difference(still_need_pop, have_pop)
			need_catchment_measure = len(dsts)>0

			if need_catchment_measure:
				print("MEASURING CATCHMENT for {} dsts".format(len(dsts)))
				time.sleep(10) # give the user a chance to not
				pref = self.announce_anycast()
				print("Waiting for anycast announcement to propagate.")
				time.sleep(120)
				measurement_src_ip = pref_to_ip(pref)
				pw = Pinger_Wrapper(self.pops, self.pop_to_intf)

				pop_results = {}
				for pop in self.pops:
					dsts = get_difference(still_need_pop, have_pop)
					np.random.shuffle(dsts)
					if len(dsts) / len(still_need_anycast) < .1:
						print("Pretty much done getting pop catchments")
						break
					print("Getting catchments for PoP {}, {} clients".format(pop,len(dsts)))
					lats_results = pw.run([measurement_src_ip], 
						[self.pop_to_intf[pop]], [dsts])[measurement_src_ip]
					print("Lats results has {} dsts".format(len(lats_results)))
					for dst, meas in lats_results.items():
						try:
							pop_results[dst]
						except KeyError:
							pop_results[dst] = []
						for m in meas:
							pop_results[dst].append((m.get('startpop', None), m.get('endpop', None)))
					for dst in pop_results:
						pop_results[dst] = list(set(pop_results[dst]))
					n_good_results = 0
					reasons_bad = {0:0,1:0,2:0}
					with open(self.pop_to_clients_fn, 'a') as f:
						for dst, all_results in pop_results.items():
							try:
								self.pop_to_clients[dst]
								reasons_bad[0] += 1
								continue
							except KeyError:
								pass
							start_to_end = {p:[] for p in self.pops_list}
							for res in all_results:
								if res[0] is not None:
									if res[1] is not None:
										start_to_end[res[0]].append(res[1])
							for p in self.pops_list:
								end_pops = list(set(start_to_end[p]))
								if len(end_pops) == 0:
									reasons_bad[1] += 1
									continue
								elif len(end_pops) == 1:
									have_pop[dst] = None
									n_good_results += 1
									p = end_pops[0]
									f.write("{},{}\n".format(dst,p))
									self.pop_to_clients[p].append(dst)
									break
								else:
									reasons_bad[2] += 1
					print("{} good results".format(n_good_results))
					print(reasons_bad)
				self.withdraw_anycast(pref)

			### Second, get anycast latency
			pops = list(self.pops)
			n_prefs = len(self.available_prefixes)
			n_adv_rounds = int(np.ceil(len(pops) / n_prefs))
			# advertise all prefixes
			for pref in self.available_prefixes:	
				self.announce_anycast(pref)
			time.sleep(120)
			srcs_set = [pref_to_ip(pref) for pref in self.available_prefixes]
			pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
			pw.n_rounds = 7
			pw.rate_limit = 1000 # rate limit by a lot more since we expect tons of responses

			for adv_round_i in range(n_adv_rounds):
				taps_set = []
				clients_set = []
				these_pops = pops[adv_round_i * n_prefs:(adv_round_i + 1) * n_prefs]
				for pop in these_pops:
					print("Getting anycast latency for pop {}".format(pop))
					this_pop_clients = self.pop_to_clients[pop]
					dsts_to_measure_to = get_intersection(still_need_anycast, this_pop_clients)
					print("{} clients for this pop".format(len(dsts_to_measure_to)))
					taps_set.append(self.pop_to_intf[pop])
					clients_set.append(dsts_to_measure_to)
				print(these_pops)
				print(srcs_set)
				print(taps_set)
				print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))
				lats_results = pw.run(srcs_set ,taps_set, clients_set,
						remove_bad_meas=True)
				tnow = int(time.time())
				with open(self.anycast_latency_fn,'a') as f:
					for src in lats_results:
						try:
							pop = [p for s,p in zip(srcs_set, these_pops) if src==s][0]
						except IndexError:
							print("Warning -- no measurements whatsoever for {}".format(src))
							print(srcs_set)
							print(these_pops)
							continue
						minimum_latency = self.pop_vpn_lats[pop]
						for client_dst, meas in lats_results[src].items():
							rtts = []
							for m in meas:
								if m['pop'] != pop or m['rtt'] is None:
									continue
								rtts.append(m['rtt'])
							if len(rtts) > 0:
								rtt = np.min(rtts)
								f.write("{},{},{},{}\n".format(tnow,client_dst,rtt - minimum_latency,pop))
								self.anycast_latencies[client_dst] = rtt
							else:
								rtt = -1	
								f.write("{},{},{},{}\n".format(tnow, client_dst,rtt,pop))
			
			### These just dont work for whatever reason
			still_need_anycast = get_difference(still_need_anycast, list(self.anycast_latencies))
			with open(self.addresses_that_respond_to_ping_fn, 'a') as f:
				for dst in still_need_anycast:
					f.write("{},{}\n".format(dst,0))

			# withdraw all prefixes
			for pref in self.available_prefixes:	
				self.withdraw_anycast(pref)
		return every_client_of_interest

	
	def conduct_measurements_to_prefix_popps(self, prefix_popps, every_client_of_interest, popp_lat_fn, **kwargs):
		n_prefs = len(self.available_prefixes)
		n_adv_rounds = int(np.ceil(len(prefix_popps) / n_prefs))
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)

		# measuring_to_client_ns = []
		# for adv_set in prefix_popps:
		# 	for popp in adv_set:
		# 		these_clients = get_intersection(every_client_of_interest, self.popp_to_clients[popp])
		# 		measuring_to_client_ns.append(len(these_clients))
		# import matplotlib.pyplot as plt
		# x,cdf_x = get_cdf_xy(measuring_to_client_ns,logx=True)
		# plt.semilogx(x,cdf_x)
		# plt.xlabel("Number of Clients We're Measuring To")
		# plt.ylabel("CDF of PoPPs We Need to Measure To")
		# plt.savefig("figures/n_clients_measuring_to_per_popp.pdf")

		only_providers = kwargs.get('only_providers', False)

		for adv_round_i in range(n_adv_rounds):
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
				pref = self.get_most_viable_prefix()
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
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
				adv_set_is = []
				for adv_set_i in range(len(adv_sets)):
					try:
						# different prefixes have different numbers of PoPs involved
						# measurements are done per PoP, so this is a bit awkward
						taps_set.append(adv_set_to_taps[adv_set_i][pop_iter])
						srcs_set.append(srcs[adv_set_i])
						this_pop = pops_set[adv_set_i][pop_iter]
						this_pops_set.append(this_pop)
						if not only_providers:
							these_clients = set()
							for popp in popps_set[adv_set_i]:
								## Get any client who we know can route through these ingresses
								if popp[0] != this_pop: continue
								these_clients = these_clients.union(self.popp_to_clients[popp])
								these_clients = list(get_intersection(these_clients, every_client_of_interest))
						else:
							these_clients = every_client_of_interest
						clients_set.append(these_clients)
						adv_set_is.append(adv_set_i)
					except IndexError:
						pass
				print("PoP iter {}".format(pop_iter))
				print(srcs_set)
				print(taps_set)
				print(this_pops_set)
				print(popps_set)
				print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))

				if CAREFUL:
					exit(0)
				lats_results = pw.run(srcs_set, taps_set, clients_set,
					remove_bad_meas=True)

				t_meas = int(time.time())
				for src,pop,dsts,asi in zip(srcs_set, this_pops_set, clients_set, adv_set_is):
					with open(popp_lat_fn,'a') as f:
						for client_dst in dsts:
							if len(popps_set[asi]) == 1:
								if popps_set[asi][0] in self.provider_popps:
									client_has_path = [popps_set[asi][0]]
								else:
									client_has_path = get_intersection(self.client_to_popps.get(client_dst,[]), popps_set[asi])
							else:
								client_has_path = get_intersection(self.client_to_popps.get(client_dst,[]), popps_set[asi])

							if len(client_has_path) != 1:
								print("WEIRDNESS -- client {} should have one path to {} but has many/none".format(
									client_dst,popps_set[asi]))
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
								f.write("{},{},{},{},{}\n".format(t_meas,client_dst,
									clientpathpop,clientpathpeer,rtt - self.pop_vpn_lats[pop]))
							else:
								### Need to save the information that this client could not reach its
								### intended destination, even though we thought it might have
								with open(os.path.join(CACHE_DIR, "unreachable_dsts", "{}-{}.csv".format(
									clientpathpop,clientpathpeer)),'a') as f2:
									f2.write("{}\n".format(client_dst))
				del lats_results
			for pref, popps in zip(pref_set, popps_set):
				for pop in set([popp[0] for popp in popps]):
					self.withdraw_from_pop(pop, pref)
			with open(self.already_completed_popps_fn, 'a') as f:
				for popps in popps_set:
					for pop,peer in popps:
						f.write("{},{}\n".format(pop,peer))

	def measure_prepainter(self):
		""" Conducts anycast measurements and per-ingress measurements, for input into a PAINTER calculation."""

		popp_lat_fn = os.path.join(CACHE_DIR, "{}_ingress_latencies_by_dst.csv".format(self.system))
		meas_by_popp, meas_by_ip = self.load_per_popp_meas(popp_lat_fn)
		meas_peers = [popp[1] for popp in meas_by_popp]
		all_peers = [popp[1] for popp in self.popps]

		already_completed_popps = [tuple(row.strip().split(',')) for row in open(self.already_completed_popps_fn,'r')]
		need_meas_peers = get_difference(all_peers, meas_peers)
		need_meas_peers = get_difference(all_peers, list([popp[1] for popp in already_completed_popps]))
		if len(need_meas_peers) > 0 and False:
			print("Still need meas for {} peers".format(len(need_meas_peers)))
			from analyze_measurements import Measurement_Analyzer
			ma = Measurement_Analyzer()
			ma.summarize_need_meas(need_meas_peers)
			self.find_needed_pingable_targets()
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
		
		## Get provider latency
		prefix_popps = self.get_advertisements_prioritized(exclude_providers=False)
		# free up memory
		del every_client_of_interest
		del meas_by_popp
		del meas_by_ip
		del have_anycast
		del self.client_to_popps
		del self.popp_to_clients
		del self.popp_to_clientasn
		del self.all_client_addresses_from_msft

		self.conduct_measurements_to_prefix_popps(prefix_popps, limited_every_client_of_interest, popp_lat_fn,
			only_providers=True)

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

	def latency_over_time_casestudy(self):
		targs = {}
		popp_ctr = {}
		for row in open(os.path.join(CACHE_DIR, 'interesting_targets_to_probe.csv'),'r'):
			ip,popps_str = row.strip().split(',')
			popps = popps_str.split("-")
			popps = [tuple(el.split("|")) for el in popps]
			for popp in popps:
				try:
					popp_ctr[popp] += 1
				except KeyError:
					popp_ctr[popp] = 1
			targs[ip] = None
		targs = list(targs)
		n_prefs = len(self.available_prefixes)
		measure_to = [ell[0] for ell in sorted(popp_ctr.items(), key = lambda el : -1 * el[1])[0:n_prefs]]
		print("Measuring to PoPPs : {}".format(measure_to))
		popp_lat_fn = os.path.join(CACHE_DIR, "{}_latency_over_time_case_study.csv".format(self.system))

		prefix_popps = [[measure_to[b]] for b in range(len(measure_to))]
		n_adv_rounds = 1
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		pw.n_rounds = 4
		
		n_meas_rounds = 10000
		measurement_period = 10 # seconds

		### Save space, replace strings with integers
		popp_to_i = {popp:i for i,popp in enumerate(measure_to)}
		with open(popp_lat_fn,'a') as f:
			for popp,i in popp_to_i.items():
				f.write("{},{},{}\n".format(popp[0],popp[1],i))
		targ_to_i = {targ:i for i,targ in enumerate(targs)}
		with open(popp_lat_fn,'a') as f:
			for targ,i in targ_to_i.items():
				f.write("{},{}\n".format(targ,i))
		exit(0)

		for adv_round_i in range(n_adv_rounds):
			adv_sets = prefix_popps[n_prefs * adv_round_i: n_prefs * (adv_round_i+1)]
			for asi,adv_set in enumerate(adv_sets):
				print("Adv set {}:\n {}".format(asi,adv_set))
			srcs = []
			pref_set = []
			popps_set = []
			pops_set = {}
			adv_set_to_taps = {}
			for i, popps in enumerate(adv_sets):
				pref = self.get_most_viable_prefix()
				pref_set.append(pref)
				measurement_src_ip = pref_to_ip(pref)
				srcs.append(measurement_src_ip)
				popps_set.append(popps)
				self.advertise_to_popps(popps, pref)

				pops = list(set([popp[0] for popp in popps]))
				pops_set[i] = pops
				adv_set_to_taps[i] = [self.pop_to_intf[pop] for pop in pops]
			if not CAREFUL:
				print("Waiting for announcement to propagate.")
				time.sleep(60) # wait for announcements to propagate
			max_n_pops = max(len(v) for v in adv_set_to_taps.values())	
			for meas_round_j in range(n_meas_rounds):
				print("Measurement round {}\n".format(meas_round_j))
				ts_meas_round = time.time()

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
							this_pop = pops_set[adv_set_i][pop_iter]
							this_pops_set.append(this_pop)
							these_clients = targs
							clients_set.append(these_clients)
							adv_set_is.append(adv_set_i)
						except IndexError:
							pass
					print("PoP iter {}".format(pop_iter))
					print(srcs_set)
					print(taps_set)
					print(this_pops_set)
					print(popps_set)
					print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))

					if CAREFUL:
						exit(0)
					lats_results = pw.run(srcs_set, taps_set, clients_set,
						remove_bad_meas=True)

					t_meas = int(time.time())
					for src,pop,dsts,asi in zip(srcs_set, this_pops_set, clients_set, adv_set_is):
						with open(popp_lat_fn,'a') as f:
							for client_dst in dsts:
								client_has_path = popps_set[asi][0]
								clientpathpop, clientpathpeer = client_has_path
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
								for rtt in rtts:
									poppi = popp_to_i[clientpathpop,clientpathpeer]
									targi = targ_to_i[client_dst]
									f.write("{},{},{},{}\n".format(t_meas,targi,
										poppi,rtt - self.pop_vpn_lats[pop]))
					del lats_results

				t_elapsed_meas_round = time.time() - ts_meas_round
				if t_elapsed_meas_round < measurement_period:
					print("Sleeping {} s between measurement rounds".format(measurement_period - t_elapsed_meas_round))
					time.sleep(measurement_period - t_elapsed_meas_round)

			for pref, popps in zip(pref_set, popps_set):
				for pop in set([popp[0] for popp in popps]):
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

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="measure_prepainter",
		choices=['measure_prepainter', 'conduct_painter', 'conduct_oneperpop',
		'conduct_oneperpop_reuse', 'find_needed_pingable_targets',
		'latency_over_time_casestudy'])
	parser.add_argument('--system', required=True, 
		choices=['peering','vultr'],help="Are you using PEERING or VULTR?")
	parser.add_argument('--maximum_inflation', required=False,
		help='Maximum inflation to use for painter conduction', default=3000, type=int)
	args = parser.parse_args()

	ae = Advertisement_Experiments(args.system,args.mode,maximum_inflation=float(args.maximum_inflation),
		quickinit=True)
	ae.run()
