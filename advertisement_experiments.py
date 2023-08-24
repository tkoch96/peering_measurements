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

		### modes for which we don't need to load all the data
		qinit_modes	= ['test_new_comms']
		if mode in qinit_modes:
			kwargs['quickinit'] = True
		super().__init__(**kwargs)
		if not kwargs.get('childoverride', False):
			self.run = {
				'simple_anycast_test': self.simple_anycast_test,
				'measure_anycast': self.check_measure_anycast,
				'find_needed_pingable_targets': self.find_needed_pingable_targets, # pings everything to see if we can find responsive addresses
				'latency_over_time_casestudy': self.latency_over_time_casestudy, # measure latency to different targets over long period of time
				'find_yunfan_targets': self.find_yunfan_targets,
				'test_new_comms': self.test_new_comms,
			}[mode]

	def simple_anycast_test(self):
		# dsts = ['8.8.8.8','1.1.1.1','201.191.30.3','189.106.21.13']
		# IWNR: [('167.62.98.92', 'miami'), ('207.245.53.134', 'mumbai'), ('148.245.66.226', 'atlanta'), 
		# ('193.109.112.12', 'amsterdam'), ('166.88.191.39', 'mumbai'), ('85.135.22.137', 'mumbai'), ('95.47.58.4', 'frankfurt'), 
		# ('143.106.212.1', 'mumbai'), ('139.195.194.1', 'singapore'), ('195.234.187.65', 'amsterdam')] 

		pops = list(self.pops)
		n_prefs = len(self.available_prefixes)
		n_adv_rounds = int(np.ceil(len(pops) / n_prefs))
		# advertise all prefixes
		for pref in self.available_prefixes:	
			self.announce_anycast(pref)
		# time.sleep(120)
		srcs_set = [pref_to_ip(pref) for pref in self.available_prefixes]
		pw = Pinger_Wrapper(self.pops, self.pop_to_intf)
		pw.n_rounds = 2
		pw.rate_limit = 1000 # rate limit by a lot more since we expect tons of responses

		for adv_round_i in range(n_adv_rounds):
			taps_set = []
			clients_set = []
			these_pops = pops[adv_round_i * n_prefs:(adv_round_i + 1) * n_prefs]
			for pop in these_pops:
				print("Getting anycast latency for pop {}".format(pop))
				this_pop_clients = self.pop_to_clients[pop]
				taps_set.append(self.pop_to_intf[pop])
				clients_set.append(dsts)
			print(these_pops)
			print(srcs_set)
			print(taps_set)
			print("Client set lens: {}".format([len(dsts) for dsts in clients_set]))
			lats_results = pw.run(srcs_set ,taps_set, clients_set,
					remove_bad_meas=True)
			print(lats_results)
			tnow = int(time.time())
			with open('test_anycast_latency.csv','a') as f:
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
						else:
							rtt = -1	
							f.write("{},{},{},{}\n".format(tnow, client_dst,rtt,pop))

		# withdraw all prefixes
		for pref in self.available_prefixes:	
			self.withdraw_anycast(pref)

	def check_measure_anycast(self, targs=None):
		# self.check_construct_client_to_peer_mapping()
		self.anycast_latencies = self.load_anycast_latencies()
		if targs is None:
			every_client_of_interest = self.get_clients_by_popp('all')
		else:
			every_client_of_interest = targs
		print("Every client of interest includes {} addresses".format(len(every_client_of_interest)))
		still_need_anycast = get_difference(every_client_of_interest, self.anycast_latencies)
		print("Could get anycast latency for {} pct of addresses".format(
			len(still_need_anycast) / len(every_client_of_interest) * 100.0))
		if len(still_need_anycast) / len(every_client_of_interest) > .01:
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
				pw.n_rounds = 2
				pw.rate_limit = 30000

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
			pw.n_rounds = 3
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

	def find_yunfan_targets(self):
		## Find pingable targets in prefixes identified by yunfan
		self.check_load_yunfan_prefixes()
		base_arr = np.arange(2**24)
		done_prefs = {}
		dst_fn = os.path.join(CACHE_DIR, 'responsive_yunfan_dsts.csv')
		for row in open(dst_fn,'r'):
			dst = row.strip()
			done_prefs[self.yunfan_user_pref_tri.get_key(dst + "/32")] = None
		import time
		while False:
			all_rand_hosts_ints = {}
			print("{} done prefs so far".format(len(done_prefs)))
			for yunfan_pref in tqdm.tqdm(self.yunfan_user_pref_tri,
				desc = "finding hosts to ping"):
				try:
					done_prefs[yunfan_pref]
					continue
				except KeyError:
					pass
				pref,preflength = yunfan_pref.split("/")
				if pref == "0.0.0.0": continue
				parts = pref.split(".")
				ipnum = (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
					(int(parts[2]) << 8) + int(parts[3])
				n_addr = 2**(32 - int(preflength)) - 2
				if n_addr <= 0:
					continue
				rand_hosts = ipnum+1+np.random.choice(base_arr[0:n_addr], size=np.minimum(n_addr,5),replace=False)
				all_rand_hosts_ints[yunfan_pref] = rand_hosts
			all_rand_hosts = list(['.'.join([str(ipint >> (i << 3) & 0xFF)
				for i in range(4)[::-1]]) for pref,rand_hosts_group in all_rand_hosts_ints.items() for ipint in rand_hosts_group])
			np.random.shuffle(all_rand_hosts)
			responsive_dsts = check_ping_responsive(all_rand_hosts)
			with open(dst_fn,'a') as f:
				for rd in responsive_dsts:
					f.write("{}\n".format(rd))
					parent_pref = self.yunfan_user_pref_tri.get_key(rd + "/32")
					done_prefs[parent_pref] = None
			time.sleep(5)

		ignore_ases = {self.utils.parse_asn(asn):None for asn in [699,8075,15169,792,37963,36351]}

		yunfan_targs = []
		all_ases = {}
		for row in open(dst_fn,'r'):
			targ = row.strip()
			asn = self.utils.parse_asn(targ)
			try:
				ignore_ases[asn]
				continue
			except KeyError:
				pass
			yunfan_targs.append(targ)
			try:
				all_ases[asn] += 1
			except KeyError:
				all_ases[asn] = 1
		# for asn,v in sorted(all_ases.items(), key = lambda el : -1 * el[1])[0:100]:
		# 	print("{} ({}) -- {} targs".format(asn,self.utils.org_to_as.get(asn,None),v))
		self.check_measure_anycast(targs = yunfan_targs)

	def test_new_comms(self):
		pref = self.get_most_viable_prefix()
		popps = [('newyork','2914'), ('tokyo','13445')]

		self.advertise_to_popps(popps,pref)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="measure_anycast",
		choices=["simple_anycast_test","measure_anycast", 'find_needed_pingable_targets',
		'latency_over_time_casestudy', 'find_yunfan_targets',
		'test_new_comms'])
	parser.add_argument('--system', required=True, 
		choices=['peering','vultr'],help="Are you using PEERING or VULTR?")
	parser.add_argument('--maximum_inflation', required=False,
		help='Maximum inflation to use for painter conduction', default=3000, type=int)
	args = parser.parse_args()

	ae = Advertisement_Experiments(args.system,args.mode,maximum_inflation=float(args.maximum_inflation),
		quickinit=False)
	ae.run()
