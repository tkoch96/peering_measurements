from helpers import *
from config import *
from advertisement_experiments import Advertisement_Experiments
import os, itertools, numpy as np, glob, json, re
from subprocess import call

PROPAGATE_TIME = 1#15 * 60 ## time to wait for announcement to propagate
out_fn = lambda n_remove : os.path.join(CACHE_DIR, 'jiangchen_measuring_catchments_{}'.format(n_remove))

all_pops = list(POP_INTERFACE_INFO)

def measure_catchments():
	pops_of_interest = ['london', 'tokyo', 'singapore', 'frankfurt', 'amsterdam', 'saopaulo', 
		'paris', 'miami']


	remove_numbers = [3]#[2,3]
	all_remove_pop_sets = []
	for remove_number in remove_numbers:
		remove_pop_sets = list(itertools.combinations(pops_of_interest, remove_number))
		all_remove_pop_sets = remove_pop_sets + all_remove_pop_sets
	remove_pop_batches = split_seq(all_remove_pop_sets, 
		int(np.ceil(len(all_remove_pop_sets)/6))) # we have 6 prefixes to work with
	for remove_pop_batch in remove_pop_batches:
		focus_pop_batch = list([get_difference(all_pops, remove_pop_set) for \
			remove_pop_set in remove_pop_batch])
		ae = Advertisement_Experiments(system='vultr', mode='measure_catchment', quickinit=False)
		results = ae.run(propagate_time=PROPAGATE_TIME, out_fn=out_fn,
			pop_batch=focus_pop_batch)

def measure_catchments_smaller_nsdi23():
	"""Measures catchments for deployments that remove 2 pops, for specific sets of clients in each deployment."""
	top_dir = os.path.join(CACHE_DIR,'jiangchen_take_out_two_smaller_catchment')
	out_fn = os.path.join(top_dir,'jiangchen_measuring_catchments_remove2_smaller.csv')
	remove_pop_sets = []
	all_client_sets = []
	for row in open(os.path.join(top_dir, 'measurement_config.csv'), 'r'):
		if row.startswith('directory'): continue
		pop1str,pop2str = row.strip().split(',')
		pop1 = re.search("\_leave\_out\_(.+)", pop1str).group(1)
		pop2 = re.search("\_ip\_(.+)\.txt", pop2str).group(1)
		remove_pop_sets.append(set([pop1,pop2]))
		clients = []
		for row in open(os.path.join(top_dir, pop1str,pop2str), 'r'):
			clients.append(row.strip())
		all_client_sets.append(clients)

	inds = list(range(len(remove_pop_sets)))
	batch_sets = split_seq(inds, int(np.ceil(len(inds)/6)))
	remove_pop_batches = list([[remove_pop_sets[i] for i in batchi] for batchi in batch_sets])
	client_batches = list([[all_client_sets[i] for i in batchi] for batchi in batch_sets])

	measured = True
	if not measured:
		ae = Advertisement_Experiments(system='vultr', mode='measure_catchment', quickinit=False)
		for remove_pop_batch,client_batch in zip(remove_pop_batches, client_batches):
			focus_pop_batch = list([get_difference(all_pops, remove_pop_set) for \
				remove_pop_set in remove_pop_batch])
			results = ae.run(propagate_time=PROPAGATE_TIME, out_fn=out_fn,
				pop_batch=focus_pop_batch, targ_batch=client_batch)

	pref_to_uid = {}
	pops_to_uid = {}
	uid_to_meas = {}
	uid = 0
	for row in open(out_fn, 'r'):
		if row.startswith('prefix_to_pop'):
			_,pref,pops = row.strip().split(',')
			pops = tuple(sorted((pops.split('-'))))
			pref_to_uid[pref_to_ip(pref)] = uid
			pops_to_uid[pops] = uid
			uid_to_meas[uid] = {}

			uid += 1
		else:
			pref,dst,pop = row.strip().split(',')
			try:
				uid_to_meas[pref_to_uid[pref]][pop].append(dst)
			except KeyError:
				uid_to_meas[pref_to_uid[pref]][pop] = [dst]
	high_level_out_dir = os.path.join(top_dir, 'formatted_results')
	for row in open(os.path.join(top_dir, 'measurement_config.csv'), 'r'):
		if row.startswith('directory'): continue
		pop1str,pop2str = row.strip().split(',')
		pop1 = re.search("\_leave\_out\_(.+)", pop1str).group(1)
		pop2 = re.search("\_ip\_(.+)\.txt", pop2str).group(1)

		this_dir = os.path.join(high_level_out_dir, '{}-{}'.format(pop1,pop2))
		if not os.path.exists(this_dir):
			call("mkdir {}".format(this_dir), shell=True)

		this_uid = pops_to_uid[tuple(sorted((get_difference(all_pops, [pop1,pop2]))))]
		for pop,dsts in uid_to_meas[this_uid].items():
			with open(os.path.join(this_dir, pop + '.txt'), 'w') as f:
				for dst in dsts:
					f.write(dst + '\n')




def output_catchment_information_nsdi23():
	### output format is withdrawal_set, asn, catchment, population_percent

	n_remove = 2

	parsed_out_fn = os.path.join(CACHE_DIR,'jiangchen_dstip_catchment_output_remove{}.csv'.format(n_remove))
	if not os.path.exists(parsed_out_fn):

		# load population data
		ae = Advertisement_Experiments(system='vultr', mode='measure_catchment', quickinit=True)
		as_to_pop = {}
		rows = csv.reader(open(os.path.join(CACHE_DIR, 'aspop.csv'),'r'))
		for row in rows:
			asn = ae.utils.parse_asn(row[1][2:])
			pop = float(row[6])
			try:
				as_to_pop[asn] += pop
			except KeyError:
				as_to_pop[asn] = pop
		catchment_results, pref_to_uid, uid_to_popset = {}, {}, {}
		popsetuid = 0
		for row in open(out_fn, 'r'):
			if row.startswith('prefix_to_pop'):
				_,pref,popstr = row.strip().split(',')
				pops_included = tuple(sorted(popstr.split('-')))
				pref_to_uid[pref_to_ip(pref)] = popsetuid
				uid_to_popset[popsetuid] = pops_included
				catchment_results[popsetuid] = {}
				popsetuid += 1
			else:
				pref,dst,pop = row.strip().split(',')
				uid = pref_to_uid[pref]
				try:
					catchment_results[uid][dst].append(pop)
				except KeyError:
					catchment_results[uid][dst] = [pop]
			# if np.random.random() > .999999:break

		all_results_by_asn = {}
		results_by_dst = {}
		no_pop_asn, pop_asn, pop_dst = {}, {}, {}
		multipop_dst, all_dst = {}, {}
		for uid in catchment_results:
			this_popset_results = catchment_results[uid]
			results_by_asn = {}
			results_by_dst[uid] = {}
			for dst, pops in this_popset_results.items():
				all_dst[dst] = None
				if len(set(pops)) > 1:
					multipop_dst[dst] = None
					continue
				pop = pops[0]
				asn = ae.utils.parse_asn(dst)
				if asn is None: continue
				try:
					popln = as_to_pop[asn]
					pop_asn[asn] = None
					pop_dst[dst] = None
				except KeyError:
					no_pop_asn[asn] = None
					continue
				try:
					results_by_asn[asn]
				except KeyError:
					results_by_asn[asn] = {}
				try:
					results_by_asn[asn][pop] += 1
				except KeyError:
					results_by_asn[asn][pop] = 1
				results_by_dst[uid][dst] = pop
			all_results_by_asn[uid] = results_by_asn
		print("No population for {} ASNs, {} ASNs had pop data, {} dsts".format(
			len(no_pop_asn), len(pop_asn), len(pop_dst)))
		print("{} dsts total, {} were multipop".format(len(all_dst), len(multipop_dst)))
		# x = list(no_pop_asn)
		# print(x[0:100])

		with open(os.path.join(CACHE_DIR, 'jiangchen_catchment_output_{}.csv'.format(n_remove)), 'w') as f:
			for uid,popset in uid_to_popset.items():
				f.write("{},{}\n".format(uid,"-".join(popset)))
			for uid,res_set in all_results_by_asn.items():
				for asn,popdst in res_set.items():
					total_v = sum(list(popdst.values()))
					for pop,n in popdst.items():
						pct = round(n/total_v, 4)
						f.write("{},{},{},{}\n".format(uid,asn,pop,pct))
		with open(parsed_out_fn,'w') as f:
			for uid,popset in uid_to_popset.items():
				f.write("{},{}\n".format(uid,"-".join(popset)))
			for uid in results_by_dst:
				for dst,pop in results_by_dst[uid].items():
					f.write("{},{},{}\n".format(uid,dst,pop))
	else:
		### Load them
		uid_to_popset, results_by_dst = {}, {}
		for row in tqdm.tqdm(open(parsed_out_fn, 'r'), desc="Loading preparsed results"):
			fields = row.strip().split(',')
			if len(fields) == 2:
				uid,pops = fields
				pops = pops.split('-')
				uid_to_popset[int(uid)] = pops
				print("UID {}, Excluded {}".format(uid,get_difference(all_pops,pops)))
				results_by_dst[int(uid)] = {}
			else:
				uid,dst,pop = fields
				results_by_dst[int(uid)][dst] = pop

	### Further parsing to help JC
	for fn in glob.glob(os.path.join(CACHE_DIR, 'jiangchen_organizing_results', '*')):
		if 'parsed_results' in fn: continue
		leaveoutone = re.search("leave\_out\_(.+)", fn).group(1)
		for iplistfn in glob.glob(os.path.join(fn, '*')):
			leaveouttwo = re.search("anycast\_ip\_(.+)\.txt", iplistfn).group(1)
			leaveoutpops = set([leaveoutone, leaveouttwo])
			remaining_pops = set(get_difference(all_pops, leaveoutpops))
			try:
				corresponding_uid = [uid for uid,pops in uid_to_popset.items() if set(pops) == remaining_pops][0]
			except IndexError:
				continue
			these_ips = []
			for row in open(iplistfn, 'r'):
				these_ips.append(row.strip())

			pops_str = "-".join(sorted(list(leaveoutpops)))
			outdir = os.path.join(CACHE_DIR, 'jiangchen_organizing_results', 'parsed_results', pops_str)
			if not os.path.exists(outdir):
				call("mkdir {}".format(outdir), shell=True)
			remaining_pops = sorted(list(remaining_pops))
			filenames = list([os.path.join(outdir, pop + ".txt") for pop in remaining_pops])
			filenames.append(os.path.join(outdir,'unknown.txt'))
			filepointers = [open(fn,'w') for fn in filenames]
			pop_to_i = {pop:i for i,pop in enumerate(remaining_pops)}
			for ip in these_ips:
				try:
					pop = results_by_dst[corresponding_uid][ip]
					filepointers[pop_to_i[pop]].write("{}\n".format(ip))
				except KeyError:
					filepointers[-1].write("{}\n".format(ip))

			for f in filepointers: f.close()


def measure_catchments_by_provider_config_nsdi23():
	done_meas = False
	np.random.seed(31415)
	jc_cfg_dir = "jiangchen_nsdi_catchment_configs"
	this_experiment = []
	these_clients = []
	out_dir = os.path.join(CACHE_DIR, jc_cfg_dir, 'results')
	out_fn = os.path.join(out_dir, 'ingress_results.txt')
	fn_pop_to_clients = {}
	for fn in glob.glob(os.path.join(CACHE_DIR, jc_cfg_dir, "*")):
		if ".txt" not in fn: continue
		fn_re = re.search("leave\_out\_(.+)\.(.+)\.txt", fn)
		leaveout = fn_re.group(1)
		anycastip = fn_re.group(2)
		x = json.load(open(fn,'r'))
		for el,poppsdct in x.items():
			single_adv = []
			single_clients = []
			if len(poppsdct) == 1: continue
			for pop,peers in poppsdct.items():
				for peer in peers:
					single_adv.append((pop,peer))
			this_experiment.append(single_adv)
			for row in open(os.path.join(CACHE_DIR, jc_cfg_dir, 
				'anycast_ips_leave_out_{}'.format(leaveout), 'anycast_ip_{}.txt'.format(anycastip)),'r'):
				single_clients.append(row.strip())
			these_clients.append(single_clients)
	if not done_meas:
		ae = Advertisement_Experiments(system='vultr', mode='measure_catchment', quickinit=False)
		ae.conduct_measurements_to_prefix_popps(this_experiment, these_clients, out_fn,
			using_manual_clients=True,propagate_time=PROPAGATE_TIME,logcomplete=False)
	
	results_by_popps = {}
	popps_to_pref = []
	pref_ctr = {}
	for row in open(out_fn,'r'):
		fields = row.strip().split(',')
		if len(fields) == 2:
			pref, popps = fields
				
			try:
				pref_ctr[pref_to_ip(pref)] += 1
			except KeyError:
				pref_ctr[pref_to_ip(pref)] = 1

			popps = set([tuple(popp.split('-')) for popp in popps.split('--')])
			popps_to_pref.append((popps,pref_to_ip(pref), pref_ctr[pref_to_ip(pref)]))
			results_by_popps[pref_to_ip(pref), pref_ctr[pref_to_ip(pref)]] = {}

		else:
			pref,t,ip,pop,_,_ = fields
			try:
				results_by_popps[pref,pref_ctr[pref_to_ip(pref)]][pop].append(ip)
			except KeyError:
				results_by_popps[pref,pref_ctr[pref_to_ip(pref)]][pop] = [ip]

	for fn in glob.glob(os.path.join(CACHE_DIR, jc_cfg_dir, "*")):
		if ".txt" not in fn: continue
		print(fn)
		fn_re = re.search("leave\_out\_(.+)\.(.+)\.txt", fn)
		leaveout = fn_re.group(1)
		anycastip = fn_re.group(2)
		x = json.load(open(fn,'r'))

		for el,poppsdct in x.items():
			this_expt_dir_name = "catchment_confd_leave_out_{}_{}".format(leaveout, el)
			this_full_path = os.path.join(out_dir, 'formatted_results', this_expt_dir_name)
			if not os.path.exists(this_full_path):
				call("mkdir {}".format(this_full_path), shell=True)
			clients = []
			for row in open(os.path.join(CACHE_DIR, jc_cfg_dir, 
				'anycast_ips_leave_out_{}'.format(leaveout), 'anycast_ip_{}.txt'.format(anycastip)),'r'):
				clients.append(row.strip())
			if len(poppsdct) == 1:
				pop = list(poppsdct)[0]
				with open(os.path.join(this_full_path, 'traffic_{}.txt'.format(pop)), 'w') as f:
					for client in clients:
						f.write("{}\n".format(client))
			else:
				popps = set([(pop,peer) for pop in poppsdct for peer in poppsdct[pop]])
				_,pref,n = [el for el in popps_to_pref if el[0] == popps][0]
				_id = (pref,n)
				print("{} -- {}".format(this_expt_dir_name, _id))
				for pop in results_by_popps[_id]:
					with open(os.path.join(this_full_path, 'traffic_{}.txt'.format(pop)), 'w') as f:
						for client in results_by_popps[_id][pop]:
							f.write("{}\n".format(client))	


def measure_catchments_smaller_sigcomm24():
	"""Measures catchments for deployments that remove 3 pops, for specific sets of clients in each deployment."""
	### we measure each of two pop sub-systems, and then top-n failure cases in those sub-systems. bc 20 choose 3 is too high

	top_dir = os.path.join(CACHE_DIR, 'jiangchen_sigcomm2024')
	top_config_dir = os.path.join(top_dir,'jiangchen_sigcomm2024_catchment_configs')
	top_result_dir = os.path.join(top_dir, 'measurement_results')
	n_pops_per_sub2deployment = 5
	out_fn = os.path.join(top_result_dir,'jiangchen_measuring_catchments_remove2-{}_smaller.csv'.format(
		n_pops_per_sub2deployment))
	pop_to_client_sets = {}
	ordered_client_sets = {}
	not_done_yet = pickle.load(open('tmp-not-done.pkl','rb'))#### TEMPORARY. CODE
	for subdeployment_dir in tqdm.tqdm(glob.glob(os.path.join(top_config_dir, "*")), 
		desc="Organizing measurement batches..."):
		i=0
		dirpopre = re.search("anycast\_ips\_lo\_(.+)\_(.+)", subdeployment_dir)
		toppop1 = dirpopre.group(1)
		toppop2 = dirpopre.group(2)
		for row in open(os.path.join(subdeployment_dir, 'pop2capacity_sqrt_ppl.txt'),'r'):
			i+=1
			toppop3,n = row.strip().split('\t')
			if i == 1:
				top_n = float(n)
			if i > n_pops_per_sub2deployment and float(n) / top_n < .4:
				break
			
			remove_pop_set = tuple([toppop1,toppop2,toppop3])
			focus_pop_set = tuple(sorted(get_difference(all_pops, remove_pop_set)))
			try: #### TEMPORARY. CODE
				not_done_yet[focus_pop_set]
			except KeyError:
				continue
			try:
				pop_to_client_sets[focus_pop_set]
			except KeyError:
				pop_to_client_sets[focus_pop_set] = {}
			clients = []
			for row in open(os.path.join(subdeployment_dir, 'anycast_ip_{}.txt'.format(toppop3)), 'r'):
				pop_to_client_sets[focus_pop_set][row.strip()] = None
				clients.append(row.strip())
			ordered_client_sets[remove_pop_set] = clients

	inds = list(range(len(pop_to_client_sets)))
	batch_sets = split_seq(inds, int(np.ceil(len(inds)/6)))
	pop_sets = list(pop_to_client_sets)
	pop_batches = list([[pop_sets[i] for i in batchi] for batchi in batch_sets])
	client_batches = list([[pop_to_client_sets[pops] for pops in pop_batch] for pop_batch in pop_batches])

	measured = False
	if not measured:
		ae = Advertisement_Experiments(system='vultr', mode='measure_catchment', quickinit=False)
		for focus_pop_batch,client_batch in zip(pop_batches, client_batches):
			results = ae.run(propagate_time=PROPAGATE_TIME, out_fn=out_fn,
				pop_batch=focus_pop_batch, targ_batch=client_batch)

	pref_to_uid = {}
	uid_to_pops = {}
	uid_to_pref = {}
	pops_to_meas = {}
	uid_to_meas = {}
	uid = 0
	for row in tqdm.tqdm(open(out_fn, 'r'), desc="Parsing measurements..."):
		if row.startswith('prefix_to_pop'):
			_,pref,pops = row.strip().split(',')
			pops = tuple(sorted((pops.split('-'))))
			
			uid += 1
			pref_to_uid[pref_to_ip(pref)] = uid
			uid_to_pref[uid] = pref
			uid_to_pops[uid] = pops
		else:
			pref,dst,pop = row.strip().split(',')
			uid = pref_to_uid[pref]
			pops = uid_to_pops[uid]
			try:
				pops_to_meas[pops]
			except KeyError:
				pops_to_meas[pops] = {}
			try:
				pops_to_meas[pops][pop].append(dst)
			except KeyError:
				pops_to_meas[pops][pop] = [dst]
			try:
				uid_to_meas[uid]
			except KeyError:
				uid_to_meas[uid] = {}
			try:
				uid_to_meas[uid][pop].append(dst)
			except KeyError:
				uid_to_meas[uid][pop] = [dst]
	# tups = {}#### TEMPORARY. CODE
	# for uid in uid_to_pops:
	# 	try:
	# 		uid_to_meas[uid]
	# 	except KeyError:
	# 		# no measurements, weird!
	# 		print("UID {} has no meas, prefix {}".format(uid, uid_to_pref.get(uid)))
	# 		tups[uid_to_pops[uid]] = None
	# pickle.dump(tups,open('tmp-not-done.pkl','wb'))
	# exit(0)
	
	high_level_out_dir = os.path.join(top_dir, 'formatted_results')
	out_file_structure = {}
	for subdeployment_dir in tqdm.tqdm(glob.glob(os.path.join(top_config_dir, "*")),desc="Organizing outputs..."):
		i=0
		dirpopre = re.search("anycast\_ips\_lo\_(.+)\_(.+)", subdeployment_dir)
		toppop1 = dirpopre.group(1)
		toppop2 = dirpopre.group(2)
		for row in open(os.path.join(subdeployment_dir, 'pop2capacity_sqrt_ppl.txt'),'r'):
			i+=1
			if i > n_pops_per_sub2deployment:
				break
			toppop3,n = row.strip().split('\t')

			failing_pops = tuple([toppop1,toppop2,toppop3])
			used_pops = tuple(sorted((get_difference(all_pops, failing_pops))))
			try:
				meas_exists = any(len(dsts) > 0 for dsts in pops_to_meas[used_pops].values())
			except KeyError:
				continue
			if not meas_exists: continue

			for pop,dsts in pops_to_meas[used_pops].items():
				dsts = get_intersection(ordered_client_sets[failing_pops], dsts)
				if len(dsts) == 0: continue
				try:
					out_file_structure[failing_pops]
				except KeyError:
					out_file_structure[failing_pops] = {}
				for dst in dsts:
					try:
						out_file_structure[failing_pops][pop][dst] = None
					except KeyError:
						out_file_structure[failing_pops][pop] = {dst: None}

	for toppop1,toppop2,toppop3 in out_file_structure:
		this_dir = os.path.join(high_level_out_dir, 'anycast_lb_lo_{}_{}_fail_{}'.format(toppop1,toppop2,toppop3))
		if not os.path.exists(this_dir):
			call("mkdir {}".format(this_dir), shell=True)
			
		for pop,dsts in out_file_structure[toppop1,toppop2,toppop3].items():
			thisoutfn = os.path.join(this_dir, "traffic_" + pop + '.txt')
			with open(thisoutfn, 'a') as f:
				for dst in dsts:
					f.write(dst + '\n')

def check_doneness_catchment_sigcomm24():
	"""Assesses whether we measured catchments correctly, plots how many we're missing."""
	top_dir = os.path.join(CACHE_DIR, 'jiangchen_sigcomm2024')
	top_config_dir = os.path.join(top_dir,'jiangchen_sigcomm2024_catchment_configs')
	top_result_dir = os.path.join(top_dir, 'measurement_results')
	n_pops_per_sub2deployment = 5
	out_fn = os.path.join(top_result_dir,'jiangchen_measuring_catchments_remove2-{}_smaller.csv'.format(
		n_pops_per_sub2deployment))
	pop_to_client_sets = {}
	ordered_client_sets = {}
	pcts = []
	n_multi_clients = []
	for subdeployment_dir in tqdm.tqdm(glob.glob(os.path.join(top_config_dir, "*")), 
		desc="Organizing measurement batches..."):
		i=0
		dirpopre = re.search("anycast\_ips\_lo\_(.+)\_(.+)", subdeployment_dir)
		toppop1 = dirpopre.group(1)
		toppop2 = dirpopre.group(2)
		for row in open(os.path.join(subdeployment_dir, 'pop2capacity_sqrt_ppl.txt'),'r'):
			i+=1
			toppop3,n = row.strip().split('\t')
			remove_pop_set = tuple([toppop1,toppop2,toppop3])
			focus_pop_set = tuple(sorted(get_difference(all_pops, remove_pop_set)))
			if i == 1:
				top_n = float(n)
			if i > n_pops_per_sub2deployment and float(n) / top_n < .4:
				break
			
			expected_clients = []
			for row in open(os.path.join(subdeployment_dir, 'anycast_ip_{}.txt'.format(toppop3)), 'r'):
				expected_clients.append(row.strip())

			actual_clients = {}
			for fn in glob.glob(os.path.join(top_dir, 'formatted_results',
				'anycast_lb_lo_{}_{}_fail_{}'.format(toppop1,toppop2,toppop3), "*")):
				seen_this_f = {}
				for row in open(fn,'r'):
					try:
						seen_this_f[row.strip()]
						continue
					except KeyError:
						pass
					seen_this_f[row.strip()] = None
					try:
						actual_clients[row.strip()] += 1
					except KeyError:
						actual_clients[row.strip()] = 1
			if len(actual_clients) > 0:
				x = list([ip for ip,v in actual_clients.items() if v > 1])
				n_multi_clients.append(len(x)/len(actual_clients))
			else:
				print("{} {} {}".format(toppop1,toppop2,toppop3))
				print("Expected {} clients, got 0".format(len(expected_clients)))
				print("-".join(focus_pop_set))
				exit(0)

			# print("{} in expected not in actual".format(len(get_difference(expected_clients,actual_clients))))
			if len(get_difference(actual_clients,expected_clients)) > 0: # this should never happen
				print("{} {} {}".format(toppop1,toppop2,toppop3))
				print("{} in actual not in in expected".format(len(get_difference(actual_clients,expected_clients))))
				exit(0)
			pcts.append(len(get_intersection(expected_clients,actual_clients)) / len(expected_clients))
			# print("{} in both".format(len(get_intersection(expected_clients,actual_clients))))


	import matplotlib.pyplot as plt
	x,cdf_x = get_cdf_xy(pcts)
	plt.plot(x,cdf_x)
	plt.xlabel("Percent targets found catchments")
	plt.ylabel("CDF of deployments")
	plt.grid(True)
	plt.savefig('figures/jiangchen_sigcomm24_doneness.pdf')
	plt.clf(); plt.close()

	x,cdf_x = get_cdf_xy(n_multi_clients)
	plt.plot(x,cdf_x)
	plt.xlabel("Percent targets per scenario that had multi catchment")
	plt.ylabel("CDF of deployments")
	plt.grid(True)
	plt.savefig('figures/jiangchen_sigcomm24_multitargs.pdf')




if __name__ == "__main__":
	measure_catchments_smaller_sigcomm24()
	check_doneness_catchment_sigcomm24()
	# measure_catchments()
	# output_catchment_information()
	# measure_catchments_by_provider_config()
	# measure_catchments_smaller()