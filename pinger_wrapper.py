import socket, time, struct, select, numpy as np, re, datetime, tqdm, multiprocessing, copy
from config import *
import netifaces as ni, os
from subprocess import call, check_output

# PINGER = "sudo /home/tom/pinger/target/release/pinger"
PINGER = "sudo /home/ubuntu/pinger/target/release/pinger"

LOW_MEM = True

pop_to_outfn = lambda pop : os.path.join(TMP_DIR, "tcpdumpout_{}.txt".format(pop))
def parse_ping_results(*args):
	pop, meas_ret, _id_to_meas_i, srcsdict, dstsdict, = args[0]
	n_rounds = len(_id_to_meas_i)
	print("Parsing pop : {}".format(pop))
	for row in open(pop_to_outfn(pop), 'r'):
		# parse the measurements
		try:
			if "request" in row:
				rowre = re.search("(.+) IP (.+) \> (.+)\: ICMP echo request\, id (.+)\, seq", row)
				t,src,dst,_id = rowre.group(1), rowre.group(2), rowre.group(3),rowre.group(4).strip()
				which = 'start'
			elif "reply" in row: # ping reply to us
				rowre = re.search("(.+) IP (.+) \> (.+)\: ICMP echo reply\, id (.+)\, seq", row)
				t,dst,src,_id = rowre.group(1), rowre.group(2), rowre.group(3),rowre.group(4).strip()
				which = 'end'
			else:
				continue
		except AttributeError: # invalid line
			continue
		try:
			srcsdict[src]
		except KeyError:
			# not a ping from our program
			continue
		try:
			dstsdict[dst] # not a dst we're tracking, random noise
		except KeyError:
			continue
		try:
			_id = int(_id)
		except ValueError:
			# malformed string
			continue
		try:
			this_dst_meas_i = _id_to_meas_i[_id]
		except KeyError:
			# not a dst we're tracking, random noise
			continue
		try:
			t = datetime.datetime.strptime(t, '%H:%M:%S.%f').timestamp()
		except ValueError:
			continue
		try:
			meas_ret[src][dst]
		except KeyError:
			meas_ret[src][dst] = [{'pop': None, 't_start': None, 't_end': None, 'rtt': None} for _ in \
				range(n_rounds)]
		meas_ret[src][dst][this_dst_meas_i]["t_" + which] = float(t)
		if which == 'end':
			meas_ret[src][dst][this_dst_meas_i]['pop'] = pop
		meas_ret[src][dst][this_dst_meas_i][which + 'pop'] = pop
	# ## reduce size of transfer between processes by deleting empty information
	# to_del = []
	# for src in meas_ret:
	# 	for dst in meas_ret[src]:
	# 		for i,meas in enumerate(reversed(meas_ret[src][dst])):
	# 			if meas['t_end'] is None:
	# 				to_del.append((src,dst,n_rounds-i-1))
	# for src,dst,i in to_del:
	# 	del meas_ret[src][dst][i]
	# 	if len(meas_ret[src][dst]) == 0:
	# 		del meas_ret[src][dst]
	return meas_ret

class Pinger_Wrapper:
	def __init__(self, pops, pop_to_intf):
		self.pops = pops
		self.n_rounds = 3
		self.rate_limit = 1000 ## pps
		self.custom_table_start = 201
		self.src_to_table = {}
		self.bird_table = 20000
		self.pop_to_intf = pop_to_intf
	
	def tap_to_nexthop(self, tap):
		for pop, info in POP_INTERFACE_INFO.items():
			if info['dev'] == tap:
				return info['ip']

	#### NOTE -- if you kill the program and these don't delete properly, could cause problems
	#### so manually delete them from time to time
	def setup_iptable(self, ip_address, tap):
		if not CAREFUL:
			## set up rule that packets with source address ip_address should go out interface tap
			next_hop = self.tap_to_nexthop(tap)
			call("sudo ip rule add from {} table {}".format(ip_address, self.src_to_table[ip_address]), shell=True)
			call("sudo ip route add default via {} dev {} table {}".format(
				next_hop, tap, self.src_to_table[ip_address]), shell=True)
			call("sudo ip route flush cache", shell=True)

	def remove_iptable(self, ip_address, tap):
		if not CAREFUL:
			next_hop = self.tap_to_nexthop(tap)
			call("sudo ip route del default via {} dev {} table {}".format(
				next_hop, tap, self.src_to_table[ip_address], ip_address), shell=True)	
			call("sudo ip rule del from {} table {}".format(ip_address, self.src_to_table[ip_address]), shell=True)
			call("sudo ip route flush cache", shell=True)

	def run(self, srcs, taps, dsts_sets, **kwargs):
		## set up tpdump on all interfaces
		i = 0
		for src, tap in zip(srcs,taps):
			self.src_to_table[src] = self.custom_table_start + i
			self.setup_iptable(src, tap)
			i += 1
		target_fn = os.path.join(TMP_DIR, 'tmp_targ_fn.txt')
		pseudo_timeout = 2
		OUR_PING_ID = 31415

		## write targets to file
		srci=0
		for src, dsts in zip(srcs,dsts_sets):
			with open(target_fn + str(srci),'w') as f:
				for targ in dsts:
					f.write(targ + "\n")
			srci += 1

		override_sleep_period = kwargs.get('sleep_period', None)

		try:
			if not CAREFUL:
				for pop in self.pops:
					call("sudo tcpdump -i {} -n icmp > {} &".format(self.pop_to_intf[pop], pop_to_outfn(pop)), shell=True)
				# Let these start up
				time.sleep(5)
				
				### Time to complete is approximately however many rate limit rounds there are		
				if override_sleep_period is None:
					n_s_per_round = np.max([np.ceil(len(dsts) / self.rate_limit) + \
						pseudo_timeout for dsts in dsts_sets]) + 3
				else:
					n_s_per_round = override_sleep_period
				tstart = time.time()
				max_time_allowed = n_s_per_round * self.n_rounds
				for i in range(self.n_rounds):
					srci=0
					for src, dsts in zip(srcs,dsts_sets):
						this_target_fn = target_fn + str(srci)
						cmd = "cat {} | {} -s {} -r {} -i {} &".format(this_target_fn, 
							PINGER, src, self.rate_limit, OUR_PING_ID + i) 
						call(cmd, shell=True)
						srci += 1
					time.sleep(n_s_per_round)
					if time.time() - tstart > max_time_allowed:
						break
				# Wait for tcpdump to finish writing
				time.sleep(10)
		except:
			import traceback
			traceback.print_exc()
			## kill tcpdump
			call("sudo killall tcpdump", shell=True)
			## get rid of any iptable rules that survived
			for src, tap in zip(srcs,taps):
				self.remove_iptable(src, tap)
			call("rm {}".format(target_fn), shell=True)
			exit(0)
		finally:
			## kill tcpdump
			call("sudo killall tcpdump", shell=True)
			## get rid of any iptable rules that survived
			for src, tap in zip(srcs,taps):
				self.remove_iptable(src, tap)
			call("rm {}*".format(target_fn), shell=True)


		meas_ret = {src: {} for src in srcs}
		_id_to_meas_i = {OUR_PING_ID+i:i for i in range(self.n_rounds)}
		srcsdict = {src:None for src in srcs}
		dstsdict = {dst:None for dstset in dsts_sets for dst in dstset}
		# Parse all the logs
		print("Parsing tcpdump logs")
		if not LOW_MEM:
			pop_jobs = [(pop,copy.deepcopy(meas_ret), copy.deepcopy(_id_to_meas_i), 
				copy.deepcopy(srcsdict), copy.deepcopy(dstsdict), ) for pop in self.pops]
			ppool = multiprocessing.Pool(processes=1)
			rets = ppool.map(parse_ping_results,  pop_jobs)
			ppool.close()
			print("Combining rets from workers")
		else:
			rets = []
			for pop in self.pops:
				args = (pop, meas_ret, _id_to_meas_i, srcsdict, dstsdict, )
				rets.append(parse_ping_results(args))
		for ret in tqdm.tqdm(rets,desc="Parsing all measurements..."): # little complicated to combine
			for src in ret:
				for dst in ret[src]:
					for i,meas in enumerate(ret[src][dst]):
						for k in meas:
							if meas[k] is not None:
								try:
									meas_ret[src][dst]
								except KeyError:
									meas_ret[src][dst] = [{'pop': None, 't_start': None, 't_end': None, 'rtt': None} for _ in \
										range(self.n_rounds)]
								meas_ret[src][dst][i][k] = ret[src][dst][i][k]

		for src in meas_ret:
			for dst in meas_ret[src]:
				for meas in meas_ret[src][dst]:
					if meas['t_start'] is not None and meas['t_end'] is not None \
					 and meas['t_end'] - meas['t_start'] > 0:
						meas['rtt'] = meas['t_end'] - meas['t_start']
		if kwargs.get('remove_bad_meas', False):
			for src in meas_ret:
				for dst, meas in meas_ret[src].items():
					for i in reversed(range(len(meas))):
						if meas[i]['rtt'] is None:
							del meas[i]

		# call("rm {}".format(os.path.join(TMP_DIR, "tcpdumpout*")), shell=True)
		return meas_ret
	
	def simple_test(self):
		call("sudo tcpdump -i ens5 -n icmp > tmp/out.txt &", shell=True)
		time.sleep(3)
		srcs = ['172.31.37.83']
		self.src_to_table[srcs[0]] = '20010'
		self.n_rounds = 1
		taps = ['ens5']
		dst_sets = [['8.8.8.8','1.1.1.1']]
		print(self.run(srcs,taps,dst_sets))
		call("sudo killall tcpdump", shell=True)

if __name__ == "__main__":
	pw = Pinger_Wrapper(['amsterdam'],{'amsterdam': 'tap3'})
	pw.simple_test()

