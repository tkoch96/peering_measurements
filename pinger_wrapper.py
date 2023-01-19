import socket, time, struct, select, numpy as np, re, datetime, tqdm
from constants import *
import netifaces as ni, os
from subprocess import call, check_output

PINGER = "sudo /home/tom/pinger/target/release/pinger"

class Pinger_Wrapper:
	def __init__(self, pops, pop_to_intf):
		self.pops = pops
		self.n_rounds = 10
		self.rate_limit = 100000 ## pps
		self.custom_table_start = 201
		self.src_to_table = {}
		self.bird_table = 20000
		self.pop_to_intf = pop_to_intf
	
	def tap_to_nexthop(self, tap):
		for row in check_output("ip route show table {} | grep {}".format(
			self.bird_table, tap), shell=True).decode().split('\n'):
			next_hop = re.search("via (.+) dev", row).group(1)
			break
		return next_hop

	#### NOTE -- if you kill the program and these don't delete properly, could cause problems
	#### so manually delete them from time to time
	def setup_iptable(self, ip_address, tap):
		## set up rule that packets with source address ip_address should go out interface tap
		next_hop = self.tap_to_nexthop(tap)
		call("sudo ip rule add from {} table {}".format(ip_address, self.src_to_table[ip_address]), shell=True)
		call("sudo ip route add default via {} dev {} table {}".format(
			next_hop, tap, self.src_to_table[ip_address], ip_address), shell=True)
		call("sudo ip route flush cache", shell=True)

	def remove_iptable(self, ip_address, tap):
		next_hop = self.tap_to_nexthop(tap)
		call("sudo ip route del default via {} dev {} table {}".format(
			next_hop, tap, self.src_to_table[ip_address], ip_address), shell=True)	
		call("sudo ip rule del from {} table {}".format(ip_address, self.src_to_table[ip_address]), shell=True)
		call("sudo ip route flush cache", shell=True)

	def run(self, srcs, taps, dsts, **kwargs):
		## set up tpdump on all interfaces
		i = 0
		for src, tap in zip(srcs,taps):
			self.src_to_table[src] = self.custom_table_start + i
			self.setup_iptable(src, tap)
			i += 1
		target_fn = os.path.join(CACHE_DIR, 'tmp_targ_fn.txt')
		pseudo_timeout = 3
		OUR_PING_ID = 31415
		try:
			pop_to_outfn = lambda pop : os.path.join(CACHE_DIR, "tcpdumpout_{}.txt".format(pop))

			for pop in self.pops:
				call("sudo tcpdump -i {} -n icmp > {} &".format(self.pop_to_intf[pop], pop_to_outfn(pop)), shell=True)
			# Let these start up
			time.sleep(5)

			with open(target_fn,'w') as f:
				for targ in dsts:
					f.write(targ + "\n")
			n_s_per_round = np.ceil(len(dsts) / self.rate_limit) + pseudo_timeout
			for i in range(self.n_rounds):
				for src in srcs:
					cmd = "cat {} | {} -s {} -r {} -i {} &".format(target_fn, 
						PINGER, src, self.rate_limit, OUR_PING_ID + i) 
					print(cmd)
					call(cmd, shell=True)
				time.sleep(n_s_per_round)
		except:
			import traceback
			traceback.print_exc()
			self.remove_iptable(self.src, tap)
			exit(0)
		finally:
			## kill tcpdump
			call("sudo killall tcpdump", shell=True)
			## get rid of any iptable rules that survived
			for src, tap in zip(srcs,taps):
				self.remove_iptable(src, tap)
			call("rm {}".format(target_fn), shell=True)

		meas_ret = {src: {dst: [{'pop': None, 't_start': None, 't_end': None, 'rtt': None} for _ in \
				range(self.n_rounds)] for dst in dsts} for src in srcs}
		_id_to_meas_i = {OUR_PING_ID+i:i for i in range(self.n_rounds)}
		srcsdict = {src:None for src in srcs}
		for pop in tqdm.tqdm(self.pops, desc="Parsing ping results."):
			for row in open(pop_to_outfn(pop), 'r'):
				# parse the measurements
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
				try:
					srcsdict[src]
				except KeyError:
					# not a ping from our program
					continue
				try:
					meas_ret[src][dst] # not a dst we're tracking, random noise
				except KeyError:
					continue
				_id = int(_id)
				try:
					this_dst_meas_i = _id_to_meas_i[_id]
				except KeyError:
					# not a dst we're tracking, random noise
					continue
				t = datetime.datetime.strptime(t, '%H:%M:%S.%f').timestamp()
				meas_ret[src][dst][this_dst_meas_i]["t_" + which] = float(t)
				if which == 'end':
					meas_ret[src][dst][this_dst_meas_i]['pop'] = pop
				meas_ret[src][dst][this_dst_meas_i][which + 'pop'] = pop

		for src in meas_ret:
			for dst in meas_ret[src]:
				for meas in meas_ret[src][dst]:
					if meas['t_start'] is not None and meas['t_end'] is not None \
					 and meas['t_end'] - meas['t_start'] > 0:
						meas['rtt'] = meas['t_end'] - meas['t_start']
					elif meas['t_end'] is not None:
						print("Weirdness --- reply but no request from {}".format(dst))
		if kwargs.get('remove_bad_meas'):
			for src in meas_ret:
				for dst, meas in meas_ret[src].items():
					for i in reversed(range(self.n_rounds)):
						if meas[i]['rtt'] is None:
							del meas[i]

		call("rm {}".format(os.path.join(CACHE_DIR, "tcpdumpout*")), shell=True)
		return meas_ret

