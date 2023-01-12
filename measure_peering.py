import netifaces as ni, os, re, socket, csv, time, json, numpy as np, pickle, geopy.distance
from subprocess import call, check_output
import subprocess

from custom_ping import Custom_Pinger


CAREFUL =  False
np.random.seed(31415)

### To get latency measurements
# start tcpdump
# for peer in pop
# measure to all users of interest from the peer
# end tcpdump
# post-process pcaps to look at latencies


### For pairwise preferences
# for pair in pairs
# start tcpdump
# measure to all users of interest
# end tcpdump
# parse tcpdump to look at who wins

def check_ping_responsive(ips):
	print("Checking responsiveness for {} IP addresses".format(len(ips)))
	ret = []
	ip_chunks = split_seq(ips,100)
	for ips in ip_chunks:
		addresses_str = " ".join(ips)
		if addresses_str != "":
			out_fn = 'tmp.warts'
			scamp_cmd = 'sudo scamper -O warts -c "ping -c 1" -p 8000 -M tak2154atcolumbiadotedu'\
				' -l peering_interfaces -o {} -i {}'.format(out_fn, addresses_str)
			try:
				check_output(scamp_cmd, shell=True)
				cmd = "sc_warts2json {}".format(out_fn)
				out = check_output(cmd, shell=True).decode()
				for meas_str in out.split('\n'):
					if meas_str == "": continue
					measurement_obj = json.loads(meas_str)
					meas_type = measurement_obj['type']
					if meas_type == 'ping':
						dst = measurement_obj['dst']
						if measurement_obj['responses'] != []:
							ret.append(dst)
			except:
				# likely bad input
				import traceback
				traceback.print_exc()
				pass
	return ret
def get_intersection(set1, set2):
	"""Gets intersection of two sets."""
	return list(set(set1) & set(set2))
def get_difference(set1, set2):
	"""Gets set1 - set2."""
	set1 = set(set1); set2 = set(set2)
	return list(set1.difference(set2))
def split_seq(seq, n_pieces):
	# splits seq into n_pieces chunks of approximately equal size
	# useful for splitting things to divide among processors
	newseq = []
	splitsize = 1.0/n_pieces*len(seq)
	for i in range(n_pieces):
		newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
	return newseq
DATA_DIR = "data"
min_t_between = 600 # PEERING testbed policy


class Peering_Pinger():
	def __init__(self, mode):
		self.run = {
			'calculate_latency': self.calculate_latency,
			'pairwise_preferences': self.pairwise_preferences,
		}[mode]

		# TODO -- maybe extend to all pops
		self.pop = 'amsterdam01'
		self.pops = ['amsterdam01']
		all_muxes_str = check_output("sudo client/peering openvpn status", shell=True).decode().split('\n')
		this_mux_str = [mux_str for mux_str in all_muxes_str if self.pop in mux_str]
		assert len(this_mux_str) == 1
		this_mux_str = this_mux_str[0]
		self.pop_intf = "tap" + re.search("tap(\d+)", this_mux_str).group(1)
		self.default_intf = 'eth0'
		self.internal_ip = ni.ifaddresses(self.default_intf)[ni.AF_INET][0]['addr']
		self.internal_ip_bytes = socket.inet_aton(self.internal_ip)

		self.unicast_pref = "184.164.241.0/24"
		self.unicast_addr = "184.164.241.1"

		peering_peers = csv.DictReader(open(os.path.join(DATA_DIR, "peering_peers.csv"), 'r'))
		self.peers = {}
		self.peer_to_id = {}

		self.peer_macs = {}
		self.peer_mac_fn = os.path.join(DATA_DIR, 'peer_macs.csv')
		if not os.path.exists(self.peer_mac_fn):
			self._call("touch {}".format(self.peer_mac_fn))
		with open(self.peer_mac_fn, 'r') as f:
			for row in f:
				if row[0:3] == 'pop': continue
				pop,pm,peer = row.strip().split("\t")
				try:
					self.peer_macs[pop][pm] = peer
				except KeyError:
					self.peer_macs[pop] = {pm: peer}
		try:
			self.peer_macs[self.pop]
		except KeyError:
			self.peer_macs[self.pop] = {}

		for row in peering_peers:
			pop, peer, session_id = row["BGP Mux"], row["Peer ASN"], row["Session ID"]
			if ":" in row["Peer IP address"]: continue
			try:
				self.peers[pop].append(peer)
			except KeyError:
				self.peers[pop] = [peer]
			try:
				self.peer_to_id[pop,peer]
				# print("Warning -- multiple session ID's for {} {}".format(pop,peer))
			except KeyError:
				self.peer_to_id[pop,peer] = session_id
		for pop in self.peers:
			self.peers[pop] = list(set(self.peers[pop])) # remove dups

		self.reachable_dsts, self.measured_to = {}, {}
		with open(os.path.join(DATA_DIR, "client_lats.csv"), 'r') as f:
			for row in f:
				if row.startswith("ip"): continue
				client_dst, peer, rtt = row.strip().split('\t')
				if rtt != '-1':
					try:
						self.reachable_dsts[peer].append(client_dst)
					except KeyError:
						self.reachable_dsts[peer] = [client_dst]
				try:
					self.measured_to[peer].append(client_dst)
				except KeyError:
					self.measured_to[peer] = [client_dst]

		def parse_row_loc(row):
			if row['lat'] == 'NaN' or row['lon'] == 'NaN':
				return None
			return (round(float(row['lat']),2), round(float(row['lon']),2))
		self.pop_to_loc = {'amsterdam01': (52.359,4.933)}

		client_pop_mapping_fn = os.path.join('cache', 'client_loc_to_pop.pkl')
		if os.path.exists(client_pop_mapping_fn):
			self.client_loc_to_pop, self.pop_to_client_loc = pickle.load(open(client_pop_mapping_fn, 'rb'))
		else:
			client_24_loc_d = list(csv.DictReader(open(os.path.join(DATA_DIR, "client_24_loc.csv"), 'r')))
			locs = list(set([parse_row_loc(row) for row in client_24_loc_d]))
			pop_locs = np.array([self.pop_to_loc[pop] for pop in self.pops])
			self.client_loc_to_pop = {}
			self.pop_to_client_loc = {}
			for i,loc in enumerate(locs):
				if loc is None: continue
				closest_pop = np.argmin(np.linalg.norm(loc - pop_locs,axis=1))
				if geopy.distance.geodesic(self.pop_to_loc[self.pops[closest_pop]], loc).km < 1000:
					try:
						self.pop_to_client_loc[self.pops[closest_pop]].append(loc)
					except KeyError:
						self.pop_to_client_loc[self.pops[closest_pop]] = [loc]
					self.client_loc_to_pop[tuple(loc)] = self.pops[closest_pop]
			pickle.dump([self.client_loc_to_pop, self.pop_to_client_loc], open(client_pop_mapping_fn,'wb'))

		self.pop_to_clients = {pop:[] for pop in self.pops}
		client_24_loc_d = csv.DictReader(open(os.path.join(DATA_DIR, "client_24_loc.csv"), 'r'))
		for row in client_24_loc_d:
			loc = parse_row_loc(row)
			try:
				pop = self.client_loc_to_pop[loc]
			except KeyError:
				continue
			if ":" in row['reachable_addr']: continue
			self.pop_to_clients[pop].append(row['reachable_addr'])

		self.addresses_that_respond_to_ping = {0:[],1:[]}
		self.addresses_that_respond_to_ping_fn = os.path.join(DATA_DIR, "addresses_that_respond_to_ping.csv")
		if os.path.exists(self.addresses_that_respond_to_ping_fn):
			with open(self.addresses_that_respond_to_ping_fn, 'r') as f:
				for row in f:
					ipadr,reach = row.strip().split(',')
					self.addresses_that_respond_to_ping[int(reach)].append(ipadr)

		self.default_ip = ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']

	def _call(self, cmd, careful=False):
		print(cmd)
		if not careful:
			call(cmd, shell=True)

	def _check_output(self, cmd, careful=False):
		print(cmd)
		if not careful:
			return check_output(cmd,shell=True)

	def advertise_to_peer(self, peer):
		# advertise only to this peer
		self._call("sudo client/peering prefix announce -m {} -c 47065,{} {}".format(
			self.pop, self.peer_to_id[self.pop, peer], self.unicast_pref),careful=CAREFUL)
	def withdraw_from_peer(self, peer):
		# advertise only to this peer
		self._call("sudo client/peering prefix withdraw -m {} -c 47065,{} {}".format(
			self.pop, self.peer_to_id[self.pop, peer], self.unicast_pref),careful=CAREFUL)

	def announce_anycast(self):
		self._call("sudo client/peering prefix announce -m {} {}".format(
			self.pop, self.peer_to_id[self.pop, peer], self.unicast_pref),careful=CAREFUL)
	def withdraw_anycast(self):
		self._call("sudo client/peering prefix withdraw -m {} {}".format(
			self.pop, self.peer_to_id[self.pop, peer], self.unicast_pref),careful=CAREFUL)

	def calculate_latency(self):
		# # Get latency from VM to clients
		# cp = Custom_Pinger(self.default_ip, 'eth0', client_dsts)
		# cp.run()
		# lats = cp.get_finished_meas()
		# self.parse_latencies(lats, peer='VM')

		# # Get latency from mux to VM
		# self.announce_anycast()
		# cp = Customer_Pinger(self.unicast_addr, self.pop_intf, [self.unicast_addr])
		# lats = cp.get_finished_meas()
		# self.parse_latencies(lats, peer='VM')
		# self.withdraw_anycast()

		# Get latencies (and reachabilities) from all clients to all peers
		for peer in self.peers[self.pop]:
			client_dsts = self.get_pop_to_clients(self.pop, peer=peer)
			if client_dsts == []: 
				print("Done with peer {}, skipping".format(peer))
				continue
			print("Measuring to {} dsts, peer : {}.".format(len(client_dsts), peer))
			self.advertise_to_peer(peer)
			ta = time.time() 

			cp = Custom_Pinger(self.unicast_addr, self.pop_intf, client_dsts)
			cp.run()

			lats = cp.get_finished_meas()

			print("Done measuring, {} results.".format(len(lats)))


			self.parse_latencies(lats, peer=peer)
			# We don't need to explicitly withdraw due to the nature of the advertisement

			tslp = np.max([ min_t_between - (time.time() - ta), 1])
			print("Sleeping for {} seconds.".format(tslp))
			time.sleep(tslp)

		self.withdraw_from_peer(peer)
	
	def pairwise_preferences(self):
		# Get pairwise preferences
		for peer_i in self.peers[self.pop]:
			self.advertise_to_peer(peer_i)
			for peer_j in self.peers[self.pop]:
				if peer_j == peer_i: break

				# get set of clients for whom peer i and peer j are both reachable
				client_dsts = get_intersection(self.reachable_dsts[self.pop,peer_i], self.reachable_dsts[self.pop, peer_j])
				self.advertise_to_peer(peer_j)
				taj = time.time()

				# Measure to these clients
				cp = Customer_Pinger(self.unicast_addr, self.pop_intf, client_dsts)
				cp.run()
				lats = cp.get_finished_meas()

				# note which peer is preferred
				self.parse_preferences(lats)

				while time.time() - taj < min_t_between:
					tslp = time.time() - (taj + min_t_between) + 1
					time.sleep(tslp)
				self.withdraw_from_peer(peer_j)

			self.withdraw_from_peer(peer_i)

	def parse_latencies(self, lats, peer=None):
		if peer is not None and peer != "VM":
			peer_macs = []
			for meas in lats.values():
				if meas['peer_mac'] is not None:
					for pm in meas['peer_mac']:
						peer_macs.append(pm)
			peer_macs = list(set(peer_macs))
			if peer_macs != []: 
				if len(peer_macs) > 1:
					print("WARNING -- only 1 peer active but 2 macs registered: {}".format(peer_macs))
				for pm in peer_macs:
					self.peer_macs[self.pop][pm] = peer
				with open(self.peer_mac_fn, 'a') as f:
					for pm in peer_macs:
						f.write("{}\t{}\t{}\n".format(self.pop, pm, peer))

		with open(os.path.join(DATA_DIR, "client_lats.csv"), 'a') as f:
			for client_dst in lats:
				rtt,peer_mac = lats[client_dst]['rtt'], lats[client_dst]['peer_mac']
				if peer_mac is not None:
					peer_mac = peer_mac[0]
				if rtt != -1:
					if peer != "VM":
						_peer = self.peer_macs[self.pop][peer_mac]
					else:
						_peer = "VM"
					try:
						self.reachable_dsts[self.pop, _peer].append(client_dst)
					except KeyError:
						self.reachable_dsts[self.pop, _peer] = [client_dst]
				f.write("{}\t{}\t{}\n".format(client_dst, peer, rtt))

	def parse_preferences(self, lats, peer_i, peer_j):
		with open(os.path.join(DATA_DIR, "client_preferences.csv"), 'a') as f:
			for client_dst, meas in lats.items():
				peer = self.peer_macs[self.pop][meas['peer_mac'][0]]
				if peer == peer_i:
					bit = '0'
				else:
					bit = '1'
				f.write("{}\t{}\t{}\t{}\n".format(client_dst, peer_i, peer_j, bit))

	def get_pop_to_clients(self, pop, peer=None):
		# TODO -- eventually just return everything
		np.random.shuffle(self.pop_to_clients[pop])
		dsts = self.pop_to_clients[pop]
		if peer is not None and peer in self.measured_to:
			dsts = get_difference(dsts, self.measured_to[peer])

		## see which ones respond to ping
		# check if we've already measured responsiveness
		already_know_responsiveness = self.addresses_that_respond_to_ping[0] + self.addresses_that_respond_to_ping[1]
		dont_know_responsiveness = get_difference(dsts,already_know_responsiveness)
		responsive_dsts = check_ping_responsive(dont_know_responsiveness) + get_intersection(dsts, self.addresses_that_respond_to_ping[1])
		responsive_dsts = list(set(responsive_dsts))
		self.addresses_that_respond_to_ping[1] += dsts
		self.addresses_that_respond_to_ping[0] += get_difference(dont_know_responsiveness, dsts)
		with open(self.addresses_that_respond_to_ping_fn, 'w') as f:
			for i in [0,1]:
				for dst in self.addresses_that_respond_to_ping[i]:
					f.write("{},{}\n".format(dst,i))

		return responsive_dsts

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="calculate_latency",choices=['calculate_latency','pairwise_preferences'])
	args = parser.parse_args()

	pp = Peering_Pinger(mode=args.mode)
	pp.run()