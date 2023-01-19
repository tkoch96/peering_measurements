import socket, time, struct, select, numpy as np
import netifaces as ni, os

# NETWORK CONSTANTS
IP_HDR_STR = "!BBHHHBBH4s4s"
TCP_HDR_STR = "!HHLLBBHHH"
UDP_HDR_STR = "!HHHH"
PSEUDO_HEADER_STR = '!4s4sBBH'
L_ETH = 14 # bytes
L_V4_IP_HDR = 20 # bytes

# max number of active measurements
MAX_N_ACTIVE = 2000

def tcp_checksum(msg):
	# https://gist.github.com/fffaraz/57144833c6ef8bd9d453
	s = 0
	# loop taking 2 characters at a time
	for i in range(0, len(msg), 2):
		try:
			w = (msg[i] << 8) + msg[i+1]
		except IndexError:
			w = (msg[i] << 8)
		s = s + w
	while s >> 16:
		s = (s>>16) + (s & 0xffff);
	#complement and mask to 4 byte short
	s = ~s & 0xffff
	return s

def get_ip_hdr(src,dst,tp_proto=socket.IPPROTO_UDP):
	# Version IHL
	ihl, version = 5, 4
	# DSC ECN
	dsc, ecn = 0, 0
	iplen = 20
	ttl = 255
	ipid = 31415
	# Flags
	ip_rsv = 0
	ip_dtf = 0
	ip_mrf = 0
	ip_frag_offset = 0
	ip_flg = (ip_rsv << 7) + (ip_dtf << 6) + (ip_mrf << 5) + (ip_frag_offset)
	# Transport protocol
	ip_proto = tp_proto
	# IP Checksum
	ip_chk = 0 # kernel handles this for us

	ip_hdr = [ihl + (version << 4), (dsc << 2) + ecn, iplen, ipid, ip_flg, ttl, 
		ip_proto, ip_chk, src, dst]
	ip_hdr = struct.pack(IP_HDR_STR, *ip_hdr)

	return ip_hdr

def get_icmp_echo_request_packet(src, dst, proc_id, seq_num):
	"""Gets icmp packet."""
	src = socket.inet_aton(src)
	dst = socket.inet_aton(dst)
	ip_hdr = get_ip_hdr(src, dst, tp_proto=1)

	icmp_checksum = 0
	# ICMP type is ICMP_ECHO = 8
	icmp_header = [8, 0, icmp_checksum, proc_id, seq_num]
	icmp_header = struct.pack("!BBHHH", *icmp_header)
   
	pad_bytes = []
	startVal = 0x42
	packet_size = 40
	for i in range(startVal, startVal + (packet_size)):
		pad_bytes += [(i & 0xff)]  # Keep chars in the 0-255 range
	icmp_data = bytes(pad_bytes)
	
	icmp_packet = icmp_header + icmp_data
	icmp_checksum = tcp_checksum(icmp_packet)
	icmp_header = [8, 0, icmp_checksum, proc_id, seq_num]
	icmp_header = struct.pack("!BBHHH", *icmp_header)

	return ip_hdr + icmp_header + icmp_data

class Custom_Pinger:
	def __init__(self, src, tap, dsts):
		
		# should be ~ RTT from VM to PoP + max expected client RTT
		self.timeout = 300 + 400 # milliseconds

		self.dsts = dsts # destinations we are measuring to
		self.n_meas_per_dst = 5
		self.stop = False
		self.sent_all = False
		self.proc_id = 1
		self.seq_num = 1

		# Measurement parameters
		self.unqueued_measurements = {dst:{'n':self.n_meas_per_dst} for dst in self.dsts}
		self.active_measurements = {}
		self.finished_measurements = {dst:[] for dst in self.dsts}

		# Measurement socket
		self.src = src
		s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)
		s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, True) # we will write IP headers
		s.bind((self.src,0x0800))
		self.meas_send_sock = s

		intf = tap
		s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
		s.bind((intf,0x0800))
		self.meas_read_sock = s

	def _check_send_meas(self):
		dsts = self.get_dst()

		if dsts is None: return
		for dst in dsts:
			# send measurement
			icmp_packet = get_icmp_echo_request_packet(self.src, dst, self.proc_id,
				self.seq_num)
			t_now = time.time()
			self.meas_send_sock.sendto(icmp_packet, (dst, 0))
			self.unqueued_measurements[dst]['n'] -= 1
			self.active_measurements[dst] = {
				"t_s": t_now,
				"timeout": self.timeout / 1000, # timeout should be in seconds
				"seq_num": self.seq_num,
			}
			self.seq_num += 1
			self.seq_num = self.seq_num % 10000 # prevent overflow

	def _check_receive_meas(self):
		def handle_ip_packet(packet):
			eth_header = packet[0:L_ETH]
			eth_srcdst = eth_header[8:8+12]
			eth_dst, eth_src = eth_header[0:6],eth_header[6:]

			# Remove ethernet header
			packet = packet[L_ETH:]
			# Look at where this packet is going
			ip_header = packet[0:L_V4_IP_HDR]
			ip_header = struct.unpack(IP_HDR_STR, ip_header)
			packet_len = ip_header[2]
			packet = packet[0:packet_len]
			transport_protocol = ip_header[6]
			ip_src = ip_header[8]
			ip_src_readable = socket.inet_ntoa(ip_src)
			ip_dst = ip_header[9]
			ip_dst_readable = socket.inet_ntoa(ip_dst)
			if transport_protocol == socket.IPPROTO_ICMP:
				if ip_src_readable not in self.dsts: return # not for us
				t_now = time.time()
				# TODO -- get a better check
				icmp_packet = packet[L_V4_IP_HDR:]
				icmp_header = icmp_packet[0:8]
				icmp_data = icmp_packet[8:]
				icmp_header = struct.unpack("!BBHHH", icmp_header)
				seq_num = icmp_header[4]
				try:
					t_sent = self.active_measurements[ip_src_readable]["t_s"]
					eth_src_readable = ':'.join('%02x' % b for b in eth_src[0:6])
					self.finished_measurements[ip_src_readable].append({
						'timeout': False, 'rtt': int((t_now - t_sent) * 1000), 'peer_mac': eth_src_readable
					})
					del self.active_measurements[ip_src_readable]
				except KeyError:
					return # possibly timed out, possibly not for us
			else:
				return # not for us				

		buf = b''
		read_sockets, _, _ = select.select([self.meas_read_sock], [], [], 0)
		while self.meas_read_sock in read_sockets and not self.stop:
			packet,_ = self.meas_read_sock.recvfrom(2048)
			read_sockets, _, _ = select.select([self.meas_read_sock], [], [], 0)
			buf += packet
		while buf != b'':
			# try:
			# need to remove ethernet header
			ip_header_raw = buf[L_ETH:L_ETH+L_V4_IP_HDR]
			ip_header = struct.unpack(IP_HDR_STR, ip_header_raw)
			version_ihl = ip_header[0]
			version = version_ihl >> 4
			if version != 4:
				buf = buf[1:] # bad starting byte
				continue
			# first, confirm we're looking at an IP packet by 
			# checking the checksum
			ip_chk = ip_header[7]
			if tcp_checksum(ip_header_raw) != 0:
				buf = buf[1:] # bad starting byte
				continue
			# valid IP packet -- send the packet to the next layer
			len_packet = ip_header[2]
			handle_ip_packet(buf[0:len_packet+L_ETH])
			buf = buf[len_packet+L_ETH:]
			# except:
			# 	buf = buf[1:] # bad starting byte

	def _check_timeout_meas(self):
		t_now = time.time()
		to_del = []
		for dst, meas in self.active_measurements.items():
			if t_now - meas['t_s'] >= meas["timeout"]:
				to_del.append(dst)
		
		for dst in to_del:
			del self.active_measurements[dst]
			self.finished_measurements[dst].append({
				"timeout": True,
				"rtt": -1
			})

	def get_dst(self):
		if len(self.active_measurements) >= MAX_N_ACTIVE:
			# don't let too many measurements back up
			return None
		ret = []
		n_to_grab = MAX_N_ACTIVE - len(self.active_measurements)
		dsts = list(self.unqueued_measurements.keys())
		np.random.shuffle(dsts)
		for dst in dsts:
			if self.unqueued_measurements[dst]['n'] > 0:
				if dst not in self.active_measurements:
					ret.append(dst)
					if len(ret) == n_to_grab: break
		if len(ret) > 0:
			 return ret
		self.sent_all = True
		return None

	def get_finished_meas(self):
		ret = {}
		for dst,msmts in self.finished_measurements.items():
			valid_meas = [meas for meas in msmts if meas['rtt'] != -1]
			if len(valid_meas) == 0:
				ret[dst] = {'rtt': -1, 'peer_mac': None}
			else:
				ret[dst] = {
					'rtt': min([meas['rtt'] for meas in valid_meas]),
					'peer_mac': list(set([meas['peer_mac'] for meas in valid_meas]))
				}
		return ret

	def run(self):		
		while not self.stop:
			## Main loop
			self._check_send_meas()
			self._check_receive_meas()
			self._check_timeout_meas()

			self.stop = self.sent_all and len(self.active_measurements) == 0


if __name__ == "__main__":
	### Old test script
	# targ = ['95.47.119.153']
	# import numpy as np
	# res = []
	# for src in ['184.164.240.1','184.164.241.1']:
	# 	np.random.shuffle(targ)
	# 	cp = Custom_Pinger(src, 'tap5', targ)
	# 	cp.run()
	# 	res.append(cp.get_finished_meas())
	# in_both = [k for k in res[0] if k in res[1]]
	# in_both = [k for k in res[0] if res[0][k]['rtt'] != -1 and res[1][k]['rtt'] != -1]
	# diffs = [res[0][k]['rtt'] - res[1][k]['rtt'] for k in in_both]
	# for dst, diff in zip(in_both, diffs):
	# 	print("{} -- ({} ms diff), Peer A: {}, Peer B: {}".format(dst,
	# 		diff, res[0][dst]['peer_mac'], res[1][dst]['peer_mac']))

	import argparse
	cp = Custom_Pinger(src,tap,targs)
	pickle.dump(cp.get_finished_meas(), open(args.outfn,'wb'))
