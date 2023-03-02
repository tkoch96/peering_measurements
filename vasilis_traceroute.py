# Author -- Vasilis Giotsas
import os
import sys
import pyasn
import radix
import logging
from itertools import groupby
from collections import defaultdict
from helpers import *

class Traceroute(object):
    def __init__(self, asndb_file=False, ip_to_asn=None):
        if asndb_file:
            self.asndb = pyasn.pyasn(asndb_file)
        self.ixp_interfaces = {}
        self.ixppref_tree = radix.Radix()
        if not ip_to_asn:
            self.ip_to_asn_mapping = self.ip_to_asn_mapping()
        else:
            self.ip_to_asn_mapping = lambda k : (ip_to_asn(ip32_to_24(k)), None)
            
    def read_ixp_ip_members(self, ixp_interfaces_file):
        """
        Reads a file that includes IXP members and their corresponding peering IP interfaces
        and populates:
            - a dictionary that maps the members' ASNs to their IXP IPs
            - a radix tree with the /24 that includes the IP of each member
        :param string ixp_interfaces_file: the path to the file with the IXP membership data
        :returns: None
        """
        # Read the IXP IPs to ASNs mapping
        with open(ixp_interfaces_file, "rt", encoding="ISO-8859-1") as fin:
            for line in fin:
                line = line.strip()
                if line.startswith("#"):
                    continue
                fields = line.split("|")
                if len(fields) > 1:
                    interface = fields[0]
                    member = "%s@%s" % (fields[1], fields[2])
                    self.ixp_interfaces[interface] = member
                    # Check if we need to add the corresponding /24 prefix in the radix tree
                    rnode = self.ixppref_tree.search_best(interface)
                    if rnode is None:
                        # Get the /24 prefix from the interface
                        pfx_interface = '.'.join(interface.split(".")[0:3]) + ".0/24"
                        rnode = self.ixppref_tree.add(pfx_interface)
                        rnode.data["origin"] = fields[2]

    def read_ixp_prefixes(self, ixp_prefixes_file):
        """
        Reads a file that includes the prefixes used by IXP peering LANs and
        populates a radix tree with the mapping between the IP prefix and the
        IXP name.
        :param string ixp_prefixes_file: the path to the file with the IXP prefixes
        :returns: None
        """
        # ixp_prefixes_file = "ixp_to_prefix_merged_pdb_pch_he.20160308.list"
        try:
            if os.stat(ixp_prefixes_file).st_size == 0:
                raise IOError('file is empty')
            logging.info('reading ixp pref file %s\n' % ixp_prefixes_file)
            with open(ixp_prefixes_file, 'rt', encoding="ISO-8859-1") as IXPPREF:
                for line in IXPPREF:
                    if not line.startswith("#"):
                        fields = line.strip().split('\t')
                        if len(fields) != 2:
                            continue
                        try:
                            rnode = self.ixppref_tree.add(fields[0])
                            ixp_names = fields[1].split(",")
                            pdb_ixp_name = ixp_names[-1]
                            he_ixp_name = ixp_names[0]
                            pch_ixp_name = ixp_names[1]
                            ixp_name = pdb_ixp_name

                            if pdb_ixp_name != "n/a":
                                ixp_name = '_'.join(pdb_ixp_name.split("_")[1:]).replace("_", " ")
                            elif he_ixp_name != "n/a":
                                ixp_name = '_'.join(he_ixp_name.split("_")[1:]).replace("_", " ")
                            elif pch_ixp_name != "n/a":
                                ixp_name = '_'.join(pch_ixp_name.split("_")[1:]).replace("_", " ")
                            if ixp_name != "n/a":
                                rnode.data["origin"] = ixp_name
                        except ValueError:
                            logging.error('error adding ixp pref %s\n' % fields[1])
        except OSError as o:
            logging.error('ixp pref file error: %s\n' % o)
        except IOError as i:
            logging.error('File open failed: %s\n' % i)

    @staticmethod
    def detect_cycles(hops, void_hop):
        """
        Detect if a path has cycles (same hop repeated in non-consecutive positions)
        :param hops: the list of hops
        :param Object void_hop: the representation for the unresponsive hop
        :return: True if the paths contains hops, False otherwise
        """
        has_cycle = False
        if len(hops) > 2:  # we cannot have cycles in paths shorter than two hops
            for idx, hop in enumerate(hops[2:]):
                orig_idx = idx + 2
                if hop in hops[:orig_idx] and hop != hops[orig_idx - 1] and hop != void_hop and hops[
                    orig_idx - 1] != void_hop:
                    has_cycle = True
                    break
        return has_cycle


    @staticmethod
    def remove_prepending(as_hops, vp_asn=None):
        if vp_asn:
            as_hops.insert(0, vp_asn)
        non_prepended_path = [asn[0] for asn in groupby(as_hops)]
        indeces_to_delete = list()
        for idx, hop in enumerate(non_prepended_path):
            # remove unresponsive/unresolved hops and their previous hop if they are between hops of the same AS
            # for instance the sequence `174 0 174` will become `174`
            if non_prepended_path[idx] is None and (len(non_prepended_path) - 1) > idx > 0 \
                    and non_prepended_path[idx - 1] == non_prepended_path[idx + 1]:
                indeces_to_delete += [idx, idx + 1]
        for i in sorted(indeces_to_delete, reverse=True):
            del non_prepended_path[i]

        return non_prepended_path

    def get_border_links(self, ip_path, probe_id, destination_address):
        """
        Returns the mapping between the ASN links and the corresponding
        IP-level links in the traceroute path
        :param list ip_path: the list of hops
        :return: mapping between ASN links and IP hops
        :rtype: (dict, dict)
        """

        previous_asn = None
        for idx, ip in enumerate(ip_path):
            if ip is not None:
                # Check if the hop is an IXP
                asn = None
                asn = self.ixp_interfaces.get(ip, None)
                if not asn:
                    rnode = self.ixppref_tree.search_best(ip)
                    if rnode:
                        if idx + 1 > len(ip_path):
                            asn = self.ip_to_asn_mapping(ip_path[idx + 1])
                    else:
                        asn, prefix = self.ip_to_asn_mapping(ip)
                else:
                    asn = int(asn.split("@")[0])
                if asn is not None and asn != previous_asn and previous_asn is not None:
                    as_link = "{0} {1}".format(previous_asn, asn)
                    ip_link = "{0} {1}".format(ip_path[idx - 1], ip)
                    self.border_links[as_link].add(ip_link)
                    self.border_link_frequency[ip_link].add((probe_id, destination_address))
                previous_asn = asn
            else:
                previous_asn = None

    def ip_to_asn(self, atlas_hops):
        """
        Takes as input a list of lists containing the IPs at each hop
        and returns the AS-level path.
        :param list atlas_hops: the list of Atlas hops
        :return: the list of AS hops
        :rtype: list
        """
        asn_hops = []
        for hop in atlas_hops:
            previous_asn = None
            resolved_asns = set()
            for ip in set(hop):
                asn = None
                if ip is not None and ip != "*":
                    # First check if the IP belongs to an IXP
                    asn = self.ixp_interfaces.get(ip, None)
                    if not asn:
                        try:
                            rnode = self.ixppref_tree.search_best(ip)
                            if rnode:
                                asn = previous_asn
                            else:
                                asn, prefix = self.ip_to_asn_mapping(ip)
                        except ValueError:
                            logging.error("Could not parse addr {}".format(ip))
                    else:
                        asn = int(asn.split("@")[0])
                    resolved_asns.add(asn)
            resolved_asns.discard(None)
            if len(resolved_asns) == 1:
                previous_asn = resolved_asns.pop()
            else:
                previous_asn = None
            asn_hops.append(previous_asn)

        return asn_hops

    def get_ixp_triplets(self, ip_hops, ixp_name=False):
        """
        Extracts the (near-end, IXP, far-end) IP triplets from traceroute paths
        that traverse the target IXP or all IXPs if an IXP name isn't provided
        :param list<Hop> ip_hops: the sequence of Hop objects
        :param str ixp_name: the name of the target IXP
        :return: the set of IXP IP triplets in the traceroute path and the corresponding min RTTs
        :rtype: set<str>
        """
        ixp_triplets = set()

        for i in range(0, len(ip_hops)):
            rtts = list()
            interfaces = set()

            for packet in ip_hops[i].packets:
                interfaces.add(packet.origin)
                rtts.append(packet.rtt)
            if len(interfaces) == 1:
                try:
                    ip = ip_hops[i].packets[0].origin
                    if ip:
                        ixp_member = self.ixppref_tree.search_best(ip)
                        if ixp_member and (not ixp_name or (ixp_name == ixp_member.data['origin'])):
                            # First check if the IP belongs to an IXP
                            asn = self.ixp_interfaces.get(ip, None)
                            ip = "%s|%s" % (ip, asn)
                            triplet = []
                            # parse the previous hop
                            if i > 0:
                                interfaces = set()
                                previous_rtts = list()
                                for packet in ip_hops[i - 1].packets:
                                    interfaces.add(packet.origin)
                                    previous_rtts.append(packet.rtt)
                                if len(interfaces) == 1:
                                    previous_ip = ip_hops[i - 1].packets[0].origin
                                    asn = "x"
                                    min_rtt = "x"
                                    if previous_ip:
                                        asn, prefix = self.ip_to_asn_mapping(previous_ip)
                                        min_rtt = min(rtts) - min(previous_rtts)
                                    else:
                                        previous_ip = "x"
                                    previous_hop = "%s|%s" % (previous_ip, asn)
                                    triplet.append(previous_hop)
                                    triplet.append("%s|%s" % (ip, min_rtt))
                                    # parse the next hop
                                    if i + 1 < len(ip_hops):
                                        interfaces = set()
                                        next_rtts = list()
                                        for packet in ip_hops[i + 1].packets:
                                            interfaces.add(packet.origin)
                                            next_rtts.append(packet.rtt)
                                        if len(interfaces) == 1:
                                            next_ip = ip_hops[i + 1].packets[0].origin
                                            asn = "x"
                                            min_rtt = "x"
                                            if next_ip:
                                                asn, prefix = self.ip_to_asn_mapping(next_ip)
                                                min_rtt = min(next_rtts) - min(rtts)
                                                '''
                                                min_rtt = next_rtts[0] - rtts[0]
                                                for idx in range(1, len(rtts)):
                                                    if len(next_rtts) > idx and min_rtt > (next_rtts[idx] - rtts[idx]):
                                                        min_rtt = next_rtts[idx] - rtts[idx]
                                                '''
                                            else:
                                                next_ip = "x"
                                            next_hop = "%s|%s|%s" % (next_ip, asn, min_rtt)
                                            triplet.append(next_hop)
                                    else:
                                        triplet.append('x')
                            else:
                                triplet.append('x')

                            try:
                                if len(triplet) == 3:
                                    ixp_triplets.add(','.join(triplet))
                            except TypeError as te:
                                print(str(te))
                except ValueError as e:
                    logging.critical(str(e))
                    sys.exit(-1)
                except TypeError as e:
                    logging.critical(str(e))
                    sys.exit(-1)

        return ixp_triplets

    def get_non_ixp_links(self, ip_hops, asn_to_ixp):
        """
        Extracts the non-ixp links for IPs that also appear as near-end hops in IXP links
        :param list<Hop> ip_hops: the sequence of Hop objects
        :param set<str> near_end_ips: the set of near-end IPs for which the function will search for non-ixp links
        :return: the set of the non-IXP IP links in the traceroute path and the corresponding min RTTs
        :rtype: set<str>
        """
        for i in range(0, len(ip_hops)):
            rtts = list()
            interfaces = set()

            for packet in ip_hops[i].packets:
                interfaces.add(packet.origin)
                rtts.append(packet.rtt)
            if len(interfaces) == 1:
                try:
                    ip = ip_hops[i].packets[0].origin
                    ne_asn, prefix = self.ip_to_asn_mapping(ip)
                    if ne_asn and ne_asn in asn_to_ixp:
                        if i + 1 < len(ip_hops):
                            interfaces = set()
                            next_rtts = list()
                            for packet in ip_hops[i + 1].packets:
                                interfaces.add(packet.origin)
                                next_rtts.append(packet.rtt)
                            if len(interfaces) == 1:
                                next_ip = ip_hops[i + 1].packets[0].origin
                                fe_asn, prefix = self.ip_to_asn_mapping(next_ip)
                                # Check that the IP doesn't belong to an IXP
                                ixp_member = self.ixppref_tree.search_best(next_ip)

                                if not ixp_member and fe_asn in asn_to_ixp:
                                    ip_link = ip + " " + next_ip
                                    as_link = "%s %s" % (ne_asn, fe_asn)
                                    min_rtt = min(next_rtts) - min(rtts)
                                    yield "%s|%s|%s" % (ip_link, as_link, min_rtt)

                except ValueError as e:
                    logging.critical(str(e))
                    # sys.exit(-1)
                except TypeError as e:
                    logging.critical(str(e))
                    # sys.exit(-1)

    @staticmethod
    def ip_path_string(ip_path):
        """
        Converts the list of hops in a Traceroute path to a string.
        The interfaces at each hop are separated with comma.
        If traceroute used more than one probes per TTL and for the same TTL (hop) there are more than one interfaces,
        they are separated with dash '-'
        :param list<list> ip_path: the list of IP interfaces at each traceroute hop
        :return: the string representation of the traceroute path
        """
        hops = list()
        for hop in ip_path:
            hop_ifaces = set()
            for ip in hop:
                hop_ifaces.add(ip)
            if len(hop_ifaces) > 1:
                hop_ifaces.discard(None)
                hop_ifaces.discard('x')
            hops.append('-'.join(map(str, hop_ifaces)))
        return ','.join(hops)