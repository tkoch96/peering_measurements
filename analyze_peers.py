import csv

all_popps_d = list(csv.DictReader(open('data/vultr_peers.csv','r')))
all_classifications_d = list(csv.DictReader(open('data/2022-12_categorized_ases.csv','r')))

all_classifications = {row['ASN']:row for row in all_classifications_d}


all_peers = list(set([int(row['peer']) for row in all_popps_d]))

as_classifications_d = list(csv.DictReader(open('data/AS_Features_032022.csv','r')))
as_classifications = {row['asNumber']: row for row in as_classifications_d}

class_to_n = {}

for peer in sorted(all_peers):
	try:
		corresponding_classification = all_classifications["AS{}".format(peer)]
		hr_name = None
		try:
			hr_name = as_classifications[str(peer)]['Organization']
		except KeyError:
			pass
		# print("PEER : {}, classify : {}, ({}) {}".format(peer, corresponding_classification['Category 1 - Layer 1'],
		# 	corresponding_classification['Category 1 - Layer 2'], hr_name))
		try:
			class_to_n[corresponding_classification['Category 1 - Layer 1'],
			corresponding_classification['Category 1 - Layer 2']] += 1
		except KeyError:
			class_to_n[corresponding_classification['Category 1 - Layer 1'],
			corresponding_classification['Category 1 - Layer 2']] = 1

	except KeyError:
		print("PEER : {}, no info".format(peer))

print(sorted(class_to_n.items() , key = lambda el : el[1]))

with open('data/vultr_peers_with_org.csv','w') as f:
	f.write("pop,asn,next_hop,org\n")
	for row in all_popps_d:
		org = ""
		try:
			org = as_classifications[row['peer']]['Organization']
		except KeyError:
			pass			
		f.write("{},{},{},{}\n".format(row['pop'],row['peer'],row['next_hop'],org))