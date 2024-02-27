provider_popps = []
for row in open('cache/vultr_provider_popps.csv' ,'r'):
	provider_popps.append(tuple(row.strip().split(',')))
for row in open('cache/already_completed_popps.csv','r'):
	pop,peer = row.strip().split(',')
	if (pop,peer) in provider_popps:
		print("First instance is {}".format((pop,peer)))
		break
