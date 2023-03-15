import re
from subprocess import call, check_output
def delete_table_rules():
	for i in range(6):
		table_i = 201 + i
		try:
			active_rule = check_output("sudo ip route show table {}".format(table_i), shell=True).decode().strip()
		except:
			continue
		if active_rule == "": continue
		rulere = re.search("default via (.+) dev (.+)", active_rule)
		nh, tap = rulere.group(1), rulere.group(2)
		call("sudo ip route del default via {} dev {} table {}".format(nh,tap,table_i),shell=True)


	for line in check_output("sudo ip rule show",shell=True).decode().split('\n'):
		line = line.strip()
		if line == "": continue
		prio = line.split(":")[0]
		if int(prio) > 19000 and int(prio) < 20000:
			call("sudo ip rule del priority {}".format(prio), shell=True)

if __name__ == "__main__":
	delete_table_rules()