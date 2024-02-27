../venv/bin/python advertisement_experiments.py --system vultr --mode anycast_unicast_withdrawal_experiment &


sleep 120

echo 'meep'

sudo client/peering prefix withdraw -m newyork 184.164.239.0/24 
sudo client/peering prefix withdraw -m newyork 184.164.238.0/24 
sudo client/peering prefix withdraw -m newyork 184.164.241.0/24 
