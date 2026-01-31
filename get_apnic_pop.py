from helpers import *
import csv
import sys
sys.path.append("../peering_measurements")
import generic_measurement_utils

def load_all_ip_addresses():
	ret = {}
	## Obtain this file with awk -F',' '{print $2}' vultr_ingress_latencies_by_dst.csv | sort -u > vultr_all_dsts.csv
	fn = os.path.join('cache','vultr_all_dsts.csv')
	for row in open(fn,'r'):
		ret[row.strip()] = None
	return ret

def load_asn_to_apnic_pop():
	## path to APNIC to population dump
	file_path = '../peering_measurements/cache/aspop.csv'

	# Dictionary to store the mapping
	asn_to_users = {}

	try:
		with open(file_path, mode='r') as f:
			# csv.reader automatically handles the quotes around columns
			reader = csv.reader(f)
			
			for row in reader:
				# Ensure the row isn't empty
				if not row:
					continue
				
				# 1. Get the 2nd column (index 1)
				# 2. Remove "AS" from the start of the string
				asn_raw = row[1]
				asn_clean = asn_raw.replace("AS", "")
				
				# 3. Get the last column (index -1) and convert to integer
				try:
					user_count = int(row[-1])
				except ValueError:
					# This handles cases where the last column might be a header or empty
					continue
					
				# 4. Map the cleaned ASN to the user count
				asn_to_users[asn_clean] = user_count

		# Displaying the first few results as a test
		print(f"{'ASN':<10} | {'Users':<15}")
		print("-" * 28)
		for asn, count in list(asn_to_users.items())[:10]:
			print(f"{asn:<10} | {count:<15,}")

	except FileNotFoundError:
		print(f"Error: The file '{file_path}' was not found.")

	return asn_to_users

def main():
	out_fn = os.path.join('cache', 'vultr_all_dsts_asn_apnic_pop.csv')
	## 1) get all ip addresses in vultr ingress measurements
	all_ips = load_all_ip_addresses()
	all_24s = list(set(ip32_to_24(ip) for ip in all_ips))
	## 2) map those ip addresses to asns
	all_24s_to_lookup = []
	print("{} /24s, {} to lookup".format(len(all_24s), len(all_24s_to_lookup)))
	## 3) map asn to apnic vol
	asn_to_apnic_pop_tmp = load_asn_to_apnic_pop()
	asn_to_apnic_pop = {}
	## 4) map both of those asn datasets to the org-like ones we have elsewhere
	utils = generic_measurement_utils.AS_Utils_Wrapper()
	utils.check_load_siblings()
	utils.check_load_ip_to_asn()
	utils.check_load_as_rel()
	utils.update_cc_cache()
	utils.lookup_asns_if_needed(all_24s)
	for asn,pop in asn_to_apnic_pop_tmp.items():
		try:
			asn_to_apnic_pop[utils.parse_asn(asn)] += pop
		except KeyError:
			asn_to_apnic_pop[utils.parse_asn(asn)] = pop
	# 5) save IP, asn, n_users file
	covered_pop = {}
	with open(out_fn, 'w') as f:
		for ip in all_ips:
			ip24 = ip32_to_24(ip)
			try:
				asn = utils.parse_asn(ip24)
			except KeyError:
				continue
			try:
				pop = asn_to_apnic_pop[asn]
				covered_pop[asn] = pop
			except KeyError:
				continue
			f.write("{},{},{}\n".format(ip,asn,pop))
	covered_pop = sum(list(covered_pop.values()))
	pct_coverage = covered_pop*100.0 / float(sum(list(asn_to_apnic_pop.values())))
	print("We have measurements to addresses representing {} percent of addresses".format(round(pct_coverage,2)))


if __name__ == "__main__":
	main()


