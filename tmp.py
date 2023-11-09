import pickle

d = pickle.load(open('cache/popp_failure_latency_comparison_actual.pkl','rb'))
sparse_adv = d['adv'][0]['sparse']
print(d['latencies'][0]['sparse'])