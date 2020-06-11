import pickle
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 15
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontP = FontProperties()
fontP.set_size('xx-small')


results = []
runs = 1000
for exp in range(0,35):
    with open('results_' + str(exp) + '.data', 'rb') as handle:
        results.append(pickle.load(handle))

style = {'wald': '-.', 'naive': '--', 'ost': '-', 'split0.1': ':', 'split0.3': ':', 'split0.5': ':', 'split0.8': ':'}
labels = {'wald': 'Wald', 'ost': 'ost', 'naive': 'naive', 'split0.1': 'split 0.1', 'split0.3': 'split 0.3', 'split0.5': 'split 0.5', 'split0.8': 'split 0.8'}
samplesize = []
methods = []

power = {}
for res in results:
    if (res['samplesize'] in samplesize) == False:
        samplesize.append(res['samplesize'])
    method = res['method']
    if method in power.keys():
        power[method].append(res['power'])
    else:
        power[method] = [res['power']]
        methods.append(method)
for method in methods:
    plt.errorbar(samplesize, [1-pow for pow in power[method]],  label=labels[method], ls=style[method])
plt.legend(ncol=2)
plt.xlabel('sample size n')
plt.ylabel('Type-II error')
plt.savefig('evaluation.pdf', bbox_inches='tight')
