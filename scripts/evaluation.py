import argparse
from pathlib import Path
import pickle

from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt

font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size': 15
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontP = FontProperties()
fontP.set_size('xx-small')

style = {'wald': '-.', 'naive': '--', 'ost': '-', 'split0.1': ':',
         'split0.3': ':', 'split0.5': ':', 'split0.8': ':'}
labels = {'wald': 'Wald', 'ost': 'ost', 'naive': 'naive', 'split0.1': 'split 0.1',
          'split0.3': 'split 0.3', 'split0.5': 'split 0.5', 'split0.8': 'split 0.8'}
markers = {'wald': 'x', 'ost': 'D', 'naive': '', 'split0.1': '*',
           'split0.3': '^', 'split0.5': 'o', 'split0.8': 'v'}


if __name__ == '__main__':

    #: Default directory containing the results
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dir',
        help="Directory containing the results",
        type=str,
        default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Directory containing the results
    data_dir = Path(args.dir)

    # Load results
    results = []
    for file_path in sorted(data_dir.glob('results_*.data')):
        with open(file_path, 'rb') as handle:
            results.append(pickle.load(handle))

    # Make plot
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
        plt.plot(samplesize, [1 - pow for pow in power[method]], label=labels[method], ls=style[method],
                 marker=markers[method], markersize=5, mfc='none')
    plt.legend(ncol=2)
    plt.xlabel('sample size n')
    plt.ylabel('Type-II error')
    plt.savefig('evaluation.pdf', bbox_inches='tight')
