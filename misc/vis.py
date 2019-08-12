import seaborn as sns
import matplotlib.pyplot as plt
import pylab as plot

sns.set_style("whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 14}
plot.rcParams.update(params)


def vis(table):
    fig, ax = plt.subplots()

    sns.heatmap(table, annot=True, fmt=".2f", cmap="YlGnBu", center=0.7)
    plt.xlabel(r'\textbf{Agent $2$ Location}', size=16)
    plt.ylabel(r'\textbf{Agent $1$ Location}', size=16)
