
from pyeels.orbital import Orbital
from pyeels.atom import Atom
from pyeels.crystal import Crystal


from pyeels.band import Band
from pyeels.brillouinzone import BrillouinZone

from pyeels.tightbinding import TightBinding
from pyeels.parabolicband import ParabolicBand
from pyeels.nonparabolicband import NonParabolicBand

from pyeels.eels import EELS



import matplotlib.pyplot as plt
def plot_signals(signals, colors=None, linestyles=None, plotstyle=None, fill=None, linewidth=None, labels=None):
    
    s = signals[0]
    x_label = "{} [{}]".format(s.axes_manager.signal_axes[0].name,s.axes_manager.signal_axes[0].units)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensity [arb.]")

    custom_labels = False
    if labels:
    	if (len(labels) == len(signals)):
    		custom_labels = True

    
    if not linestyles:
        linestyles = []
        for i in range(len(signals)):
            linestyles.append('-')
    elif isinstance(linestyles,str):
        linestyles = [linestyles]

    if (len(linestyles) < len(signals)):
        for i in range(len(signals)-len(linestyles)):
            linestyles.append(linestyles[i])

    #
    if fill is None:
        fill = []
        for i in range(len(signals)):
            fill.append(False)
    elif isinstance(fill,bool):
        fill = [fill]

    if (len(fill) < len(signals)):
        for i in range(len(signals)-len(fill)):
            fill.append(fill[i])

    if not colors:
        standard_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
        colors = []
        for i in range(len(signals)):
            colors.append(standard_colors[2*i])
    elif isinstance(colors,str):
        colors = [colors]

    if (len(colors) < len(colors)):
        for i in range(len(signals)-len(colors)):
            colors.append(colors[i])

    if not linewidth:
        linewidth = 2;

    for i,s in enumerate(signals):
        x = s.axes_manager.signal_axes[0].axis
        y = s.sum().data

        if custom_labels:
        	label = labels[i]
        else:
	        label = s.metadata['General']['title']
        ax.plot(x,y,linestyles[i],color=colors[i], linewidth=linewidth, label=label)
        if fill[i]:
            ax.fill_between(x,0,y,facecolor=colors[i],alpha=0.4)
    return fig, ax            
