
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
        if (2*len(standard_colors) < len(signals)):
            for i in range(len(signals)-2*len(standard_colors)):
                standard_colors.append(standard_colors[i])
                standard_colors.append(standard_colors[i+1])

        colors = []
        for i in range(len(signals)):
            colors.append(standard_colors[2*i])
    elif isinstance(colors,str):
        colors = [colors]

    if (len(colors) < len(signals)):
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

def set_ROI(s, shape="circle"):
    """ Place an interactive Region Of Interest (ROI) on a signal [Based on create_ROI_on_DF from Thomas Aarholt]
    :type  s: hyperspy signal
    :param s: the signal to place the ROI on

    :type  shape: string
    :param shape: shape of the signal, circle/ring/rectangle
    
    :returns: interactive roi, and the sum of intensity within the roi (must be plottet separatly)
    """
    import hyperspy.api as hs
 
    if s.axes_manager.navigation_dimension < 2:
        axes = "sig"
        x_axis = s.axes_manager[s.axes_manager.signal_indices_in_array[1]]
        y_axis = s.axes_manager[s.axes_manager.signal_indices_in_array[0]]
    else:
        axes = "nav"
        x_axis = s.axes_manager[s.axes_manager.navigation_indices_in_array[1]]
        y_axis = s.axes_manager[s.axes_manager.navigation_indices_in_array[0]]


    if shape == "circle":
        x = x_axis.axis[round(x_axis.size/2)]
        y = y_axis.axis[round(y_axis.size/2)]

        r_outer = x_axis.axis[round(3*x_axis.size/4)]
    
        sroi = hs.roi.CircleROI(x, y, r=r_outer)
        s.plot()
        sroi= sroi.interactive(s) 
        ss = hs.interactive(f=sroi.sum, event=sroi.events.data_changed)
        
    elif shape == "ring":
        x = x_axis.axis[round(x_axis.size/2)]
        y = y_axis.axis[round(y_axis.size/2)]

        r_outer = x_axis.axis[round(4*x_axis.size/5)]
        r_inner = x_axis.axis[round(3*x_axis.size/4)]
    
        sroi = hs.roi.CircleROI(x, y, r=r_outer, r_inner=r_inner)
        s.plot()
        sroi= sroi.interactive(s) 
        ss = hs.interactive(f=sroi.sum, event=sroi.events.data_changed)
    else:
        if not shape == "rectangle":
            print("Did not recognize shape, using rectangle")
        x1 = x_axis.axis[1]
        x2 = x_axis.axis[round(x_axis.size/10)]
        y1 = y_axis.axis[1]
        y2 = y_axis.axis[round(y_axis.size/10)]

        sroi = hs.roi.RectangularROI(x1, y1, x2, y2)
        s.plot()
        sroi = sroi.interactive(s)
        ss = hs.interactive(f=roi_signal.sum, event=sroi.events.data_changed) 
        
    return sroi, ss