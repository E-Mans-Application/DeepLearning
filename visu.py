
import sys
#import math

import numpy
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_pdf
import matplotlib.lines


inch = 2.54
picture_type = "pdf"


def show_params(weights, biases, title, path):
	
	if not title:
		return
	
	the_figure = matplotlib.figure.Figure(figsize=(28/inch,19/inch))
	matplotlib.backends.backend_pdf.FigureCanvas(the_figure)
	layout = matplotlib.gridspec.GridSpec(4, 3)
	layout.update(hspace = 0.45)#, wspace = 0.6, top = 0.95, bottom = 0.075, left = 0.1, right = 0.825)
	
	for a_group in sorted(weights.keys()):
		
		if "conv" in a_group:
			disambiguator = 0
			order = int(a_group.removeprefix("conv"))-1
		elif "fc" in a_group:
			disambiguator = 1
			order = int(a_group.removeprefix("fc"))-1
		else:
			continue
		
		the_plot = the_figure.add_subplot(layout[2*disambiguator, order])		# weights only
		
		the_plot.hist([x for x in weights[a_group].flatten() if abs(x) >  sys.float_info.epsilon], bins = 30)
		
		the_plot.set_title(f"weights distribution for {a_group}")
		
		the_plot = the_figure.add_subplot(layout[2*disambiguator+1, order])	# biases only
		
		the_plot.hist([x for x in biases[a_group].flatten() if abs(x) >  sys.float_info.epsilon], bins = 30)
		
		the_plot.set_title(f"biases distribution for {a_group}")
	
	the_figure.suptitle(title, fontsize = 'xx-large')
	
	# Draw all
	
	the_figure.canvas.draw()
	
	# save the figure
	
	the_figure.savefig(path / (title+"."+picture_type))


def show_accuracy(accuracies, path):
	
	the_title = "Accuracy of the prunned network"
	
	the_steps = sorted(accuracies.keys())
	the_accuracies = [accuracies[a_step] for a_step in the_steps]
	the_percentages = [a_step/100.0 for a_step in the_steps]
	print(the_percentages)
	print(the_accuracies)
	
	the_figure = matplotlib.figure.Figure(figsize=(28/inch,19/inch))
	matplotlib.backends.backend_pdf.FigureCanvas(the_figure)
	layout = matplotlib.gridspec.GridSpec(1, 1)
	#layout.update(hspace = 0.25)#, top = 0.95, bottom = 0.075, left = 0.1, right = 0.825, wspace = 0.6)
	
	the_plot = the_figure.add_subplot(layout[0,0])
	
	the_plot.plot(the_percentages, the_accuracies, marker='o', linestyle='')
	
	the_plot.set_xlim([-5/100.0, 105/100.0])
	
	the_plot.set_xlabel("Prunning (%)")
	
	the_plot.set_xticks(the_percentages)
	
	the_plot.set_xticklabels(the_percentages)
	
	the_plot.set_ylim([0, 100])
	
	#the_plot.set_yticks(the_percentages)
	
	#the_plot.set_yticklabels(the_accuracies)
	
	the_plot.set_ylabel("Accuracy (%)")
	
	the_plot.grid("both")
	
	the_plot.set_title(the_title, fontsize = 'xx-large')
	
	# Draw all
	
	the_figure.canvas.draw()
	
	# save the figure
	
	the_figure.savefig(path / (the_title+"."+picture_type))
	
	
	
	
	
