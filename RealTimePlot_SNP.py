import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
#import cmdLineArgument as cla

anticipation_time = '300'
coordinate = 'yaw' 
mode = 'movement'

# initiate the Windows of real time plot
app = QtGui.QApplication([])                                # Create the application
pg.setConfigOption('background', 'k')

if mode == "movement":
	plot_title = 'Real Time Plot'
	subtitle = 'Orientation'
elif mode == "error":
	plot_title = 'Real Time Error Plot'
	subtitle = 'Error'
	

win = pg.GraphicsWindow(title=plot_title)			# Create empty space in the window
p = win.addPlot(title= subtitle + '(' + coordinate + '), Anticipation Time: ' + str(anticipation_time) + ' ms') 			# Add the plot to be drawn

if coordinate == "yaw":
	p.setYRange(-100,100,padding = 0)
else:
	p.setYRange(-60,60,padding = 0)

p.addLegend()
p.setLabel('bottom', text='time(s)')
p.setLabel('left', text='angles(deg)')
stream_0 = p.plot(pen=(255, 255, 255), name='actual')                	 # plot color
stream_1 = p.plot(pen=(255,0, 0), name='SNP')
stream_2 = p.plot(pen=(0,255,0), name='Context')

window_width = 200
stream0 = np.zeros(shape=(window_width, 2))
stream1 = np.zeros(shape=(window_width, 2))
stream2 = np.zeros(shape=(window_width, 2))

	
def RealTimePlot(x, y):
	# this part is for "sliding" the plot
	stream0[:-1] = stream0[1:]
	stream1[:-1] = stream1[1:]
	stream2[:-1] = stream2[1:]
	#this part is for "filling the gap" left by sliding
	stream0[-1] = np.array([x,y[0]])
	stream1[-1] = np.array([x,y[1]])
	stream2[-1] = np.array([x,y[2]])

	
	# this part is for "real time plotting"
	# print(stream0)
	stream_0.setData(stream0)
	stream_1.setData(stream1)
	stream_2.setData(stream2)

	QtGui.QApplication.processEvents()