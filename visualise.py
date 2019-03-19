# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:19:23 2016

@author: tvzyl
"""
import mvn

from numpy import ceil, rot90, reshape, mgrid, c_
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, subplot, plot, suptitle
from matplotlib.pyplot import title, subplots_adjust, colorbar, axes
from matplotlib import cm
import seaborn as sns
sns.set()
sns.set_style('white')

import pandas as pd

def getGrid(dataFrame, getDensity, res_x=100, res_y=None, buff=0.2):
    xmin = dataFrame[0].min() - abs(dataFrame[0].min()*buff)
    xmax = dataFrame[0].max() + abs(dataFrame[0].max()*buff)
    ymin = dataFrame[1].min() - abs(dataFrame[1].min()*buff)
    ymax = dataFrame[1].max() + abs(dataFrame[1].max()*buff)
    if res_y is None:
        res_y = int(float(ymax-ymin)/(float(xmax-xmin)/float(res_x)))
    grid_size_x = (res_x)*1.00j
    grid_size_y = (res_y)*1.00j
    area = (xmax-xmin)/(grid_size_x.imag-1)*(ymax-ymin)/(grid_size_y.imag-1)
    X, Y = mgrid[xmin:xmax:grid_size_x, ymin:ymax:grid_size_y]
    points = pd.DataFrame(c_[X.ravel(), Y.ravel()])
    GRID_I = rot90(reshape(getDensity(points).T, X.shape)*area)
    return GRID_I, (xmin, xmax, ymin, ymax), area, (res_y, res_x)

def plotDensity(dataFrame, kdename, getDensity, total_plots=1, plot_number=0, figname="plot"):    
    GRID_I, (xmin, xmax, ymin, ymax), area, (res_y, res_x) = getGrid(dataFrame, getDensity)
    if total_plots>1:
        f = figure(figname)
        if plot_number==0:
            f.clear()
        p = subplot(3, ceil(float(total_plots+1)/3.0), plot_number+1)
        p.clear()
    else:
        f = figure("KDE %s"%kdename)
        f.clear()
#    suptitle(figname)
    im  = imshow(GRID_I,
                 cmap=cm.gist_earth_r,
                 interpolation='none',
                 extent=[xmin, xmax, ymin, ymax])
                 #,vmin=0, vmax=1)
    plot(dataFrame[0], dataFrame[1], 'r.', markersize=2)
    if plot_number%3 != 0:
        plt.setp(p.get_yticklabels(), visible=False)
    if plot_number < 6:
        plt.setp(p.get_xticklabels(), visible=False)    
    f.subplots_adjust(wspace=0.025)
    f.subplots_adjust(hspace=0.025)
#    plot(tests.x, tests.y, 'k.', markersize=2)
#    title("%s"%['Actual',
#                    'Silverman Pilot CVLS $\mathcal{H}_3$', 
#                    'Breiman CVLS K-NN Kung',
#                    'Wu Cluster CVLS',
#                    'Terrell K-NN Sqrt',
#                    'Loftsgaarden K-NN Hansen',
#                    'Gaussian K-NN Sqrt',
#                    'Lima Quadratic',
#                    'VdWalt ML BackSub'][plot_number])
#    title("%s"%['Actual',
#                    'Silverman Pilot CVLS $\mathcal{H}_2$', 
#                    'Breiman CVLS K-NN Kung',
#                    'Wu Cluster CVLS',
#                    'Terrell K-NN Hansen',
#                    'Loftsgaarden K-NN Kung',
#                    'Gaussian K-NN Sqrt',
#                    'Lima Entropy',
#                    'VdWalt ML BackSub'][plot_number])
#    title("KDE %s"%kdename)
#    if total_plots==1:        
#        colorbar(format='%.2f')
#    else:
#        subplots_adjust(left=0.05, right=0.9)
#        cax = axes([0.95, 0.1, 0.025, 0.8])
#        colorbar(cax=cax)    