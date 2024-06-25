# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:10:49 2021

@author: Tobias Kallehauge
"""

from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
rcParams['font.size'] = 16
rcParams['axes.titlesize'] = 17
rcParams['text.usetex'] = True
rcParams['xtick.labelsize']=16
rcParams['ytick.labelsize']=16
rcParams['axes.labelsize']=16
rcParams['legend.fontsize']= 12
square_size = [5.5,5.5]
rec_size = [6,4]
diff = [-.1,0,-0.2,0]
square_bbox = Bbox([[diff[0],diff[1]],[square_size[0]-diff[2], square_size[1]- diff[3]]])
diff_rec = [-.1,-.2,.4,0]

rec_bbox = Bbox([[diff_rec[0],diff_rec[1]],[rec_size[0]-diff_rec[2], rec_size[1]- diff_rec[3]]])

# Add every font at the specified location
font_dir = ['cmu.serif-roman.ttf']
for font in font_manager.findSystemFonts(os.getcwd()):
    font_manager.fontManager.addfont(font)
rcParams['font.family'] = 'CMU Serif'



# some code for correct colorbarsize 
# cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.035,ax.get_position().height])
# fig.colorbar(im,cax = cax)

