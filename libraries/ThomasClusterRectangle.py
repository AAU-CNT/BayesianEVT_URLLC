# Simulate a Thomas cluster process on a rectangle.
# Author: H. Paul Keeler, 2018.
# Website: hpaulkeeler.com
# Repository: github.com/hpaulkeeler/posts
# For more details, see the post:
# hpaulkeeler.com/simulating-a-thomas-cluster-point-process/

import numpy as np;  # NumPy package for arrays, random number generation, etc
import pandas as pd 

def r_Thomas(xMin,xMax,yMin,yMax, lambdaParent,lambdaDaughter,sigma,
             extent = 6): 
    """
    Simulate Thomas cluster point process
    
    Average number of simulated points is A*lambdaParent*lambdaDaughter
    where A = (xMax - xMin)*(yMax - yMin) is the area. 

    Parameters
    ----------
    xMin : float
        Minimum x-value 
    xMax : float
        Maximum x-value 
    yMin : float
        Minimum y-value 
    yMax : float
        Maximum y-value
    lambdaParent : float
        Density of parent distribution (cluster centers). Should be positive.
    lambdaDaughter : float
        Density of daughter distribution (clusters). Should be positive. 
    sigma : float
        Standard deviation within each cluster

    Returns
    -------
    s : ndarray
        Simulated points  as matrix: (x,y) in each row. 
    """
    
 
    # Extended simulation windows parameters
    rExt=extent*sigma; # extension parameter 
    # for rExt, use factor of deviation sigma eg 5 or 6
    xMinExt = xMin - rExt;
    xMaxExt = xMax + rExt;
    yMinExt = yMin - rExt;
    yMaxExt = yMax + rExt;
    # rectangle dimensions
    xDeltaExt = xMaxExt - xMinExt;
    yDeltaExt = yMaxExt - yMinExt;
    areaTotalExt = xDeltaExt * yDeltaExt;  # area of extended rectangle
    
    # Simulate Poisson point process for the parents
    numbPointsParent = np.random.poisson(areaTotalExt * lambdaParent);# Poisson number of points
    # x and y coordinates of Poisson points for the parent
    xxParent = xMinExt + xDeltaExt * np.random.uniform(0, 1, numbPointsParent);
    yyParent = yMinExt + yDeltaExt * np.random.uniform(0, 1, numbPointsParent);
    
    # Simulate Poisson point process for the daughters (ie final poiint process)
    numbPointsDaughter = np.random.poisson(lambdaDaughter, numbPointsParent);
    numbPoints = sum(numbPointsDaughter);  # total number of points
    
    # Generate the (relative) locations in Cartesian coordinates by
    # simulating independent normal variables
    xx0 = np.random.normal(0, sigma, numbPoints);  # (relative) x coordinaets
    yy0 = np.random.normal(0, sigma, numbPoints);  # (relative) y coordinates
    
    # replicate parent points (ie centres of disks/clusters)
    xx = np.repeat(xxParent, numbPointsDaughter);
    yy = np.repeat(yyParent, numbPointsDaughter);
    
    # translate points (ie parents points are the centres of cluster disks)
    xx = xx + xx0;
    yy = yy + yy0;
    
    # thin points if outside the simulation window
    booleInside = ((xx >= xMin) & (xx <= xMax) & (yy >= yMin) & (yy <= yMax));
    # retain points inside simulation window
    xx = xx[booleInside];  
    yy = yy[booleInside];
    
    # shape into matrix
    s = np.vstack((xx,yy)).T
    
    return(s)

# I wrote this one! 
def r_Thomas_discrete(xMin,xMax,yMin,yMax, N, grid_size, 
                      lambdaParent = 0.005,sigma = 3.5, extent = 6):
    """
    Simulate Thomas process, then thin process to a fixed number of points only
    at grid locations. 

    Parameters
    ----------
    grid_size : float
        Grid size 
    N : int
        Number of points
    
    See r_Thomas for other parameters
    
    Returns
    -------
    s : ndarray
        Simulated points  as matrix: (x,y) in each row. 
    """
    
    if 0.9*((xMax - xMin)/grid_size)**2 < N:
        raise(ValueError('Too many points for grid size'))
    
    A = (xMax - xMin)*(yMax - yMin)
    lambdaDaughter = 10*N/(A*lambdaParent) # makes 10 times the number of 
    #points needed on average 
    s = r_Thomas(xMin,xMax,yMin,yMax,lambdaParent,lambdaDaughter,sigma)
    
    s = pd.DataFrame(s)
    s_tmp = s.copy()
    
    # Sample from s in grid
    idx = []
    count = 0
    iter_max = 1000
    while len(idx) < N:
        count += 1
        idx.extend(np.random.choice(s_tmp.index,size = N - len(idx), 
                                    replace = False))
        
        idx = list((s.loc[idx]/grid_size).round().astype('int').drop_duplicates().index)
    
        s_tmp = s.drop(idx)
        if count > iter_max:
            raise(ValueError('Unable to find points, try with fewer'))
      
    s = ((s.loc[idx]/grid_size).round()*grid_size).astype('int')
    s['z'] = 1.5
    s = s.values
    return(s)
