import dionysus as d
import diode as di
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time as t

#generates "num" points in a figure 8 (2 unit circles tangent to x-axis)
def figure8pts_example(num):
    array_figure8pts = np.zeros((num,2))
    rand_list = np.random.rand(num)*(np.pi*4)
    for i in range(num):
        array_figure8pts[i][0] = np.cos(rand_list[i])
        array_figure8pts[i][1] = np.sin(rand_list[i])
        if rand_list[i] < 2*np.pi:
            array_figure8pts[i][1] += 1
        else:
            array_figure8pts[i][1] += -1
    return array_figure8pts

#show an image of "pts" in R2
def display_pts(pts):
    fig, ax = plt.subplots()
    ax.scatter(pts[:,0], pts[:,1])
    plt.show() 
    
def rand_rot_pts(pts, dim):
    q, r = np.linalg.qr(np.random.normal(0, 100, (dim,dim)))
    print(q.shape)
    
    add_dim = dim-len(pts[1])
    fred = np.zeros((len(pts), add_dim))
    print( fred.shape )
    pts = np.concatenate((pts, fred), 1)
    pts = np.matmul(pts, q)
    return pts    

pts = figure8pts_example(200)
pts = rand_rot_pts(pts, 50)
print(pts)
#display_pts(pts)

start = t.time()
filt = d.fill_rips(pts, 2, 3.9) #get a rips filtration from arg1, with max dimension arg2 and max distance arg3
matrix = d.homology_persistence(filt)
dgms = d.init_diagrams(matrix, filt)
elapsed = str(t.time() - start)
print("rips computation in 50 dim took " +elapsed+ " seconds.")

pts = np.sqrt(50.0/3.0)*pts[:, :3]

start = t.time()
filt = d.fill_rips(pts, 2, 3.9) #get a rips filtration from arg1, with max dimension arg2 and max distance arg3
matrix = d.homology_persistence(filt)
dgms = d.init_diagrams(matrix, filt)
elapsed = str(t.time() - start)
print("rips computation in 3 dim took " +elapsed+ " seconds.")

start = t.time()
simplices = di.fill_alpha_shapes(pts)
filt = d.Filtration(simplices)
matrix = d.homology_persistence(filt)
dgms = d.init_diagrams(matrix, filt)
elapsed = str(t.time() - start)
print("alpha computation in 3 dim took " +elapsed+ " seconds.")

pts = np.sqrt(3.0/2.0)*pts[:, :2]

start = t.time()
filt = d.fill_rips(pts, 2, 3.9) #get a rips filtration from arg1, with max dimension arg2 and max distance arg3
matrix = d.homology_persistence(filt)
dgms = d.init_diagrams(matrix, filt)
elapsed = str(t.time() - start)
print("rips computation in 2 dim took " +elapsed+ " seconds.")

pts = np.concatenate((pts, np.zeros((len(pts), 1))), 1)

start = t.time()
simplices = di.fill_alpha_shapes(pts)
filt = d.Filtration(simplices)
matrix = d.homology_persistence(filt)
dgms = d.init_diagrams(matrix, filt)
elapsed = str(t.time() - start)
print("alpha computation in 2 dimtook " +elapsed+ " seconds.")


display_pts(pts)
print(dgms[0])
d.plot.plot_bars(dgms[1], show = True)

#pdb.set_trace()
