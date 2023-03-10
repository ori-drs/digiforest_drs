#!/usr/bin/env python
import numpy as np
import pcl

import shutil
import sys
import os

def cropBox(cloud, midpoint, boxsize):
    clipper = cloud.make_cropbox()
    outcloud = pcl.PointCloud()
    tx = 0
    ty = 0
    tz = 0
    clipper.set_Translation(tx, ty, tz)
    rx = 0
    ry = 0
    rz = 0
    clipper.set_Rotation(rx, ry, rz)
    minx = midpoint[0] - boxsize/2
    miny = midpoint[1] - boxsize/2
    minz = -1000000000
    mins = 0
    maxx = midpoint[0] + boxsize/2
    maxy = midpoint[1] + boxsize/2
    maxz = 10000000000
    maxs = 0
    clipper.set_MinMax(minx, miny, minz, mins, maxx, maxy, maxz, maxs)
    outcloud = clipper.filter()
    return outcloud

def filterUpNormal(x, upthreshold, keepUp=True):
    # filter out not-up points from PCLXYZNormal
    xy_dat = x.to_array()
    if (keepUp):
        x_displayed = xy_dat[(xy_dat[:,5] > upthreshold)]
    else:
        x_displayed = xy_dat[(xy_dat[:,5] < upthreshold)]
    x.from_array(x_displayed)
    return x

def getMinimumHeight(pBox):
    heights = pBox.to_array()[:,2]
    heights_a = np.array(heights)
    return np.min(heights_a)


def getRobustHeight(pBox):
    heights = pBox.to_array()[:,2]
    xmean = np.mean(pBox,0)
    mean_height = xmean[2]

    m = np.percentile(heights, 10, axis=0)

    return m

    #
    height_bins = np.arange(mean_height-3,mean_height+3,0.01)
    hist, bin_edges = np.histogram(heights,height_bins)
    con = np.array([1, 2, 3, 4, 3, 2, 1])
    con = con / np.sum(con)

    hist_smooth = np.convolve(hist, con, mode='same')
    idx=np.argmax(hist_smooth)
    robust_height = height_bins[idx]

    return robust_height

def getTerrainHeight(p):
    # input is a PCLXYZ
    # output is an np.array of points [x,y,z] 

    #p = filterUpNormal(p)
    print (p.size)

    # number of cells is (cloud_boxsize/cell_size) squared
    cloud_boxsize = 80
    cell_size = 2.0
    cloud_midpoint_round = np.round(np.mean(p,0)/cell_size)*cell_size

    d_x=np.arange(cloud_midpoint_round[0]-cloud_boxsize/2,cloud_midpoint_round[0]+cloud_boxsize/2,cell_size)   
    d_y=np.arange(cloud_midpoint_round[1]-cloud_boxsize/2,cloud_midpoint_round[1]+cloud_boxsize/2,cell_size)
    X = np.empty(shape=[0, 3])

    for xx in d_x:
        for yy in d_y:
            cell_midpoint = np.array([xx,yy,0])
            pBox = cropBox(p, cell_midpoint, cell_size)
            #print (xx , " ", yy, " ", pBox.size)
            if (pBox.size<100):
                height = np.NAN # 10.0
            else:
                height = getMinimumHeight(pBox)
                #height = getRobustHeight(pBox)
                #xmean = np.mean(pout,0)
                #height = xmean[2]
            X = np.append(X, [[xx, yy, height]], axis=0)

    return X

def load(filename):
    p = pcl.load(filename)
    return p

def generate_height_map(filename):
    cloud_pc = pcl.PointCloud_PointNormal()
    cloud_pc._from_pcd_file(filename.encode('utf-8'))

    # remove non-up points
    cloud = filterUpNormal(cloud_pc, 0.95)

    # drop from xyznormal to xyz
    array_xyz = cloud.to_array()[:, 0:3]
    cloud = pcl.PointCloud()
    cloud.from_array(array_xyz)

    # get the terrain height
    heights_array_raw = getTerrainHeight(cloud)
    pcd = pcl.PointCloud()
    pcd.from_list(heights_array_raw)
    pcd.to_file(b'/tmp/height_map.pcd')

def generate_height_maps(directory, output_dir):

    if not os.path.isdir(directory):
        print("Cannot find the folder", directory)
        return

    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for file in files:
        filename, extension = os.path.splitext(file)

        if extension != ".pcd":
            continue

        print("Processing", file)
        filename, _  = os.path.splitext(os.path.basename(file))
        s = filename.split("_") # assuming a file with a name like cloud_1663668488_446585000.pcd

        if len(s) != 3:
            continue

        height_map_filename = "height_map_"+s[1]+"_"+s[2]+".ply"

        generate_height_map(file)
        os.system("rosrun digiforest_drs generate_mesh") #TODO improve

        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                print("Cannot create", output_dir)
                return
        shutil.copyfile('/tmp/height_map.ply', os.path.join(output_dir, height_map_filename))

