import numpy as np
import pandas as pd
from turbo_colormap import interpolate
import open3d as o3d
import os

def array_to_dataframe(a,sliceby=1):
    x, y, z = a.shape
    coords = np.column_stack((np.repeat(np.arange(z), y*x),  # z-column
                              np.tile(np.repeat(np.arange(z), y), x), # y-column
                              np.tile(np.tile(np.arange(z), y), x)))  # x-column
    coords = coords[::sliceby,:] #reduce number of points to show
    color = a.reshape(x*y*z, -1)
    color = color[::sliceby,:] #reduce number of points to show
    return coords.astype('uint16'), color

def scale_array(a): #convert to 0-1 range
    if np.ptp(a) != 0:
        return (a - np.min(a))/np.ptp(a)
    else:
        return a

def array_to_RGB(a,colormap="redness"):
    #input (n,1)
    a = scale_array(a) #scaled array to (0-1) range
    #output (n,3); col 1: redness(0-1), col2: greeness(0-1), col3: blueness(0-1)
    if colormap == "redness":
        colorscale = np.column_stack((a, np.zeros(len(a)), np.zeros(len(a))))
    if colormap == "turbo":
        colorscale = np.zeros((len(a),3))
        for ind, ele in enumerate(a):
            colorscale[ind,:] = np.array(interpolate(ele)).flatten()
    return colorscale

def numpy_to_ply(coordinates,color,file_name,overwrite,colormap = "turbo"):
    if os.path.isfile(file_name): #if file exists
        if not overwrite: #if I don't want to overwrite it
            return o3d.io.read_point_cloud(file_name)
        else: #if I do want to overwrite existing file
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coordinates)
            if colormap == "turbo":
                colors = array_to_RGB(color,colormap="turbo")
            elif colormap == "binary":
                colors = binary_colormap((color))
            elif colormap == "segmented":
                colors = segmented_colormap((color))
            pcd.colors = o3d.utility.Vector3dVector(colors) #how to add alpha/transparency??
            o3d.io.write_point_cloud(file_name,pcd)
            return o3d.io.read_point_cloud(file_name)
    else: #if file doesn't exists
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        if colormap == "turbo":
            colors = array_to_RGB(color,colormap="turbo")
        elif colormap == "binary":
            colors = binary_colormap((color))
        elif colormap == "segmented":
            colors = segmented_colormap((color))
        pcd.colors = o3d.utility.Vector3dVector(colors) #how to add alpha/transparency??
        o3d.io.write_point_cloud(file_name,pcd)
        return o3d.io.read_point_cloud(file_name)

def convert_to_structure(coord,color, show_pore_space = False):
    if show_pore_space == False:
        ind_remove = np.where(color != 0)[0]
        coord = np.delete(coord, ind_remove, axis = 0)
        color = np.delete(color, ind_remove, axis = 0)
        color += 1
    else:
        ind_remove = np.where(color == 0)[0]
        coord = np.delete(coord, ind_remove, axis = 0)
        color = np.delete(color, ind_remove, axis = 0)
    return coord,color

def binary_colormap(a):
    colorscale = np.zeros((len(a),3))
    return colorscale

def segmented_colormap(a):
    colorscale = np.zeros((len(a),3)) #default is black
    for i,k in enumerate(a):
        if k == 4:
            colorscale[i] = (0.2,0.3,0.3) #gray
        elif k == 3:
            colorscale[i] = (1,0,0) #red
        elif k == 2:
            colorscale[i] = (1,0.6,0) #orange (1,1,1) #white
        elif k == 1:
            colorscale[i] = (0,0,1) #blue
    return colorscale

def only_percolating_path(coord,color,):
    ind_remove = np.where(np.logical_and(color != 3,color != 4))[0]
    coord = np.delete(coord, ind_remove, axis = 0)
    color = np.delete(color, ind_remove, axis = 0)
    return coord,color

def high_velocity_region(coord,color,):
    ind_remove = np.where(np.logical_or(color == 1,color == 0))[0]
    coord = np.delete(coord, ind_remove, axis = 0)
    color = np.delete(color, ind_remove, axis = 0)
    return coord,color

if __name__ == "__main__":
    a = np.array([[[0, 0, 0],
                   [0, 0, -1],
                   [0, 0, 2]],
                  [[0, 1, 0],
                   [0, 1, 1],
                   [0, 1, 2]],
                  [[0, 2, 0],
                   [0, 2, 1],
                   [0, 2, 2]]])
    out_coords, out_color = array_to_dataframe(a)
    assert len(out_coords) == len(a.flatten())
    scaled_a = scale_array(out_color)
    assert np.logical_and(np.all(scaled_a >= 0), np.all(scaled_a <= 1))
    a_colorscale = array_to_RGB(out_color, colormap="redness")
    assert np.all(a_colorscale[:,0] >= 0) and np.all(a_colorscale[:,0] <= 1)
    assert np.all(a_colorscale[:,1] == 0) and np.all(a_colorscale[:,2] == 0)
    a_colorscale = array_to_RGB(out_color, colormap="turbo")
    assert np.all(a_colorscale >= 0) and np.all(a_colorscale <= 1)
    pointcloud = numpy_to_ply(out_coords,out_color, file_name="example.ply")
    assert os.path.isfile("example.ply")
    binarized_coord, binarized_color = convert_to_structure(out_coords,out_color)
    assert len(out_coords) - len(binarized_coord) == len(np.where(a != 0)[0])

def add_edges(a):
    x_max,y_max,z_max = a.shape
    x_max -= 1
    y_max -= 1
    z_max -= 1
    a[:,0,0] = 4
    a[:,0,z_max] = 4
    a[:,y_max,0] = 4
    a[:,y_max,z_max] = 4
    a[0,:,0] = 4
    a[0,:,z_max] = 4
    a[x_max,:,0] = 4
    a[x_max,:,z_max] = 4
    a[0,0,:] = 4
    a[0,y_max,:] = 4
    a[x_max,0,:] = 4
    a[x_max,y_max,:] = 4
    return a

def make_movie(pcd):
    vis = o3d.visualization.Visualizer()
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(3.0,0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)
