#! /usr/bin/env python2

"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

# Requirements: Numpy as PIL/Pillow
import numpy as np
from PIL import Image

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

# def depth_read(filename):
#     """ Read depth data from file, return as numpy array. """
    # try:
    #     f = open(filename, 'rb')

    #     check = np.fromfile(f, dtype=np.float32, count=1)[0]
    #     assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)
    #     width = np.fromfile(f, dtype=np.int32, count=1)[0]
    #     height = np.fromfile(f, dtype=np.int32, count=1)[0]
    #     size = width * height
    #     assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width, height)
    #     depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    #     return depth
    # except Exception as e:
    #     print("Error reading depth file:", e)
    #     return None



# def depth_read(filename):
#     """ Read depth data from file, return as numpy array. """
#     f = open(filename,'rb')
#     # check = np.fromfile(f,dtype=np.float32,count=1)[0]
    
#     check_array = np.fromfile(f, dtype=np.float32)
#     if len(check_array) > 0:
#         check = check_array[0]      
#     else:
#             # 处理数组为空的情况，例如给出警告或者进行其他处理
#         print("警告：check 数组为空，无法获取元素。")
#         return None

#     assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    
#     # width = np.fromfile(f,dtype=np.int32,count=1)[0]

#     width_array = np.fromfile(f,dtype=np.int32,count=1)
#     print("显示文件：",f )
#     print("显示宽度数组：",width_array ) 


    # if len(width_array) > 0:
    #     width = width_array[0]      
    # else:
    #         # 处理数组为空的情况，例如给出警告或者进行其他处理
    #     print("路径为：",width_array )    
    #     print("警告：width_array 数组为空，无法获取元素。")
    #     return None
    
    # # height = np.fromfile(f,dtype=np.int32,count=1)[0]
    
    # height_array = np.fromfile(f,dtype=np.int32)
    # if len(height_array) > 0:
    #     height = height_array[0]      
    # else:
    #         # 处理数组为空的情况，例如给出警告或者进行其他处理
    #     print("警告：height 数组为空，无法获取元素。")
    #     return None
    
    # size = width*height
    # assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    # depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    # return depth


def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    try:
        f = open(filename, 'rb')
        # 添加调试输出以查看文件指针位置
        # print("Current file position:", f.tell())
        
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        # print("Check value:", check)  # 添加调试输出以查看check的值
        
        assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)

        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        # print("Width value:", width)  # 添加调试输出以查看width的值
        
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        # print("Height value:", height)  # 添加调试输出以查看height的值
        
        size = width * height
        # print("Size:", size)  # 添加调试输出以查看size的值
        
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width, height)

        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        return depth
    except Exception as e:
        # print("An error occurred:", e)
        return None
    finally:
        f.close()  # 确保文件在退出函数之前被关闭
    


def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N