# this lib, together with the preprocess_mice_trials script
# is used to process the raw excel files that have the tracking
# of the mouse in the arena looking for the hidden food
#
# the experiments were originally performed by Kelly Xu from Len Maler lab
#
# the original excel files are not going to be included in this repository
#
# instead, we provide only the processed files (in .mat format)
# that contain all the relevant information for the trajectories
# extracted from the excel files...

import math
import numpy
import itertools
import collections
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageColor
import modules.helper_func_class as misc
import modules.process_mouse_trials_lib as plib
from collections import deque # for cluster algorithm
import warnings
#import cv2
#import warnings

#warnings.filterwarnings('error')

def find_arena_in_image(input_image_or_path,mark_img=None,arena_color=None,**find_arena_in_image_circle_detection_params):
    """
    given an input image, this function tries to detect the arena boundaries in it
    
    the image can be a marks file, with the arena marked in a color given in mark_scheme
    OR
    the image is a preprocessed arena picture (with contrast enhanced) directly from experiments;
    in which case, the parameters of the function
    find_arena_in_image_circle_detection
    must also be provided

    returns the arena center and radius, (in pixels):
        x,y,r
    """
    if (type(arena_color) is type(None)) or ( (type(mark_img) is str) and (len(mark_img) == 0) ) or (type(mark_img) is type(None)):
        return find_arena_in_image_circle_detection(input_image_or_path,**find_arena_in_image_circle_detection_params)
    #mark_scheme = misc.structtype()
    #if not mark_scheme.IsField('arena_boundary'):
    #    raise ValueError("'mark_scheme must contain a field 'arena_boundary' with the color of the arena boundary in the given image")
    c = find_clusters(_get_PIL_img(mark_img),threshold=0.5,find_less_than_threshold=False,join_clusters=True,color=arena_color)
    if len(c) == 0:
        warnings.warn('No circle was found with these parameters, returning nan')
        return numpy.nan,numpy.nan,numpy.nan
    #if len(c) > 1:
    #    warnings.warn('More than 1 circle found, returning center of the one with largest radius')
    #    c = sorted(c,key=lambda s: s[-1])[0]
    else:
        c = c[0]
    return c

def _get_rgb_color(color):
    """
    gets the color

    if str -> it must be a 6-character Hexadecimal color string, e.g. FF0000, or #FF0000
    if tuple, ndarray, list -> returns an ndarray with each entry scaled to 0...255 int

    color must not contain alpha values
    """
    if type(color) is str:
        return PIL.ImageColor.getrgb(color) #numpy.array([int(color[i:i+2], 16) for i in (0, 2, 4)])
    return numpy.asarray(color,dtype=int)

def _get_color_norm(color_list):
    if not(type(color_list) is numpy.ndarray):
        color_list = numpy.asarray(color_list)
    if color_list.ndim == 1: # only one color is given
        if color_list.size in [3,4]:
            return numpy.linalg.norm(color_list[:3])
        else:
            return color_list.flatten()
    elif color_list.ndim == 3: # a picture was given
        return numpy.linalg.norm(color_list[:,:,:3],axis=2).flatten()
    else:
        if ((color_list.ndim == 2) and (color_list.shape[1]==2)): # col1 -> gray, col2 -> alpha
            return color_list[:,0]
        #if ((color_list.ndim == 2) and (color_list.shape[1] in [3,4])):
        return numpy.linalg.norm(color_list[:,:3],axis=1)
        #else:
        #    color_list.flatten()
    #T = img.flatten() if img.ndim <= 2 else (numpy.linalg.norm(img,axis=2).flatten() if (img.shape[2] <= 3) else numpy.linalg.norm(img[:,:,:3],axis=2).flatten())

def find_clusters(img,threshold=0.5,find_less_than_threshold=False,join_clusters=False,color=None):
    """
    returns clusters of pixels from img (must be connex pixels)
    if color is given, only return clusters with the given color

    img                      -> PIL image
    threshold                -> color threshold for normalized colors to 0,1 (0->min color norm),(1->max color norm) (find clusters with pixel color norm greater than threshold)
    find_less_than_threshold -> good for finding black clusters instead of white
    join_clusters            -> if True, all found clusters are considered part of the same cluster
    color                    -> if given, only finds clusters of pixels colored with color    

    returns
        C -> [ (x0,y0,r0), ... ]: list of cluster centers and radii
    """
    if type(img) is numpy.ndarray:
        img = PIL.Image.fromarray(img.astype('uint8'),'RGB')

    has_color = not(type(color) is type(None))
    if has_color:
        img = select_color(img,color,(0,0,0),(255,255,255))
        threshold = 0.5
        find_less_than_threshold = False

    #N = numpy.prod(img.size) # total amount of pixels 
    #get_color_scalar = lambda s: numpy.linalg.norm(s[:,:3],axis=1) if ((s.ndim > 1) and (s.shape[1]>=3)) else (s[:,0] if ((s.ndim > 1) and (s.shape[1]==2)) else s.flatten())
    # gets all pixel which color is greater than threshold to act as seeds
    #if type(img) is numpy.ndarray:
    #    T = img.flatten() if img.ndim <= 2 else (numpy.linalg.norm(img,axis=2).flatten() if (img.shape[2] <= 3) else numpy.linalg.norm(img[:,:,:3],axis=2).flatten())
    #    h,w = img.shape
    T = _get_color_norm(numpy.array(img.getdata()))
    w,h = img.size
    T = T/numpy.max(T)
    if find_less_than_threshold:
        T = 1.0-T
    seeds = numpy.nonzero(T > threshold)[0]
    assigned = numpy.zeros(T.size,dtype=bool)
    all_clusters = []
    for s in seeds:
        if not assigned[s]:
            c = grow_cluster_BFS(T,s,threshold,w,h) # generates a cluster of all the elements connected to i using Breadth First Search (BFS)
            all_clusters.append(c)
            for k in c:
                assigned[k] = True
    if join_clusters:
        cluster_center_radius = [ _calc_center_radius_pixel_set(itertools.chain(*all_clusters),w,h) ]
    else:
        cluster_center_radius = [ _calc_center_radius_pixel_set(c,w,h) for c in all_clusters ]
    return cluster_center_radius

def grow_cluster_BFS(T,k,threshold,w,h):
    """
    grows a connex cluster in T from k
    T is a flattened pixel list (ndarray), such that each index in the list is given by
    n = y + x*w, where x and y are the pixel coordinates in the original image T;
    neighbors of n: x+1,y -> right
                    x-1,y -> left
                    x,y+1 -> bottom
                    x,y-1 -> top
    returns a list of pixel indices, each index given by their position from the function
        _img_pixel_index()
    """
    N = T.size
    visited = numpy.zeros(N)
    q = deque([k])
    c = []
    while len(q) > 0:
        i = q.popleft()
        n = _get_neighbor_pixels(i,w,h)
        q.extendleft([ p for p in n if (not visited[p]) and (T[p]>=threshold) ])
        #q = [n(~visited(n)), q];
        visited[i] = 1
        c.append(i)
    return c

def _calc_center_radius_pixel_set(c,w,h):
    pos = numpy.array([_get_pixel_position(k,w,h) for k in c],dtype=float)
    pos_avg = numpy.round(numpy.mean(pos,axis=0),decimals=0).astype(int)
    r = int(numpy.ceil(numpy.max(numpy.max(pos,axis=0)-numpy.min(pos,axis=0))/2.0))
    return int(pos_avg[0]),int(pos_avg[1]),r

def _get_pixel_position(k,w,h):
    """converts k to x,y pixel position"""
    x = ((((k+1)%(h*w)) - 1)%w)
    y = ((int((k - x) / w))%h)
    return x,y

def _get_neighbor_pixels(k,w,h):
    x,y = _get_pixel_position(k,w,h)
    n = []
    if x > 0:
        n.append(_img_pixel_index(x-1,y,w))
    if x < (w-1):
        n.append(_img_pixel_index(x+1,y,w))
    if y > 0:
        n.append(_img_pixel_index(x,y-1,w))
    if y < (h-1):
        n.append(_img_pixel_index(x,y+1,w))
    return n

def _img_pixel_index(x,y,w):
    return x + y * w

def select_color(img,color,bgcolor=None,new_color=None):
    """
    this function returns a copy of img contaning only the parts colored with color;
    the rest is filled with bgcolor

    img -> PIL image

    """
    img = (img.copy()).convert('RGB')
    if type(bgcolor) is type(None):
        bgcolor = (255,255,255)
    if type(bgcolor) is str:
        bgcolor = PIL.ImageColor.getrgb(bgcolor)
    color = tuple(_get_rgb_color(color))[:3]
    if type(new_color) is type(None):
        new_color = color
    bgcolor = tuple(bgcolor)[:3]
    d = list(img.getdata())
    for k,v in enumerate(d):
        #print(v)
        #print(color)
        #print('-')
        if v == color:
            d[k] = new_color
        else:
            d[k] = bgcolor
    img.putdata(d)
    return img

def erase_outside_circle(img,center_xy,radius,bgcolor=None,copy_fig=False):
    """
    recolors pixels outside a circle defined by center_xy and radius (both in px coord within the width and height of img)

    img -> PILLOW image
    center_xy -> (x,y) in px; x corresponds to horizontal (width) center; y corresponds to vertical (height) center
    radius -> circle radius (in px)
    bgcolor -> (tuple / ndarray / list) pixel color (normalized) that will fill the erased parts outside of the circle
               values should be float or
    """
    if copy_fig:
        img = img.copy()
    is_inside_arena = lambda x,y,c,r:  (x-c[0])**2 + (y-c[1])**2 < r**2
    def fix_pixel_color(col,b):
        fix_range = lambda c: 0 if (c<0) else (255 if (c>255) else c)
        if len(b) == 1:
            col = fix_range(int(float(col if numpy.isscalar(col) else col[0])*256.0))
        else:
            col = numpy.array([col]) if numpy.isscalar(col) else numpy.array(col)
            col = tuple( fix_range(v*256.0) for v in (numpy.tile( col, int(numpy.ceil(len(b)/len(col))) )[:len(b)] if len(b) > len(col) else col[:len(b)]) )
        return col
    if type(bgcolor) is type(None):
        bgcolor = 1.0
    bgcolor = fix_pixel_color(bgcolor,img.getbands())
    d = list(img.getdata())
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if not is_inside_arena(x,y,center_xy,radius):
                d[_img_pixel_index(x,y,img.size[0])] = bgcolor if numpy.isscalar(bgcolor) else bgcolor
    img.putdata(d)
    return img

def align_arena_center_and_crop(img_ref,img,c_ref,c,bgcolor_rgb=None,dx_correct_px=0,dy_correct_px=0,xRange_world_ref=None,yRange_world_ref=None,xRange_world=None,yRange_world=None,showCroppedRegion=False):
    """
    aligns the arena center of img to the arena center of img_ref, and crops the shifted img to the size of img_ref

    img_ref,img -> images openned by PIL or paths to images
    c_ref -> (x,y) center of the arena in pixels for the reference picture
    c -> (x,y) center of the arena in pixels for the image to be aligned
    bgcolor_rgb -> (R,G,B) color for the background, if needed
    dx_correct_px,dy_correct_px -> x and y displacement (in pixels) to manually correct alignment issues (to be added to the center to center displacement vector)
    xRange_world -> (x0,x1) in world coordinates (left and right extent limits of the original img figure, before being cropped)
    yRange_world -> (y0,y1) in world coordinates (bottom and top extent limits of the original img figure, before being cropped)
                    if xRange_world and yRange_world are set, this function also returns the extent (limits) of the cropped img in world coordinates
                    in the format [ left, right, bottom, top ]
    xRange_world_ref,yRange_world_ref -> same, but the the reference img_ref
    """
    bgcolor_rgb = numpy.array( (255,255,255), dtype=numpy.uint8 ) if (type(bgcolor_rgb) is type(None)) else numpy.asarray(bgcolor_rgb)[:3]

    img_ref = _get_PIL_img(img_ref)
    img     = _get_PIL_img(img)
    c_ref = numpy.asarray(c_ref)[:2].astype(int)
    c = numpy.asarray(c)[:2].astype(int)
    w_ref,h_ref = img_ref.size
    w,h = img.size
    resolution_ref = img_ref.info['dpi']
    
    # the grid of pixels to be moved
    xx,yy = numpy.meshgrid(numpy.arange(w),numpy.arange(h))
    r_px = numpy.column_stack((xx.reshape((xx.size,)), yy.reshape((yy.size,)))).astype(int)
    
    # shifting the pixels
    c = c - numpy.array((dx_correct_px,dy_correct_px))
    dr = c_ref - c
    r_shift = r_px - dr

    # now we assign the img pixels
    # first we create a new img with the bg color
    h_max = numpy.max((h,h_ref))
    w_max = numpy.max((w,w_ref))
    img_shift = numpy.tile(bgcolor_rgb,(h_max,w_max,1))

    # selecting only the translated pixels inside the boundary of the image
    ind = numpy.logical_and(numpy.logical_and(numpy.greater_equal(r_shift[:,1],0),numpy.less(r_shift[:,1],h_max)),
                            numpy.logical_and(numpy.greater_equal(r_shift[:,0],0),numpy.less(r_shift[:,0],w_max)) )

    # assigning the new color to the pixels from the rotated img coords
    img_shift[r_px[ind,1],r_px[ind,0],:] = numpy.array(img.convert('RGB'))[ r_shift[ind,1], r_shift[ind,0], : ]

    new_img = PIL.Image.fromarray(img_shift[:h_ref,:w_ref,:].astype('uint8'),'RGB')
    new_img.info['dpi'] = resolution_ref

    img_extent = None
    if (not(type(xRange_world) is type(None))) and (not(type(yRange_world) is type(None))) and (not(type(xRange_world_ref) is type(None))) and (not(type(yRange_world_ref) is type(None))):
        T = misc.LinearTransf2D(  (0.0,w), xRange_world, (0.0,h), numpy.flip(yRange_world)  ) # flip y so that 0 corresponds to the positive coordinate (top) and h to the negative (bottom)
        new_img_r1_world = T(r_shift[0,:]) # new x,y of the (left,top) world coordinates
        new_img_r2_world = T(r_shift[0,:]+numpy.array((w_ref,h_ref))) # new x,y of the (right,bottom) world coordintates
        img_extent = [ new_img_r1_world[0],new_img_r2_world[0],new_img_r2_world[1],new_img_r1_world[1] ]
        if showCroppedRegion:
            plt.imshow(img,extent=[*xRange_world,*yRange_world])
            plt.plot(T(c)[0],T(c)[1],'+r')
            plt.axvline(new_img_r1_world[0],yRange_world[0],yRange_world[1],c='m',linestyle='--')
            plt.axvline(new_img_r2_world[0],yRange_world[0],yRange_world[1],c='m',linestyle='--')
            plt.axhline(new_img_r1_world[1],xRange_world[0],xRange_world[1],c='m',linestyle='--')
            plt.axhline(new_img_r2_world[1],xRange_world[0],xRange_world[1],c='m',linestyle='--')
            plt.text(new_img_r1_world[0],new_img_r1_world[1],'%.2f,%.2f'%(new_img_r1_world[0],new_img_r1_world[1]),c='y',va='bottom',ha='right')
            plt.text(new_img_r2_world[0],new_img_r2_world[1],'%.2f,%.2f'%(new_img_r2_world[0],new_img_r2_world[1]),c='y',va='top',ha='left')
            plt.show()

    if type(img_extent) is type(None):
        warnings.warn('Not able to calculate the cropped img extent because the limits of the img and img_ref were not provided')
    return new_img,img_extent

def transform_pic_to_match_distance(pic_ref,pic,dx_world_ref,dy_world_ref,dx_world,dy_world,distance_ref=1.0,distance=1.0):
    """
    This function defines a transform T, such that
    given two pictures, pic_ref and pic, we resize pic by T(pic) to make its world coordinates match
    the world coordinates of pic_ref.
    In the resulting T(pic), the pixel size of any world distance in either x or y direction 
    will match the pixel size of that same world distance.
    I.e., 10 cm in T(pic) correspond to 10 cm in pic_ref

    this function assumes that both pictures have the same "zoom" property (i.e., the arena looks the same size in both pictures)

    to compensate for zoomed in/out pictures, use the distance_ref which is the distance that match in both worlds
    """
    distance_ref = distance_ref if misc.exists(distance_ref) else 1.0
    distance     = distance     if misc.exists(distance    ) else 1.0
    #from PIL import Image

    # defining width and height in pixel coordinates
    img_ref           = _get_PIL_img(pic_ref)
    w_ref_px,h_ref_px = img_ref.size
    resolution_ref    = img_ref.info['dpi']
    img_ref.close()

    img       = _get_PIL_img(pic)
    w_px,h_px = img.size

    # defining the width and height in the world coordinates
    w_ref_world , h_ref_world = sum((abs(x) for x in dx_world_ref)) , sum((abs(y) for y in dy_world_ref))
    w_world     , h_world     = sum((abs(x) for x in dx_world))     , sum((abs(y) for y in dy_world))

    # transforms the reference picture from the world to pixels
    Twp_ref = misc.LinearTransf2D( (0.0,w_ref_world), (0.0,w_ref_px), (0.0,h_ref_world), (0.0,h_ref_px))
    # transforms the picture from world to pixels
    Twp     = misc.LinearTransf2D( (0.0,w_world)    , (0.0,w_px)    , (0.0,h_world)    , (0.0,h_px)    )

    # this is using the (1,1) vector in both frames of reference, because they should have had the same length
    d_ref = Twp_ref(distance_ref * numpy.ones(2)) # transforming the (1,1) vector to pixels in the reference picture
    d     =     Twp(distance     * numpy.ones(2))     # transforming the (1,1) vector to pixels in the given picture

    # defining the main transform to move from the pixels in the given picture to pixels in the reference picture
    T                      = misc.LinearTransf2D( (0.0,d[0]), (0.0,d_ref[0]), (0.0,d[1]), (0.0,d_ref[1]))
    wh_new                 = T(numpy.array(img.size)).astype(int)
    pic_transf             = img.resize(wh_new, PIL.Image.BICUBIC)
    pic_transf.info['dpi'] = resolution_ref
    img.close()
    return pic_transf

def _get_PIL_img(input_image_or_path,copy=True):
    if type(input_image_or_path) is str:
        input_image = PIL.Image.open(input_image_or_path).convert('RGB')
    else:
        if copy:
            input_image = (input_image_or_path.copy()).convert('RGB')
        else:
            input_image = input_image_or_path.convert('RGB')
    return input_image

"""
##################################
##################################
##################################
################################## canny edge detector
################################## https://www.codingame.com/playgrounds/38470/how-to-detect-circles-in-images
##################################
##################################
"""

def find_arena_in_image_circle_detection(input_image_or_path,dr=8,n_circle_points=100,circle_match_threshold=0.4):
    """
    tries to match the arena to a circle and, if successful, returns the center of the matched circle in pixels
    dr -> radius precision in pixels (look for arenas being circles of r - dr to r + dr, where r is 60 cm converted to pixels in the pilot arena image from 2019 experiments)
    n_circle_points -> number of points in the attempted circles to match the arena
    circle_match_threshold -> minimum percentage of matching circle

    returns x0,y0,r (in pixels)
    """
    w_px,h_px = plib.get_arena_picture_file_width_height()
    pic_bounds = plib.get_arena_picture_bounds()
    r_px,_ = plib.get_center_radius_in_px(pic_bounds.arena_pic_left,pic_bounds.arena_pic_right,pic_bounds.arena_pic_bottom,pic_bounds.arena_pic_top, plib.get_arena_diameter_cm() / 2.0, plib.get_arena_center(), w_px,h_px)
    c = find_circles_in_image(input_image_or_path,int(r_px - dr),int(r_px + dr),n_circle_points=n_circle_points,circle_match_threshold=circle_match_threshold)
    if len(c) == 0:
        warnings.warn('No circle was found with these parameters, returning nan')
        return numpy.nan,numpy.nan,numpy.nan
    if len(c) > 1:
        warnings.warn('More than 1 circle found, returning center of the one with largest radius')
        c = sorted(c,key=lambda s: s[-1])[0]
    else:
        c = c[0]
    return c

def find_circles_in_image(input_image_or_path,rmin,rmax,n_circle_points=100,circle_match_threshold=0.4,verbose=False):
    """
    given an input image, a radius range (in pixels), look for every circle that match a radius from rmin to rmax (step of 1)

    returns a list of circle center and radii (in pixels):
    [(x0,y0,r0),(x1,y1,r1),...]
    """
    # define candidate points constrained by the circle equation that we will look for in the edge points
    circle_trials = []
    for r in range(rmin, rmax + 1):
        for t in range(n_circle_points):
            circle_trials.append((r, int(r * math.cos(2 * math.pi * t / n_circle_points)), int(r * math.sin(2 * math.pi * t / n_circle_points))))
    # build a collection of attempted circles from the edge points
    acc = collections.defaultdict(int)
    for x, y in canny_edge_detector(input_image_or_path):
        for r, dx, dy in circle_trials:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1
    # match the attempted circles with the predefined circle points to determine the best set of center (a,b) and radius r
    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / n_circle_points >= circle_match_threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            if verbose:
                print(v / n_circle_points, x, y, r)
            circles.append((x, y, r))
    return circles

def canny_edge_detector(input_image_or_path):
    #from PIL import Image
    input_image = _get_PIL_img(input_image_or_path)
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height
    # Transform the image to grayscale
    grayscaled = canny_compute_grayscale(input_pixels, width, height)
    # Blur it to remove noise
    blurred = canny_compute_blur(grayscaled, width, height)
    # Compute the gradient
    gradient, direction = canny_compute_gradient(blurred, width, height)
    # Non-maximum suppression
    gradient = canny_filter_out_non_maximum(gradient, direction, width, height)
    # Filter out some edges
    keep = canny_filter_strong_edges(gradient, width, height, 20, 25)
    input_image.close()
    return keep

def canny_compute_grayscale(input_pixels, width, height):
    grayscale = numpy.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale

def canny_compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = numpy.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = numpy.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred

def canny_compute_gradient(input_pixels, width, height):
    gradient = numpy.zeros((width, height))
    direction = numpy.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = math.sqrt(magx**2 + magy**2)
                direction[x, y] = math.atan2(magy, magx)
    return gradient, direction

def canny_filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + math.pi
            rangle = round(angle / (math.pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0
    return gradient

def canny_filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)

