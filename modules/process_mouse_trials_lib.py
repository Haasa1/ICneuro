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

__HAS_ZLIB__ = True

import os # path and file handling
import re as regexp
import glob # name pattern expansion
import copy
import numpy
import pandas # load excel and data files
import openpyxl
import scipy.io
import scipy.sparse
import scipy.spatial
import scipy.optimize
import scipy.interpolate
import matplotlib.image as mpimg
try:
    import zlib # zipping objects
except ImportError:
    __HAS_ZLIB__ = False
    pass
import PIL.Image
import modules.io as io
import modules.helper_func_class as misc
#import warnings

#warnings.filterwarnings('error')

__TRIAL_NUM_TRAINING__    = 1000
__TRIAL_NUM_PROBE__       = 2000
__TRIAL_NUM_EXPLORE__     = 3000
__TRIAL_NUM_HABITUATION__ = 4000
__TRIAL_NUM_FLIP__        = 5000
__TRIAL_NUM_REVERSE__     = 6000
__TRIAL_NUM_OTHER__       = 9000
__TRIAL_NUM_ROTATION__    = 10000
__TRIAL_NUM_UNKNOWN__     = 20000
__TRIAL_NUM_TO_NAME__     = { __TRIAL_NUM_TRAINING__    : 'training',
                              __TRIAL_NUM_PROBE__       : 'probe',
                              __TRIAL_NUM_EXPLORE__     : 'explore',
                              __TRIAL_NUM_HABITUATION__ : 'habituation',
                              __TRIAL_NUM_FLIP__        : 'flip',
                              __TRIAL_NUM_REVERSE__     : 'reverse',
                              __TRIAL_NUM_OTHER__       : 'other',
                              __TRIAL_NUM_ROTATION__    : 'rotation',
                              __TRIAL_NUM_UNKNOWN__     : 'unknown'}

def get_trial_name(trial_number):
    if trial_number < 1000:
        return 'numeric'
    else:
        n = int(numpy.floor(trial_number/1e3)*1e3)
        return __TRIAL_NUM_TO_NAME__[n]

def is_valid_trial_type(trial_type):
    """
    trial_type -> 'numeric','training','probe','explore','habituation','flip','reverse','other','rotation','unknown'
    """
    if (type(trial_type) is list) or (type(trial_type) is tuple):
        return [ is_valid_trial_type(t) for t in trial_type ]
    return (type(trial_type) is str) and (trial_type in (['all','numeric'] + list(__TRIAL_NUM_TO_NAME__.values())))

def is_trial_of_type(trial,trial_type):
    """
    trial      -> is an str
    trial_type -> 'numeric','training','probe','explore','habituation','flip','reverse','other','rotation','unknown'

    if numeric trial (regular training session)
    returns True for 'numeric'
    otherwise
    named trials    -> (T) Training     :: 1000 + training # (1-based)
                       (P) Probe        :: 2000 + probe # (1-based)
                       (E) Explore      :: 3000 + explore # (1-based)
                       (H) Habituation  :: 4000 + habituation # (1-based)
                       (F) Flip         :: 5000 + flip # (1-based)
                       (Re) Reverse     :: 6000 + Reverse # (1-based)
                           other        :: 9000
                       (R) Rotation (R) :: 10000 + trial # + angle/360
                         unknown (no R) :: 20000 + trial # + angle/360
    
    """
    if (type(trial_type) is list) or (type(trial_type) is tuple):
        return numpy.any([ is_trial_of_type(trial,tt) for tt in trial_type ])
    trial_type = trial_type.lower()
    assert is_valid_trial_type(trial_type),"trial type must be one of 'training','probe','explore','habituation','flip','reverse','other','rotation','unknown'"
    if trial_type == 'all':
        return True
    if trial.isdigit():
        return trial_type == 'numeric'
    else:
        tt = trial_type + '2' + ('_2' if trial_type in ['rotation','unknown'] else '')
        #print(tt)
        return get_named_trial_number(trial) == get_named_trial_number(tt)

def get_named_trial_trialnumber(trial):
    return trial_to_number(trial) - get_named_trial_number(trial)

def get_named_trial_number(trial):
    trial = trial.strip().lower()
    if trial.isdigit():
        return int(trial)
    m = regexp.findall('\d+',trial)
    #print(m)
    if len(m) <= 1:
        if trial.startswith('t'):
            return __TRIAL_NUM_TRAINING__
        elif trial.startswith('p'):
            return __TRIAL_NUM_PROBE__
        elif trial.startswith('e'):
            return __TRIAL_NUM_EXPLORE__
        elif trial.startswith('h'):
            return __TRIAL_NUM_HABITUATION__
        elif trial.startswith('f'):
            return __TRIAL_NUM_FLIP__
        elif trial.startswith('re'):
            return __TRIAL_NUM_REVERSE__
        elif trial.startswith('r'):
            return __TRIAL_NUM_ROTATION__
        else:
            return __TRIAL_NUM_OTHER__
    else:
        if trial.startswith('r'):
            return __TRIAL_NUM_ROTATION__
        else: # unknown
            return __TRIAL_NUM_UNKNOWN__

def trial_to_number(trial):
    """
    trial is an str

    numbered trials -> returns number of the trial (str is just an integer number)
    named trials    -> (T) Training     :: 1000 + training # (1-based)
                       (P) Probe        :: 2000 + probe # (1-based)
                       (E) Explore      :: 3000 + explore # (1-based)
                       (H) Habituation  :: 4000 + habituation # (1-based)
                       (F) Flip         :: 5000 + flip # (1-based)
                       (Re) Reverse     :: 6000 + Reverse # (1-based)
                           other        :: 9000
                       (R) Rotation (R) :: 10000 + trial # + angle/360
                         unknown (no R) :: 20000 + trial # + angle/360
    """
    try:
        return int(trial) # if trial is just a digit, this will return the trial number
    except:
        trial     = trial.strip().lower()
        get_digit = lambda mm: int(mm[0]) if len(mm) > 0 else 1 # if no digit is present, 1 is assumed
        m         = regexp.findall('\d+',trial)
        N         = get_named_trial_number(trial)
        return N + get_digit(m) if len(m) <= 1 else float(N) + float(m[1]) + float(m[0])/360.0


def ethovision_to_track_matfile(file_path_to_ethovision_excel_file,num_of_header_rows_in_excel=-1,trials_to_process='all',correct_distortion=True,mouse_gender=None,correct_arena_center=False):
    valid_trials_str = "'all', 'numeric', " + str([a for a in __TRIAL_NUM_TO_NAME__.values()])[1:-1]
    """
    Generates a MATLAB-compatible track file from the Excel Ethovision file containing the mouse trajectory

    file_path_to_ethovision_excel_file -> file path to the Ethovision generated Excel file
    num_of_header_rows_in_excel        -> the number of rows before the data table in the excel file...
                                          the data table starts in the 'Trial time' row and column
    trials_to_process                  -> 'all', 'numeric', or any other that is defined in the __TRIAL_NUM_TO_NAME__ variable
                                        """+valid_trials_str+"""

    The Ethovision Excel file MUST BE COMPATIBLE WITH THE FOLLOWING STRUCTURE:

    - rows 1 to num_of_header_rows_in_excel -> header (meta-info about the experiment)
                                               1st COLUMN MUST CONTAIN (2nd col has the value of each of these fields):
                                                  * Day
                                                  * Trial
                                                  * Trial ID
                                                  * Trial name
                                                  * Mouse Number
                                                  * Start Location

    - after the header, there must be a data table with:
        * 1st row: column titles MUST INCLUDE:
            :: Trial time
            :: Recording time
            :: X center
            :: Y center
            :: X nose
            :: Y nose
            :: X tail
            :: Y tail
            :: Direction
            :: Velocity
        * 2nd row: must contain the measurement unit of each of the above quantities
        * 3rd row on: the measured values for each of the quantities

    RETURNS:

        A track file containing the above information extracted from the Excel file,
        organized in a struct with the following fields:
            * exper_date           -> [[ experiment properties ]] date of the experiment
            * trial                -> [[ experiment properties ]] string: label that identifies this trial (e.g., 'Probe', '3', 'R180 1', etc)
            * mouse_number         -> [[ experiment properties ]] string: label of the mouse
            * mouse_gender         -> [[ input (not included in excel files) ]] string: 'F' for female, 'M' for male; default is 'M' if nothing is given
            * start_location       -> [[ experiment properties ]] string: starting location label ('SW','SE','NE','NW')
            * start_quadrant       -> [[ corrected start_location label ]] int: quadrant relative to positive x (1,2,3,4)
            * is_reverse           -> [[ experiment properties ]] if the trial is reversed
            * day                  -> [[ experiment properties ]] day of the experiment
            * trial_id             -> [[ experiment properties ]] 'Trial ID' in the Excel file header
            * trial_name           -> [[ experiment properties ]] 'Trial name' in the Excel file header
            * file_name            -> [[ metadata ]] name of the original excel file
            * file_trial_idx       -> [[ metadata ]] index of the original excel file name
            * arena_diameter       -> [[ arena properties for plotting ]] diameter of the arena (assumed constant == 120cm for all experiments)
            * arena_pic_bottom     -> [[ arena properties for plotting ]] (in cm) bottom extent of the arena picture for this experiment -- when ploting, use ax.imshow(arena_pic, extent=[track.arena_pic_left, track.arena_pic_right, track.arena_pic_bottom, track.arena_pic_top]) 
            * arena_pic_left       -> [[ arena properties for plotting ]] (in cm) left extent of the arena picture for this experiment   -- when ploting, use ax.imshow(arena_pic, extent=[track.arena_pic_left, track.arena_pic_right, track.arena_pic_bottom, track.arena_pic_top]) 
            * arena_pic_right      -> [[ arena properties for plotting ]] (in cm) right extent of the arena picture for this experiment  -- when ploting, use ax.imshow(arena_pic, extent=[track.arena_pic_left, track.arena_pic_right, track.arena_pic_bottom, track.arena_pic_top]) 
            * arena_pic_top        -> [[ arena properties for plotting ]] (in cm) top extent of the arena picture for this experiment    -- when ploting, use ax.imshow(arena_pic, extent=[track.arena_pic_left, track.arena_pic_right, track.arena_pic_bottom, track.arena_pic_top]) 
            * arena_picture        -> [[ arena properties for plotting ]] PNG file name in subfolder mouse_track/arena_picture for this experiment
            * arena_picture_wh     -> [[ arena properties for plotting ]] (in pixels) width,height of the arena_picture
            * unit_direction       -> [[ measurement units ]] string: physical units for measuring direction
            * unit_r               -> [[ measurement units ]] string: physical units for measuring positions (all variables that are prefixed with r_)
            * unit_time            -> [[ measurement units ]] string: physical units for measuring time
            * unit_velocity        -> [[ measurement units ]] string: physical units for measuring velocity
            * r_arena_center       -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of the center of the arena in arena_picture
            * r_arena_holes        -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of each hole (1 per row)
            * r_start              -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of the starting position
            * r_target             -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of the target for this trial
            * r_target_alt         -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of the alternative target (2-target experiment only)
            * r_target_alt_reverse -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of the alternative target REL (2-target experiment reverse target of the alternative target)
            * r_target_reverse     -> [[ arena properties ]] numpy.ndarray: (in unit_r) x,y coordinates of the REL (reverse target)
            * time                 -> [[ time-dependent quantities ]] numpy.ndarray: (in unit_time) time values
            * r_nose               -> [[ time-dependent quantities ]] numpy.ndarray: (in unit_r) time series of the mouse nose position (one x,y per row)
            * r_center             -> [[ time-dependent quantities ]] numpy.ndarray: (in unit_r) time series of the mouse center position (one x,y per row)
            * r_tail               -> [[ time-dependent quantities ]] numpy.ndarray: (in unit_r) time series of the mouse tail position (one x,y per row)
            * velocity             -> [[ time-dependent quantities ]] numpy.ndarray: (in unit_velocity) time series of the mouse velocities
            * direction            -> [[ time-dependent quantities ]] numpy.ndarray: (in unit_direction) time series of the mouse body direction
    
    EXAMPLE: 
    >>> import modules.process_mouse_trials_lib as plib
    >>> track = plib.ethovision_to_track_matfile('2019-09-06_Raw Trial Data/Raw data-Hidden Food Maze-06Sept2019-Trial     8.xlsx')
    >>> track.arena_diameter # gives the diameter of the arena
    120.0
    >>> track.r_nose         # gives the series of (x,y) coordinates for the nose of the mouse (each row is an x,y pair)
    array([[ 49.5485, -41.8135],
       [ 49.565 , -42.1669],
       [ 49.6376, -42.0139],
       ...,
       [-38.9484, -35.6736],
       [-38.3127, -36.3303],
       [-37.6054, -37.0496]])

    """
    assert is_valid_trial_type(trials_to_process),"ethovision_to_track_matfile ::: trials_to_process must be one of: " + valid_trials_str
    num_of_header_rows_in_excel = (get_n_header_rows(file_path_to_ethovision_excel_file) - 1) if (num_of_header_rows_in_excel < 0) else num_of_header_rows_in_excel
    trial_idx                   = get_trial_file_index(file_path_to_ethovision_excel_file)
    fheader                     = read_trial_header(file_path_to_ethovision_excel_file,trial_idx,'',nrows_header=num_of_header_rows_in_excel,mouse_gender=mouse_gender)
    if not is_trial_of_type(fheader.trial,trials_to_process):
        return None
    d = read_trial_data(file_path_to_ethovision_excel_file,nrows_header=num_of_header_rows_in_excel+1)

    track = get_track_file(fheader,d)

    if (track.exper_date == '15Nov2021'): # rotating this random entrance experiment to match the one from '16Jul2021'
        print('   ... ::: ROTATING THE 15-November-2021 EXPERIMENT to match the 16-july-2021 one')
        track                = rotate_trial_file(track,None,return_only_track=True,angle=numpy.pi)
        track.start_location = _rotate_start_label(track.start_location,track.r_start,numpy.pi)
        track.start_quadrant = get_start_location_quadrant(track.r_start)
        if correct_distortion:
            print('   ... ::: TRANSFORMING THE COORDINATES 15-November-2021 EXPERIMENT to match the 16-july-2021 one')
            track                = apply_distort_transform(track,get_transform_for_15Nov2021_experiments_to_match_16Jul2021)
    
    if (track.exper_date == '12Aug2022'): # rotating this random entrance experiment to match the one from '16Jul2021'
        if correct_distortion:
            print('   ... ::: TRANSFORMING THE COORDINATES 12-August-2022 EXPERIMENT to match the 16-july-2021 one')
            track                = apply_distort_transform(track,get_transform_for_12Aug2022_experiments_to_match_16Jul2021)

    if (track.exper_date == '11Oct2022'): # rotating this random entrance experiment to match the one from '16Jul2021'
        if correct_distortion:
            print('   ... ::: TRANSFORMING THE COORDINATES 11-October-2022 EXPERIMENT to match the 16-july-2021 one')
            track                = apply_distort_transform(track,get_transform_for_11Oct2022_experiments_to_match_16Jul2021)

    if correct_arena_center:
        track.r_arena_center = numpy.mean(track.r_arena_holes,axis=0)

    return track

def get_track_file(file_header=None,data=None):
    if not misc.exists(file_header):
        file_header = read_trial_header()
    if not misc.exists(data):
        data = read_trial_data()
    return misc.trackfile(**file_header,**data)

def get_arena_grid_limits(r_center=None,track=None):
    c = numpy.asarray(r_center) if misc.exists(r_center) else get_arena_center(track)
    r = 2.0 + 60.0 # offset of 2cm outside of the arena
    X_grid_lim = c[0] + (-r,r) # this
    Y_grid_lim = c[1] + (r,-r) # the y coords have to be inverted because I want the index to grow from top to bottom (i.e. positive correspond to 0)
    return X_grid_lim,Y_grid_lim

def align_targets_group_by_start_quadrant(track,start_align_vector=None):
    """
    this function aligns all the targets in track consistently across trials,
    always using as reference the target from the mouse who entered the arena from the first quadrant

    if track is a list of list, we assume that
    track[i][j] -> mouse j of trial i

    if track is a list, it is assumed that
    track[j] -> mouse j; each element is a new mouse from the same trial

    if track is a single track file, align the entrance to start_align_vector

    return
        list of aligned tracks
    """
    if not misc.exists(start_align_vector):
        start_align_vector = (-1,1)

    # if track is a list of lists,
    # we assume the external list to be one of trials
    # and the internal list to be the mice in that trial
    # track[i][j] -> mouse j in trial i
    # so we perform this function for each trial i
    if misc.is_list_of_1d_collection_or_none(track,1,allow_none_element=False):
        return [ align_targets_group_by_start_quadrant(tr,start_align_vector=start_align_vector) for tr in track ]

    # we first align all the entrances, putting ourselves into the mouse frame of reference
    # since the target for random entrance is always fixed, this transformation generates 4 targets, because now the entrance is fixed
    track_rot         = rotate_trial_file(copy.deepcopy(track),start_align_vector,True) # plib.align_targets(copy.deepcopy(all_trials_ft),(1,0))

    if type(track) is misc.trackfile:
        return track_rot

    # now, we group those mice which started from the same entrance,
    # because all the mice who started in the same position, had to find the same of the four targets
    track_rot_group,_ = io.group_track_list(track_rot, group_by='start_quadrant', get_key_group_func=lambda v: v, sortgroups_label='mouse_number', get_key_sortgroups_func=int ) # dict(SE=0,NE=1,SW=2,NW=3)[v]

    # then, we rotate every mice to align all the targets with the ones that entered from the first quadrant...
    # this keeps a consistent alignment over trials and allows us to check whether mice are checking the previous trial target
    return sum([ rotate_trial_file(copy.deepcopy(all_mice),None,True,angle=k*numpy.pi/2) for k,all_mice in enumerate(track_rot_group) ],[])

def align_targets(track,align_vector=None):
    if not misc.exists(align_vector):
        align_vector = numpy.array((1,0))
    align_vector = align_vector if misc._is_numpy_array(align_vector) else numpy.asarray(align_vector).flatten()
    if type(track) is list:
        return [ align_targets(tr,align_vector) for tr in track ]
    else:
        R = misc.RotateTransf( track.r_target-track.r_arena_center, track.arena_diameter*align_vector, track.r_arena_center ) # we want the axis of rotation to be in track.r_arena_center
        for k in track:
            if (k.startswith('r_')) and (type(track[k]) is numpy.ndarray):
                track[k] = R(track[k])
        return track

def rotate_trial_file(track,ref_vector,return_only_track=False,angle=None):
    """
    Rotates all numpy arrays inside track file, such that the entrance matches the direction pointed by the ref_vector
    this includes hole coordinates, arena center, nose, center and tail positions, target, reverse target, and entrance

    angle (radians) is only used if ref_vector is not given
    """
    if type(track) is list:
        return [ rotate_trial_file(tr,ref_vector,return_only_track=return_only_track,angle=angle) for tr in track ]
    else:
        if misc.exists(ref_vector):
            R = misc.RotateTransf( track.r_start-track.r_arena_center, track.arena_diameter*numpy.asarray(ref_vector), track.r_arena_center ) # we want the axis of rotation to be in track.r_arena_center
        else:
            assert misc.exists(angle), "rotate_trial_file ::: when ref_vector is not given, you must provide an angle in radians"
            R = misc.RotateTransf( None, None, track.r_arena_center, angle ) # we want the axis of rotation to be in track.r_arena_center
        for k in track:
            if (k.startswith('r_')) and (type(track[k]) is numpy.ndarray):
                track[k] = R(track[k])
        if track.IsField('unit_direction'):
            to_deg = 180.0 / numpy.pi if track.unit_direction == 'deg' else 1.0
        if track.IsField('direction'):
            track.direction = track.direction + R.theta*to_deg # direction is the only exception of coordinates that dont start with r_
        if return_only_track:
            return track
        else:
            return misc.structtype(track=track,R=R)

def zoom_trial_file(track,zoom_factor):
    """
    Displaces all spatial coordinates for each track in tracks by the given displacement vector 
    this includes hole coordinates, arena center, nose, center and tail positions, target, reverse target, and entrance
    """
    if type(track) is list:
        return [ zoom_trial_file(tr,zoom_factor) for tr in track ]
    else:
        # defining the displacement transformation
        T = lambda r,zoom: r*zoom
        #R = misc.RotateTransf( track.r_start-track.r_arena_center, track.arena_diameter*ref_vector, track.r_arena_center ) # we want the axis of rotation to be in track.r_arena_center
        for k in track:
            if (k.startswith('r_')) and (type(track[k]) is numpy.ndarray):
                track[k] = T(track[k],zoom_factor)
        return track

def shift_trial_file(track,diplacement_vec):
    """
    Displaces all spatial coordinates for each track in tracks by the given displacement vector 
    this includes hole coordinates, arena center, nose, center and tail positions, target, reverse target, and entrance
    """
    if type(track) is list:
        return [ shift_trial_file(tr,diplacement_vec) for tr in track ]
    else:
        # defining the displacement transformation
        T = lambda r,d: r + d
        #R = misc.RotateTransf( track.r_start-track.r_arena_center, track.arena_diameter*ref_vector, track.r_arena_center ) # we want the axis of rotation to be in track.r_arena_center
        for k in track:
            if (k.startswith('r_')) and (type(track[k]) is numpy.ndarray):
                track[k] = T(track[k],diplacement_vec)
        return track

def rotate_arena_picture(img,R,track,bgcolor_rgba):
    """
    rotates img using the rotate transform R
    since R is defined in the arena world, we need the arena information in the track file
    pixels from outside of the img will be colored with bgcolor_rgba
    """
    if R is None:
        return img
    bgcolor_rgba = numpy.array( (1,1,1,1) ) if bgcolor_rgba is None else numpy.asarray(bgcolor_rgba)
    bgcolor_rgba = bgcolor_rgba[:3] if img.shape[2] == 3 else bgcolor_rgba
    bgcolor_rgba = bgcolor_rgba.astype(float)
    w_px,h_px = track.arena_picture_wh
    w_px,h_px = int(w_px),int(h_px)
    #r_px,c_px = get_center_radius_in_px(track.arena_pic_left,track.arena_pic_right,track.arena_pic_bottom,track.arena_pic_top, track.arena_diameter / 2.0, track.r_arena_center, w_px, h_px)

    # I want to rotate everything in the arena world
    # so first I define the transforms from pixels to arena
    # transforms a vector list rr to arena world
    T = misc.LinearTransf2D( (0.0,w_px), (track.arena_pic_left,track.arena_pic_right), (0.0,h_px), (track.arena_pic_top,track.arena_pic_bottom) )
    # and the inverse transforms (bring back to pixel world)
    # transforms a vector list rr to pixel world
    T_inv = misc.LinearTransf2D( (track.arena_pic_left,track.arena_pic_right), (0.0,w_px), (track.arena_pic_top,track.arena_pic_bottom), (0.0,h_px) )

    # the grid of pixels
    xx,yy = numpy.meshgrid(numpy.arange(w_px),numpy.arange(h_px))
    r_px = numpy.column_stack((xx.reshape((xx.size,)), yy.reshape((yy.size,))))
    # here, we transform each pixel vector to arena world, rotate it, then transform it back to pixel world
    # the rotation is made in the inverse direction because the picture has x and y switched
    Rr_px = numpy.floor(
        T_inv(
            R(
                T(r_px),
                -1.0)
            )).astype(int)
    
    # now we assign the img pixels
    # first we create a new img with the bg color
    img_r = numpy.tile(bgcolor_rgba,(h_px,w_px,1))

    # selecting only the rotated pixels inside the original boundary of the original image
    ind = numpy.logical_and(numpy.logical_and(numpy.greater_equal(Rr_px[:,1],0),numpy.less(Rr_px[:,1],h_px)),
                            numpy.logical_and(numpy.greater_equal(Rr_px[:,0],0),numpy.less(Rr_px[:,0],w_px)) )
    # assigning the new color to the pixels from the rotated img coords
    img_r[r_px[ind,1],r_px[ind,0],:] = img[ Rr_px[ind,1], Rr_px[ind,0], : ]
    return img_r

def get_center_radius_in_px(x0_world,x1_world,y0_world,y1_world, r_world, c_world, w_px,h_px):
    """
    given the x0,y0 and x1,y1 as the bottom-left and top-right coordinates of the world
    together with the center and radius in the world coordinates
    determines the r and c coordinates in the world of pixels
    r -> scalar
    c -> (x,y)_px
    """
    # defining the transformation of coordinates for the width and height of the picture
    # these transforms basically define the stretch ratio in each of the x and y directions
    # and that's what must be used to calculate the radius, since the
    # radius is a postive-definite quantity
    T_X_r = misc.LinearTransf( (0.0,numpy.abs(x0_world)+numpy.abs(x1_world)), (0.0,w_px) )
    T_Y_r = misc.LinearTransf( (0.0,numpy.abs(y0_world)+numpy.abs(y1_world)), (0.0,h_px) )
    r_px = (T_X_r(r_world) + T_Y_r(r_world))/2.0 # taking the average of the diameter in the x and y directions (there's a small discrepancy between both)
    # defining the transformation of the coordinates of the center of the arena towards the pixels
    T_X_c = misc.LinearTransf( (x0_world,x1_world), (0.0,w_px) )
    T_Y_c = misc.LinearTransf( (y1_world,y0_world), (0.0,h_px) ) # y is switched because in the pixel world, the pixels start from 0 at the top
    c_px = numpy.array(  (  T_X_c(c_world[0]),  T_Y_c(c_world[1])  )  )
    return r_px, c_px

def get_cropped_arena_picture(track,arena_offset_pix=50,bgcolor_rgba=None):
    """
    crops the arena picture from the track (tentatively imported from ./arena_picture)
    around the arena edges with an offset of arena_offset (to the outside of the arena ground)
    arena_offset -> margin to be used out of the arena for the crop
    bgcolor -> (R,G,B,alpha) a 4 element list or tuple with the color for the background, default is opacque white, values of rgba are normalized to [0,1]
    """
    arena_pic = get_arena_picture(track)
    if len(arena_pic) == 0:
        return None
    arena_offset_pix = float(arena_offset_pix)
    bgcolor_rgba = numpy.array( (1,1,1,1) ) if bgcolor_rgba is None else numpy.asarray(bgcolor_rgba)
    bgcolor_rgba = bgcolor_rgba[:3] if arena_pic.shape[2] == 3 else bgcolor_rgba
    w_px,h_px = track.arena_picture_wh
    r_px,c_px = get_center_radius_in_px(track.arena_pic_left,track.arena_pic_right,track.arena_pic_bottom,track.arena_pic_top, track.arena_diameter / 2.0, track.r_arena_center, w_px,h_px)
    y = 0
    while y < h_px:
        x = 0
        while x < w_px:
            if (x-c_px[0])**2 + (y-c_px[1])**2 > (r_px+arena_offset_pix)**2: # if the pixel is outside the radius
                arena_pic[y,x,:] = bgcolor_rgba # we recolor the pixel
            x+=1
        y+=1
    return arena_pic

def get_arena_picture(track,as_array=True):
    pic_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','arena_picture',track.arena_picture)
    if as_array:
        if os.path.exists(pic_file):
            return mpimg.imread(pic_file)
        return numpy.array([])
    else:
        if os.path.exists(pic_file):
            return PIL.Image.open(pic_file)
        return None

#def get_unzip_arena_picture(track):
#    if __HAS_ZLIB__:
#        return numpy.frombuffer(zlib.decompress(bytes(track.arena_picture,'latin1')),dtype=track.arena_picture_dtype).reshape(track.arena_picture_shape)
#    else:
#        warnings.warn('The library zlib was not found, cannot unzip the arena picture... Please load it from the file')
#        return numpy.zeros(1)

def get_trial_file_index(file_path):
    return misc.try_or_default(lambda:int(file_path.split('-Trial   ')[1].split('.')[0]))

def get_trial_files(inp_dir,file_ptrn='*-Trial*.xlsx',return_index=True):
    return [ ((get_trial_file_index(f),f) if return_index else f)  for f in glob.glob( os.path.join(inp_dir,file_ptrn) ) ]

def apply_distort_transform(track,get_transform_func,copy_track=False):
    if type(track) is list:
        return [ apply_distort_transform(tr,get_transform_func=get_transform_func,copy_track=copy_track) for tr in track ]
    else:
        tr = copy.deepcopy(track) if copy_track else track
        T = get_transform_func()
        for k in tr:
            if (k.startswith('r_')) and (type(tr[k]) is numpy.ndarray):
                tr[k] = T(tr[k])
        return tr

def get_transform_for_15Nov2021_experiments_to_match_16Jul2021():
    """
    this function converts all the coordinates of the 15Nov2021 set of experiments such that its hole positions
    match those of the 16Jul2021 experiment

    for that, I extracted the square coordinates around the arena of the 16Jul2021 experiments,
    and distorted it in photoshop such that both hole sets perfectly overlay.

    Then, I use the four vertices of the square and its distorted version to calculate
    a projective transformation matrix between the two frames of reference, according to:
    https://math.stackexchange.com/a/339033/59039

    the pictures from which I extracted the coords are in the arena_picture/distort folder
    """
    # vertices xi,yi of the reference square
    # these are the coordinates of the 16Jul2021 square coords
    # i.e., 15Nov2021 arena on the 16Jul2021 frame of ref BEFORE distort
    # these numbers are extracted using the engauge software on the reference points put in photoshop
    ri = numpy.array([ (-65.7668,-63.1622),
                       (64.2332,-63.1622),
                       (64.2332,66.8378),
                       (-65.7668,66.8378) ])
    # vertices xf,yf of the transformed square
    # these are the coordinates of the 16Jul2021 square coords
    # i.e., 15Nov2021 arena on the 16Jul2021 frame of ref AFTER distort
    # these numbers are extracted using the engauge software on the reference points put in photoshop
    rf = numpy.array([ (-64.554,-64.5002),
                       (64.4748,-58.217),
                       (58.2131,73.5383),
                       (-73.3724,63.3903) ])
    return misc.DistortTransf2D(ri,rf)

def get_transform_for_12Aug2022_experiments_to_match_16Jul2021():
    """
    this function converts all the coordinates of the 12Aug2022 set of experiments such that its hole positions
    match those of the 16Jul2021 experiment

    for that, I extracted the square coordinates around the arena of the 16Jul2021 experiments,
    and distorted it in photoshop such that both hole sets perfectly overlay.

    Then, I use the four vertices of the square and its distorted version to calculate
    a projective transformation matrix between the two frames of reference, according to:
    https://math.stackexchange.com/a/339033/59039

    the pictures from which I extracted the coords are in the arena_picture/distort folder
    """
    # vertices xi,yi of the reference square
    # these are the coordinates of the 16Jul2021 square coords
    # i.e., 12Aug2022 arena on the 16Jul2021 frame of ref BEFORE distort
    # these numbers are extracted using the engauge software on the reference points put in photoshop
    ri = numpy.array([ (-65.7668,-63.1622),
                       (64.2332,-63.1622),
                       (64.2332,66.8378),
                       (-65.7668,66.8378) ])
    # vertices xf,yf of the transformed square
    # these are the coordinates of the 16Jul2021 square coords
    # i.e., 12Aug2022 arena on the 16Jul2021 frame of ref AFTER distort
    # these numbers are extracted using the engauge software on the reference points put in photoshop
    rf = numpy.array([ (-66.1141,-67.1134),
                       (63.55,-57.6804),
                       (58.1344,67.5773),
                       (-72.458,67.268) ])
    return misc.DistortTransf2D(ri,rf)

def get_transform_for_11Oct2022_experiments_to_match_16Jul2021():
    """
    this function converts all the coordinates of the 11Oct2022 set of experiments such that its hole positions
    match those of the 16Jul2021 experiment

    for that, I extracted the square coordinates around the arena of the 16Jul2021 experiments,
    and distorted it in photoshop such that both hole sets perfectly overlay.

    Then, I use the four vertices of the square and its distorted version to calculate
    a projective transformation matrix between the two frames of reference, according to:
    https://math.stackexchange.com/a/339033/59039

    the pictures from which I extracted the coords are in the arena_picture/distort folder
    """
    # vertices xi,yi of the reference square
    # these are the coordinates of the 16Jul2021 square coords
    # i.e., 11Oct2022 arena on the 16Jul2021 frame of ref BEFORE distort
    # these numbers are extracted using the engauge software on the reference points put in photoshop
    ri = numpy.array([ (-65.7668,-63.1622),
                       (64.2332,-63.1622),
                       (64.2332,66.8378),
                       (-65.7668,66.8378) ])
    # vertices xf,yf of the transformed square
    # these are the coordinates of the 16Jul2021 square coords
    # i.e., 11Oct2022 arena on the 16Jul2021 frame of ref AFTER distort
    # these numbers are extracted using the engauge software on the reference points put in photoshop
    rf = numpy.array([ (-64.7215 , -64.6422),
                       (65.3294  , -60.9284),
                       (61.3837  , 68.5106 ),
                       (-67.5066 , 65.2611 ) ])
    return misc.DistortTransf2D(ri,rf)


def read_trial_data(fname='',nrows_header=37,file_header=None):
    r = misc.structtype(time           = [] ,
                        r_center       = [] ,
                        r_nose         = [] ,
                        r_tail         = [] ,
                        direction      = [] ,
                        velocity       = [] ,
                        unit_time      = '' ,
                        unit_r         = '' ,
                        unit_direction = '' ,
                        unit_velocity  = '' )
    if len(fname) == 0:
        return r
    print('     ... reading data')
    nrows_header = get_n_header_rows(fname) if nrows_header is None else nrows_header
    d = pandas.read_excel(fname,skiprows=nrows_header,na_values=['-'],header=0)

    err_msg = ' *** ERROR  :::  %s column not found in ' + io.get_filename(fname)

    r.time           = misc.try_or_default(lambda: d['Recording time'][1:].to_numpy().astype(float), default=numpy.array([]), msg=err_msg%'Recording time' )
    r.unit_time      = misc.try_or_default(lambda: d.iloc[0,:]['Recording time'], default='', msg=err_msg%'Recording time')

    r.r_center       = misc.try_or_default(lambda: d[['X center','Y center']][1:].to_numpy().astype(float), default=numpy.array([]), msg=err_msg%'X center or Y center' )
    r.r_nose         = misc.try_or_default(lambda: d[['X nose','Y nose']][1:].to_numpy().astype(float), default=numpy.array([]), msg=err_msg%'X nose or Y nose' )
    r.r_tail         = misc.try_or_default(lambda: d[['X tail','Y tail']][1:].to_numpy().astype(float), default=numpy.array([]), msg=err_msg%'X tail or Y tail' )
    r.unit_r         = misc.try_or_default(lambda: d.iloc[0,:]['X center'], default='', msg=err_msg%'X center' )

    r.direction      = misc.try_or_default(lambda: d['Direction'][1:].to_numpy().astype(float), default=numpy.array([]), msg=err_msg%'Direction' )
    r.unit_direction = misc.try_or_default(lambda: d.iloc[0,:]['Direction'], default='', msg=err_msg%'Direction')

    r.velocity       = misc.try_or_default(lambda: d['Velocity'][1:].to_numpy().astype(float), default=numpy.array([]), msg=err_msg%'Velocity' )
    r.unit_velocity  = misc.try_or_default(lambda: d.iloc[0,:]['Velocity'], default='', msg=err_msg%'Velocity')

    return r

def get_n_header_rows(fname):
    h = openpyxl.load_workbook(fname)
    return list(filter(lambda v: v[1]=='Trial time', ((k,c[0].value) for k,c in enumerate(h.active['A1':'A50']) )))[0][0]

def read_trial_header(fname='',file_trial_idx=None,arena_pic_dir='',nrows_header=None,mouse_gender=None):
    #global __ARENA_PICTURES__
    #if len(__ARENA_PICTURES__) == 0:
    #    import_all_arena_pictures(arena_pic_dir)
    r = misc.structtype(file_name            =  ''                                 ,
                        exper_date           =  ''                                 ,
                        file_trial_idx       =  numpy.nan                          ,
                        day                  =  numpy.nan                          ,
                        trial                =  ''                                 ,
                        trial_id             =  numpy.nan                          ,
                        trial_name           =  ''                                 ,
                        mouse_number         =  numpy.nan                          ,
                        mouse_gender         =  'M'                                ,
                        start_location       =  ''                                 ,
                        start_quadrant       = -1                                  ,
                        r_target             =  numpy.array((numpy.nan,numpy.nan)) ,
                        r_target_reverse     =  numpy.array((numpy.nan,numpy.nan)) ,
                        r_target_alt         =  numpy.array((numpy.nan,numpy.nan)) ,
                        r_target_alt_reverse =  numpy.array((numpy.nan,numpy.nan)) ,
                        r_start              =  numpy.array((numpy.nan,numpy.nan)) ,
                        arena_diameter       =  get_arena_diameter_cm()            ,
                        r_arena_holes        =  numpy.array((numpy.nan,numpy.nan)) ,
                        r_arena_center       =  numpy.array((numpy.nan,numpy.nan)) ,
                        arena_picture        =  ''                                 ,
                        arena_picture_wh     =  numpy.array((numpy.nan,numpy.nan)) ,
                        is_reverse           =  False )
    if len(fname) == 0:
        r.Set(**get_arena_picture_bounds(r))
        return r
    print('     ... reading header')
    nrows_header = nrows_header if misc.exists(nrows_header) else get_n_header_rows(fname)-1
    h = pandas.read_excel(fname,nrows=nrows_header,names=['name','value'],header=None,usecols='A,B')
                        #, #arena_picture_dtype='', arena_picture_shape=(0,0,0), arena_picture_decompress_str='',
                        #**get_arena_picture_bounds())
    
    err_msg = ' *** ERROR  :::  %s header not found in ' + io.get_filename(fname)

    #r.arena_picture_decompress_str = "numpy.frombuffer(zlib.decompress(bytes(arena_picture,'latin1')),dtype=arena_picture_dtype).reshape(arena_picture_shape)"
    r.file_name            = io.get_filename_with_parentdir(fname)
    r.exper_date           = io.get_filename(fname).split('-')[2].replace(' ','')
    r.file_trial_idx       = file_trial_idx
    r.day                  = misc.try_or_default(lambda: h['value'][h['name'] == 'Day'].to_numpy()[0]           , default='', msg=err_msg%'Day'           ).replace(' ','_')
    r.trial                = misc.try_or_default(lambda: h['value'][h['name'] == 'Trial'].to_numpy()[0]         , default='', msg=err_msg%'Trial'         ).replace(' ','_')
    r.trial_id             = misc.try_or_default(lambda: h['value'][h['name'] == 'Trial ID'].to_numpy()[0]      , default='', msg=err_msg%'Trial ID'      ).replace(' ','_')
    r.trial_name           = misc.try_or_default(lambda: h['value'][h['name'] == 'Trial name'].to_numpy()[0]    , default='', msg=err_msg%'Trial name'    ).replace(' ','_')
    r.mouse_number         = misc.try_or_default(lambda: h['value'][h['name'] == 'Mouse Number'].to_numpy()[0]  , default='', msg=err_msg%'Mouse Number'  ).replace(' ','_')
    r.start_location       = misc.try_or_default(lambda: h['value'][h['name'] == 'Start Location'].to_numpy()[0], default='', msg=err_msg%'Start Location').replace(' ','_')

    r.mouse_gender         = misc.try_or_default(lambda: mouse_gender[int(r.mouse_number)] if misc.exists(mouse_gender) else 'M', default='M', msg=' *** Warning ::: Using default    mouse_number: %s  --  mouse_gender: M'%r.mouse_number)

    r.is_reverse           = is_reverse_condition(r)
    r.r_target             = get_arena_target(r)
    r.r_target_alt         = get_arena_alt_target(r)
    r.r_target_reverse     = get_arena_reverse_target(r)
    r.r_target_alt_reverse = get_arena_alt_reverse_target(r)
    
    # we use the start location label from experiments to define the start coordinates and the arena picture
    r.r_start           = numpy.array(get_arena_entrance_coord(r)[r.start_location])
    r.arena_picture     = get_arena_all_picture_filename(r)[r.start_location]

    r.arena_picture_wh  = numpy.array(get_arena_picture_file_width_height(r))
    r.r_arena_holes     = get_arena_hole_coord(r)
    r.r_arena_center    = get_arena_center(r)
    
    r.start_quadrant    = get_start_location_quadrant(r.r_start)

    r.Set(**get_arena_picture_bounds(r))
    return r

def _rotate_compass_label_default(label,angle):
    """
    given a compass label (SE,SW,NW,NE)
    this function returns the rotated label (positive is CCW)
    (i.e., rotate SE by +pi/2 gives NE)
    """
    pos  = dict(NE = numpy.array(( 1, 1)),
                NW = numpy.array((-1, 1)),
                SW = numpy.array((-1,-1)),
                SE = numpy.array(( 1,-1)))
    return _rotate_start_label(label, pos[label], -angle)

def _rotate_start_label(label,r_start,angle):
    """
    given a start label (SE,SW,NW,NE) corresponding to a start position r_start

    this function returns the label corresponding to a rotation of angle
    """
    r0 = numpy.sign(r_start)
    r1 = numpy.sign(misc.RotateTransf(None,None,(0,0),-angle)(r0)).flatten()
    invert_hori = lambda c: 'E' if c == 'W' else 'W'
    invert_vert = lambda c: 'N' if c == 'S' else 'S'
    return (label[0] if r0[1] == r1[1] else invert_vert(label[0])) + (label[1] if r0[0] == r1[0] else invert_hori(label[1]))

def get_start_location_quadrant(r_start):
    if r_start[0] >= 0:     # positive x
        if r_start[1] >= 0:     # positive y
            return 1                # 1st quadrant
        else:                   # negative y
            return 4                # 4th quadrant
    else:                   # negative x
        if r_start[1] >= 0:     # positive y
            return 2                # 2nd quadrant
        else:                   # negative y
            return 3                # 3rd quadrant

#def import_all_arena_pictures(arena_pic_dir):
#    global __ARENA_PICTURES__
#    print(' ::: importing all arena pictures from %s ... '%arena_pic_dir,end='')
#    __ARENA_PICTURES__ = dict(SW=get_compressed_arena_pic('SW',arena_pic_dir),
#                              SE=get_compressed_arena_pic('SE',arena_pic_dir),
#                              NE=get_compressed_arena_pic('NE',arena_pic_dir),
#                              NW=get_compressed_arena_pic('NW',arena_pic_dir))
#    print('done')

def get_compressed_arena_pic(start_location,arena_pic_dir):
    img = mpimg.imread(os.path.join(arena_pic_dir,get_arena_all_picture_filename()[start_location]))
    if __HAS_ZLIB__:
        get_pic = lambda im: zlib.compress(im,5).decode('latin1')
    else:
        get_pic = lambda im: None # save memory
    return (get_pic(img),str(img.dtype),img.shape)


"""
################################################
################################################
################################################
################################################
################################################
################################################ The functions below need to be updated when adding new experiments to process
################################################
################################################
################################################
################################################
################################################
################################################
"""

def get_arena_all_picture_filename(file_header=None):
    fname = 'BKGDimage-pilot.png'
    if (type(file_header) is type(None)) or \
       (file_header.exper_date == '06Sept2019') or \
       (file_header.exper_date == '07Oct2019') or \
       (file_header.exper_date == '08Jul2019') or \
       (file_header.exper_date == '23May2019') or \
       (file_header.exper_date == 'Pilot'):
        return dict(SW='BKGDimage-pilot.png',SE='BKGDimage-pilot2.png',NE='BKGDimage-pilot3.png',NW='BKGDimage-pilot4.png')
    if (file_header.exper_date == '11Dec2019'):
        fname = 'dec-2019/BKGDimage-localCues_cropped.png'
    if (file_header.exper_date == '08Mar2021'):
        fname = 'mar-may-2021/BKGDimage-localCues_clear_cropped.png'
    if (file_header.exper_date == '06May2021'):
        fname = 'mar-may-2021/BKGDimage-localCues_Letter_cropped.png'
    if (file_header.exper_date == '26May2021'):
        fname = 'mar-may-2021/BKGDimage-localCues_LetterTex_cropped.png'
    if (file_header.exper_date == '22Jun2021') or (file_header.exper_date == '19Nov2021'):
        fname = 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
    if (file_header.exper_date == '16Jul2021') or (file_header.exper_date == '15Nov2021'):
        if (file_header.exper_date == '15Nov2021'):
            fname = 'jun-jul-aug-nov-2021/BKGDimage-Nov15_cropped.png'
        else:
            fname = 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
    if file_header.exper_date == '30Jul2021': 
        if (file_header.is_reverse):
            fname = 'jun-jul-aug-nov-2021/BKGDimage_asymmRev_cropped.png'
        else:
            fname = 'jun-jul-aug-nov-2021/BKGDimage_asymm_cropped.png'
    if file_header.exper_date == '11Aug2021': 
        if file_header.is_reverse:
            fname = 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_Reverse_cropped.png'
        else:
            fname = 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_cropped.png'
    if (file_header.exper_date == '12Aug2022'):
        fname = '2022/BKGDimage-220812_cropped.png'
    if (file_header.exper_date == '11Oct2022'):
        fname = '2022/BKGDimage-20221011_cropped.png'
    if (file_header.exper_date == '20Sept2022'):
        if _is_probe2_trial(file_header.trial):
            fname = '2022/BKGDimage-20220920-probe2_cropped.png'
        else:
            fname = '2022/BKGDimage-20220920_cropped.png'
    if (file_header.exper_date == '04Nov2022'):
        fname = '2022/BKGDimage-20221104_cropped.png'
    return dict(SW=fname,SE=fname,NE=fname,NW=fname)

def _is_probe2_trial(trial_str):
    return is_trial_of_type(trial_str, 'probe') and ('2' in trial_str)

def is_reverse_condition(track):
    is_reverse = False
    if track.trial.startswith(u'R') or track.trial.startswith(u'Flip') or track.trial.startswith(u'Revers'):
        is_reverse = True
    if (track.exper_date == '22Jun2021') or (track.exper_date == '19Nov2021') or (track.exper_date == '12Aug2022') or (track.exper_date == '11Oct2022') or (track.exper_date == '20Sept2022'):
        if track.trial.startswith(u'Probe'):
            d = regexp.findall('\d+',track.trial) # check if it is ProbeX, where X is a number
            if (len(d) > 0): # if this is ProbeX
                if int(d[-1]) >= 2: # and X >= 2 
                    is_reverse = True
        if track.trial.startswith(u'R'):
            is_reverse = True
        if track.trial.isdigit(): #
            if int(track.trial) > 18:
                is_reverse = True
    return is_reverse

def get_arena_picture_bounds(file_header=None,as_list=False):
    #            [x0,x1,y0,y1] # axis limits for the cropped versions of the arena pictures
    img_extent = [-97.0,97.0,-73.0,73.0]
    """
    the img_extent above is good for the following experiments:
    (file_header.exper_date == '06Sept2019') or (file_header.exper_date == '07Oct2019') or (file_header.exper_date == '08Jul2019') or (file_header.exper_date == '23May2019') or (file_header.exper_date == 'Pilot')
    using the following pictures:
    arena_picture/BKGDimage-pilot.png
    arena_picture/BKGDimage-pilot2.png
    arena_picture/BKGDimage-pilot3.png
    arena_picture/BKGDimage-pilot4.png
    """
    if type(file_header) is type(None):
        if as_list:
            return img_extent
        else:
            return misc.structtype(arena_pic_left=img_extent[0],arena_pic_right=img_extent[1], arena_pic_bottom=img_extent[2],arena_pic_top=img_extent[3])
    if (file_header.exper_date == '11Dec2019'): # 'dec-2019/BKGDimage-localCues_cropped.png'
        img_extent = [-103.78538641686184,90.43288056206092,-72.42141666666669,73.63858333333333]
    if (file_header.exper_date == '08Mar2021'): # 'mar-may-2021/BKGDimage-localCues_clear_cropped.png'
        img_extent = [-101.31718309859156,92.80112676056339,-70.21533333333336,76.02466666666666]
    if (file_header.exper_date == '06May2021'): # 'mar-may-2021/BKGDimage-localCues_Letter_cropped.png'
        img_extent = [-101.62049295774648,92.49781690140844,-68.69200000000002,77.548]
    if (file_header.exper_date == '26May2021'): # 'mar-may-2021/BKGDimage-localCues_LetterTex_cropped.png'
        img_extent = [-101.62049295774648,92.49781690140844,-69.30133333333335,76.93866666666666]
    if (file_header.exper_date == '22Jun2021') or (file_header.exper_date == '19Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
        img_extent = [-105.14112449799197, 89.0436144578313, -71.25892473118277, 74.93462365591398]
    if (file_header.exper_date == '16Jul2021') or (file_header.exper_date == '15Nov2021'):
        if (file_header.exper_date == '15Nov2021'): # jun-jul-aug-nov-2021/BKGDimage-Nov15_cropped.png
            img_extent = [-102.10698795180723, 92.07775100401605, -72.47720430107525, 73.71634408602151]
        else: # 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
            img_extent = [-104.53429718875502, 89.65044176706826, -70.64978494623655, 75.54376344086022]
    if file_header.exper_date == '30Jul2021': 
        if (file_header.is_reverse): # 'jun-jul-aug-nov-2021/BKGDimage_asymmRev_cropped.png'
            img_extent = [-101.50016064257028,92.68457831325298,-68.82236559139785,77.37118279569893]
        else:                        # 'jun-jul-aug-nov-2021/BKGDimage_asymm_cropped.png'
            img_extent = [-101.80357429718876,92.38116465863453,-68.82236559139785,77.37118279569893]
    if file_header.exper_date == '11Aug2021': 
        if file_header.is_reverse: # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_Reverse_cropped.png'
            img_extent = [-105.44453815261045,88.74020080321284,-70.04064516129033,76.15290322580645]
        else:                      # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_cropped.png'
            img_extent = [-105.74795180722892,88.43678714859436,-70.04064516129033,76.15290322580645]
    if (file_header.exper_date == '12Aug2022'):
        img_extent = [-102.8197797797798, 91.28792792792794, -70.57941071428571, 75.46916071428572] # '2022/BKGDimage-220812_cropped.png'
    if (file_header.exper_date == '11Oct2022'):
        img_extent = [-105.07659645232815, 89.01652993348114, -69.17229249011861, 76.89569169960474] # '2022/BKGDimage-20221011_cropped.png'
    if (file_header.exper_date == '20Sept2022'):
        if _is_probe2_trial(file_header.trial):
            img_extent = [-102.67863496932516, 91.569217791411, -70.96838709677417, 75.12193548387097] # '2022/BKGDimage-20220920-probe2_cropped.png'
        else:
            img_extent = [-107.53483128834355, 86.71302147239263, -70.96838709677417, 75.12193548387097] # '2022/BKGDimage-20220920_cropped.png'
    if (file_header.exper_date == '04Nov2022'):
        img_extent = [-105.53609805924413, 92.53703779366703, -72.46542805100178, 76.50943533697632] # '2022/BKGDimage-20221104_cropped.png'
    if as_list:
        return img_extent
    else:
        return misc.structtype(arena_pic_left=img_extent[0],arena_pic_right=img_extent[1], arena_pic_bottom=img_extent[2],arena_pic_top=img_extent[3])

def get_arena_diameter_cm():
    return 120.0

def get_arena_picture_file_width_height(file_header=None):
    if type(file_header) is type(None):
        return (640.0,480.0)
    img = get_arena_picture(file_header,as_array=False)
    sz  = numpy.array( img.size )
    img.close()
    return tuple(float(s) for s in sz)

def get_arena_to_arena_translate(file_header=None,pic_bounds=None,file_header_ref=None,pic_bounds_ref=None):
    pic_bounds     = pic_bounds     if misc.exists(pic_bounds)     else  get_arena_picture_bounds(file_header    ,as_list=True)
    pic_bounds_ref = pic_bounds_ref if misc.exists(pic_bounds_ref) else (get_arena_picture_bounds(file_header_ref,as_list=True) if misc.exists(file_header_ref) else get_arena_picture_bounds(None,as_list=True))
    #pic_bounds_ref = get_arena_picture_bounds(None,as_list=True)
    return misc.LinearTransf2D( pic_bounds_ref[:2], pic_bounds[:2], pic_bounds_ref[2:], pic_bounds[2:] )

def get_arena_center(file_header=None):
    c0 = [7.59409,-1.9734]
    if (type(file_header) is type(None)) or \
        (file_header.exper_date == '06Sept2019') or \
        (file_header.exper_date == '07Oct2019') or \
        (file_header.exper_date == '08Jul2019') or \
        (file_header.exper_date == '23May2019') or \
        (file_header.exper_date == 'Pilot'):
        return get_arena_to_arena_translate(file_header)(numpy.array(c0)) # this should be (almost) equivalent to the positions above, measured by the crop_arena_pictures script
    if (file_header.exper_date == '11Dec2019'): # 'dec-2019/BKGDimage-localCues_cropped.png'
        c0 = [0.9103981264637184, -1.217166666666671]
    if (file_header.exper_date == '08Mar2021'): # 'mar-may-2021/BKGDimage-localCues_clear_cropped.png'
        c0 = [3.3247183098591506, 1.0766666666666538]
    if (file_header.exper_date == '06May2021'): # 'mar-may-2021/BKGDimage-localCues_Letter_cropped.png'
        c0 = [3.0214084507042287, 2.5999999999999943]
    if (file_header.exper_date == '26May2021'): # 'mar-may-2021/BKGDimage-localCues_LetterTex_cropped.png'
        c0 = [3.0214084507042287, 1.9906666666666553]
    if (file_header.exper_date == '22Jun2021') or (file_header.exper_date == '19Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
        c0 = [-1.3736546184738927, 1.2287096774193458]
    if (file_header.exper_date == '16Jul2021') or (file_header.exper_date == '15Nov2021'):
        if (file_header.exper_date == '15Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-Nov15_cropped.png'
            c0 = [1.6604819277108334, 0.010430107526872234]
        else: # 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
            c0 = [-0.7668273092369589, 1.8378494623655826]
    if file_header.exper_date == '30Jul2021': 
        if (file_header.is_reverse): # 'jun-jul-aug-nov-2021/BKGDimage_asymmRev_cropped.png'
            c0 = [3.1775502008032106, 2.4469892473118193]
        else:                        # 'jun-jul-aug-nov-2021/BKGDimage_asymm_cropped.png'
            c0 = [2.8741365461847295, 2.4469892473118193]
    if file_header.exper_date == '11Aug2021': 
        if file_header.is_reverse: # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_Reverse_cropped.png'
            c0 = [-1.07024096385544, 0.9241397849462345]
        else:                      # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_cropped.png'
            c0 = [-1.07024096385544, 1.2287096774193458]
    if (file_header.exper_date == '12Aug2022'):
        c0 = [1.5131131131131212, 0.6192678571428729] # '2022/BKGDimage-220812_cropped.png'
    if (file_header.exper_date == '11Oct2022'):
        c0 = [-0.8283111111111339, 2.3434523809523853] # '2022/BKGDimage-20221011_cropped.png'
    if (file_header.exper_date == '20Sept2022'):
        if _is_probe2_trial(file_header.trial):
            c0 = [1.7295858895705294, 0.250645161290322] # '2022/BKGDimage-20220920-probe2_cropped.png'
        else:
            c0 = [-3.126610429447865, 0.250645161290322] # '2022/BKGDimage-20220920_cropped.png'
    if (file_header.exper_date == '04Nov2022'):
        c0 = [0.9282124616956082, 0.1598178506375234] # '2022/BKGDimage-20221104_cropped.png'
    #return get_arena_to_arena_translate(file_header)(numpy.array(c0)) # this should be (almost) equivalent to the positions above, measured by the crop_arena_pictures script
    return numpy.array(c0)

def match_entrance_labels_quadrant(labels_old,ref_labels):
    """
    This function makes the quadrant of the labels in labels_old match the quadrant of the corresponding labels in ref_labels

    each input is a dict with fields
    dict(SE=(x,y),NE=(x,y),SW=(x,y),NW=(x,y))

    the label SE in labels_old will have the coordinates corresponding to the quadrant of SE in ref_labels
    """
    # quadrant to label lookup table
    quad_to_label = { get_start_location_quadrant(numpy.asarray(r)):k for k,r in ref_labels.items()  }
    labels        = dict(SE=[],NE=[],SW=[],NW=[])
    all_labels    = list(labels.keys())
    for k in all_labels:
        labels[quad_to_label[get_start_location_quadrant(labels_old[k])]] = labels_old[k]
    return labels


def get_arena_entrance_coord(file_header=None):
    #x,y
    T = get_arena_to_arena_translate(file_header)
    get_coord = lambda x,y: tuple(T(numpy.array(  (x,y)  )))
    d_ref = dict(SE=get_coord(-36.2310,39.0066),
                 NE=get_coord(-34.0706,-45.1481),
                 SW=get_coord(48.4940,41.9264),
                 NW=get_coord(50.51,-43.9046))
    if (type(file_header) is type(None)) or \
        (file_header.exper_date == '06Sept2019') or \
        (file_header.exper_date == '07Oct2019') or \
        (file_header.exper_date == '08Jul2019') or \
        (file_header.exper_date == '23May2019') or \
        (file_header.exper_date == 'Pilot'):
        return d_ref
    if (file_header.exper_date == '11Dec2019'): # 'dec-2019/BKGDimage-localCues_cropped.png'
        d = dict(SE=numpy.array((-43.09217799,38.94933333)),
                 NE=numpy.array((-39.45058548,-44.42658333)),
                 SW=numpy.array((41.27138173,42.29654167)),
                 NW=numpy.array((43.69911007,-41.99225000)))
    if (file_header.exper_date == '08Mar2021'): # 'mar-may-2021/BKGDimage-localCues_clear_cropped.png'
        d = dict(SE=numpy.array((-40.35190141,40.37866667)),
                 NE=numpy.array((-36.71218310,-41.27200000)),
                 SW=numpy.array((41.84507042,45.25333333)),
                 NW=numpy.array((45.78809859,-38.53000000)))
    if (file_header.exper_date == '06May2021'): # 'mar-may-2021/BKGDimage-localCues_Letter_cropped.png'
        d = dict(SE=numpy.array((-40.04859155,41.90200000)),
                 NE=numpy.array((-36.40887324,-40.05333333)),
                 SW=numpy.array((41.84507042,45.55800000)),
                 NW=numpy.array((45.18147887,-37.31133333)))
    if (file_header.exper_date == '26May2021'): # 'mar-may-2021/BKGDimage-localCues_LetterTex_cropped.png'
        d = dict(SE=numpy.array((-39.74528169,41.29266667)),
                 NE=numpy.array((-36.40887324,-40.05333333)),
                 SW=numpy.array((41.23845070,45.55800000)),
                 NW=numpy.array((44.57485915,-37.92066667)))
    if (file_header.exper_date == '22Jun2021') or (file_header.exper_date == '19Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
        d = dict(SE=numpy.array((38.98036145,46.00048387)),
                 NE=numpy.array((-46.58228916,40.82279570)),
                 SW=numpy.array((43.53156627,-38.66994624)),
                 NW=numpy.array((-40.21060241,-43.84763441)))
    if (file_header.exper_date == '11Oct2022'): # '2022/BKGDimage-20221011_cropped.png'
        """
        you must match the output of crop_arena_pictures in the _holes.txt files
        """
        d = dict(SE=numpy.array((41.40305987,44.63901186)),
                 NE=numpy.array((-44.72576497,42.50885375)),
                 SW=numpy.array((42.31287140,-39.95869565)),
                 NW=numpy.array((-41.08651885,-42.39316206)))
    if (file_header.exper_date == '12Aug2022'): # '2022/BKGDimage-220812_cropped.png'
        """
        you must match the output of crop_arena_pictures in the _holes.txt files
        """
        d = dict(SE=numpy.array((-41.55453453,-41.06542857)),
                 NE=numpy.array((43.06429429,-42.58676786)),
                 SW=numpy.array((-41.55453453,41.99969643)),
                 NW=numpy.array((43.36758759,42.91250000)))
    if (file_header.exper_date == '16Jul2021') or (file_header.exper_date == '15Nov2021'): 
        if (file_header.exper_date == '15Nov2021'):
            d = dict(SE=numpy.array((-41.42425703,-39.88822581)),
                     NE=numpy.array((41.10425703,-42.93392473)),
                     SW=numpy.array((-39.60377510,42.04107527)),
                     NW=numpy.array((42.92473896,40.82279570)))
        else: # 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
            d = dict(SE=numpy.array((38.98036145,46.00048387)),
                     NE=numpy.array((-45.97546185,40.82279570)),
                     SW=numpy.array((43.53156627,-38.06080645)),
                     NW=numpy.array((-39.90718876,-43.54306452)))
            
    if file_header.exper_date == '30Jul2021': 
        if (file_header.is_reverse): # 'jun-jul-aug-nov-2021/BKGDimage_asymmRev_cropped.png'
            d = dict(SE=numpy.array((-40.81742972,42.04107527)),
                     NE=numpy.array((-37.78329317,-40.49736559)),
                     SW=numpy.array((42.01449799,47.21876344)),
                     NW=numpy.array((45.95887550,-38.06080645)))
        else:                        # 'jun-jul-aug-nov-2021/BKGDimage_asymm_cropped.png'
            d = dict(SE=numpy.array((-40.81742972,41.43193548)),
                     NE=numpy.array((-36.87305221,-39.88822581)),
                     SW=numpy.array((41.71108434,46.60962366)),
                     NW=numpy.array((45.35204819,-37.75623656)))
    if file_header.exper_date == '11Aug2021': 
        if file_header.is_reverse: # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_Reverse_cropped.png'
            d = dict(SE=numpy.array((-44.45839357,40.21365591)),
                     NE=numpy.array((-39.60377510,-42.32478495)),
                     SW=numpy.array((38.07012048,44.47763441)),
                     NW=numpy.array((41.40767068,-38.36537634)))
        else:                      # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_cropped.png'
            d = dict(SE=numpy.array((-45.36863454,39.60451613)),
                     NE=numpy.array((-38.99694779,-42.93392473)),
                     SW=numpy.array((37.76670683,45.08677419)),
                     NW=numpy.array((42.01449799,-37.75623656)))
    if (file_header.exper_date == '20Sept2022'):
        if _is_probe2_trial(file_header.trial): # '2022/BKGDimage-20220920-probe2_cropped.png'
            d = dict(SE=numpy.array((-42.27969325,-41.44596774)), # entrance labels are rotated in raw xlsx files
                     NE=numpy.array((43.00725460,-43.88080645)),
                     SW=numpy.array((-41.67266871,42.25161290)),
                     NW=numpy.array((43.61427914,44.07774194)))
        else: #                                         '2022/BKGDimage-20220920_cropped.png'
            d = dict(SE=numpy.array((-47.74291411,-41.44596774)),
                     NE=numpy.array((37.54403374,-43.88080645)),
                     SW=numpy.array((-46.83237730,42.25161290)),
                     NW=numpy.array((38.15105828,44.07774194)))
    if (file_header.exper_date == '04Nov2022'): #       '2022/BKGDimage-20221104_cropped.png'
        d = dict(SE=numpy.array((-42.40028601,-42.36009107)),
                 NE=numpy.array((42.70926456,-44.22227687)),
                 SW=numpy.array((-42.40028601,42.36936248)),
                 NW=numpy.array((43.32824311,43.92118397)))
    return d #match_entrance_labels_quadrant(d,d_ref)

def get_mean_min_hole_dist(file_header=None):
    r = get_arena_hole_coord(file_header)
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(r)) # calculating the distance between every two holes in the arena
    d2 = numpy.asarray([ numpy.sort(d)[1:3] for d in D ]) # getting the 2 closest holes to each hole
    return numpy.mean(numpy.mean(d2,axis=1)) # getting the mean distance to the 3 closest holes

def get_arena_hole_coord(file_header=None):
    """x,y"""
    T = get_arena_to_arena_translate(file_header)
    r = T(numpy.array([ [-48.2102,-6.85804],[-47.5773,5.02923],[-43.983,-21.1837],[-42.6746,23.0125],[-40.9232,-12.0397],[-39.446,-32.7662],
                        [-38.4601,2.59081],[-38.3998,31.547],[-36.9156,14.1733],[-35.1699,-23.6221],[-33.002,-4.72443],[-32.7595,-34.2902],[-31.1985,-14.7829],
                        [-29.8921,28.499],[-28.7024,15.6973],[-27.5128,2.89562],[-26.6647,-27.8894],[-26.4025,-48.0063],[-24.7031,37.9478],[-23.344,-39.4718],
                        [-23.1688,44.6534],[-22.9916,-16.3069],[-21.7538,-5.94363],[-18.6343,31.8518],[-17.8979,-52.5783],[-16.8409,16.9165],[-15.3993,-20.8789],
                        [-15.3422,6.55324],[-14.3588,40.691],[-11.9078,49.5303],[-11.7605,-25.7557],[-11.6634,20.8789],[-10.5703,-38.2526],[-10.1862,0.152401],
                        [-9.06703,-46.4823],[-8.99022,-9.60125],[-7.08705,28.1942],[-5.5489,36.7286],[-2.79191,46.4823],[-1.41944,-24.5365],[-0.460239,-1.98121],
                        [-0.425959,14.4781],[0.0323754,-57.4551],[2.08917,54.1023],[2.50814,-36.7286],[2.78873,-48.0063],[3.24897,26.9749],[4.48939,38.5574],
                        [5.28101,-19.3549],[7.12641,-9.29645],[7.15751,5.63883],[9.30635,15.3925],[9.79389,-42.5198],[11.3384,-30.9374],[11.7986,44.0438],
                        [12.0792,32.7662],[12.1941,-58.0647],[14.2509,53.4927],[14.7435,-1.98121],[15.0133,-18.4405],[15.7027,20.5741],[17.3786,-50.7495],
                        [20.1363,-40.691],[21.6744,-32.1566],[23.3509,42.8246],[23.5782,5.94363],[24.4701,-3.81002],[25.1583,34.595],[26.0445,22.0981],
                        [26.2508,-24.8413],[26.7986,-53.7975],[28.9456,-44.9582],[29.6254,-10.5157],[29.9873,17.2213],[31.4283,-20.8789],[32.4859,48.9207],
                        [33.5251,-36.119],[36.0377,2.28601],[37.2756,12.6493],[37.932,35.8142],[38.363,-49.2255],[39.898,-42.215],[40.9911,44.6534],
                        [41.2533,24.5365],[42.1008,-6.55324],[43.5932,-19.9645],[44.7829,-32.7662],[46.0906,11.1253],[47.3475,30.6326],[47.59,1.06681],
                        [50.062,19.9645],[52.1111,-18.1357],[53.3522,-6.24843],[53.8981,-36.119],[54.3381,29.1086],[55.8153,8.38205],[58.1736,-27.2797],
                        [58.8757,17.8309],[63.0769,-8.99165],[63.407,3.50522] ] ) )
    if (type(file_header) is type(None)) or \
        (file_header.exper_date == '06Sept2019') or \
        (file_header.exper_date == '07Oct2019') or \
        (file_header.exper_date == '08Jul2019') or \
        (file_header.exper_date == '23May2019') or \
        (file_header.exper_date == 'Pilot'):
        return r # this should be (almost) equivalent to the positions above, measured by the crop_arena_pictures script
    if (file_header.exper_date == '11Dec2019'): # 'dec-2019/BKGDimage-localCues_cropped.png'
        r = numpy.array([ [-5.46238876,54.77250000],[6.67625293,54.77250000],[24.58074941,50.20812500],[-19.72529274,50.20812500],[-10.62131148,47.16520833],
                          [33.38126464,46.25233333],[-30.95353630,45.33945833],[3.94505855,45.33945833],[15.47676815,44.12229167],[-21.84955504,41.38366667],
                          [-3.03466042,39.25362500],[-32.47086651,38.34075000],[-13.35250585,37.42787500],[30.34660422,37.12358333],[17.29756440,35.60212500],
                          [4.24852459,34.08066667],[-26.09807963,32.55920833],[39.75405152,32.25491667],[-45.82337237,31.95062500],[47.03723653,30.73345833],
                          [-37.32632319,29.21200000],[-14.86983607,28.90770833],[-4.55199063,27.99483333],[33.68473068,25.86479167],[-50.07189696,23.43045833],
                          [8.19358314,21.90900000],[42.48524590,21.60470833],[51.58922717,19.47466667],[-35.80899297,16.43175000],[1.51733021,16.43175000],
                          [-7.89011710,15.21458333],[-44.00257611,14.60600000],[30.04313817,13.99741667],[38.54018735,12.78025000],[48.55456674,10.04162500],
                          [-22.45648712,7.60729167],[16.08370023,6.99870833],[-0.30346604,6.69441667],[-54.32042155,5.47725000],[56.44468384,4.86866667],
                          [28.82927400,3.65150000],[-34.29166276,3.34720833],[-45.21644028,3.04291667],[40.36098361,2.43433333],[-17.29756440,0.91287500],
                          [7.28318501,-0.60858333],[-7.58665105,-0.91287500],[17.29756440,-2.73862500],[-39.75405152,-3.95579167],[46.43030445,-4.86866667],
                          [34.89859485,-5.17295833],[-28.52580796,-5.17295833],[-54.62388759,-6.69441667],[56.14121780,-7.30300000],[0.00000000,-8.21587500],
                          [-16.08370023,-8.82445833],[22.45648712,-9.12875000],[-47.34070258,-11.56308333],[-37.93325527,-14.30170833],[-29.43620609,-15.51887500],
                          [45.21644028,-16.43175000],[7.89011710,-17.04033333],[-1.82079625,-18.25750000],[36.41592506,-18.25750000],[-22.45648712,-19.77895833],
                          [24.27728337,-19.47466667],[-50.07189696,-20.69183333],[-41.57484778,-22.82187500],[-8.19358314,-23.43045833],[19.11836066,-23.43045833],
                          [-18.51142857,-24.95191667],[51.28576112,-25.25620833],[-33.07779859,-27.38625000],[4.55199063,-29.82058333],[14.86983607,-30.73345833],
                          [38.23672131,-31.34204167],[-45.51990632,-31.95062500],[-38.84365340,-33.47208333],[47.03723653,-34.08066667],[26.70501171,-34.68925000],
                          [-16.99409836,-37.12358333],[-29.73967213,-38.34075000],[33.07779859,-40.47079167],[22.45648712,-43.20941667],[-15.17330211,-45.33945833],
                          [-3.64159251,-46.55662500],[-32.47086651,-47.16520833],[31.56046838,-47.46950000],[10.92477752,-49.29525000],[-23.97381733,-51.42529167],
                          [20.33222482,-52.03387500],[-6.06932084,-55.98966667],[6.06932084,-56.59825000] ])
    if (file_header.exper_date == '08Mar2021'): # 'mar-may-2021/BKGDimage-localCues_clear_cropped.png'
        r = numpy.array([ [-3.65140845,56.83066667],[8.48098592,56.52600000],[-17.60366197,51.95600000],[26.67957746,52.26066667],[-8.50436620,49.21400000],
                          [35.17225352,48.30000000],[-28.82612676,47.38600000],[6.05450704,47.08133333],[17.58028169,46.16733333],[-19.72683099,43.12066667],
                          [-1.22492958,41.29266667],[-30.03936620,40.37866667],[-11.23415493,39.46466667],[32.44246479,39.46466667],[19.40014085,37.94133333],
                          [6.35781690,36.11333333],[41.84507042,34.59000000],[-23.97316901,34.59000000],[-43.68830986,33.98066667],[48.82119718,33.37133333],
                          [-35.19563380,30.93400000],[-12.44739437,30.93400000],[-2.43816901,30.02000000],[35.47556338,28.19200000],[20.61338028,25.75466667],
                          [-47.63133803,25.45000000],[10.30084507,24.23133333],[44.57485915,23.92666667],[53.67415493,22.09866667],[24.25309859,20.88000000],
                          [3.93133803,18.74733333],[-33.37577465,18.44266667],[-5.77457746,17.52866667],[-41.56514085,16.61466667],[31.83584507,16.61466667],
                          [40.63183099,15.39600000],[50.64105634,12.65400000],[-20.03014085,9.91200000],[18.18690141,9.60733333],[1.80816901,9.30266667],
                          [58.52711268,8.08400000],[-52.18098592,7.47466667],[30.62260563,6.56066667],[-31.85922535,5.64666667],[-42.77838028,5.34200000],
                          [42.45169014,5.34200000],[-14.87387324,3.51400000],[9.39091549,1.99066667],[-5.47126761,1.68600000],[19.09683099,0.16266667],
                          [-37.31880282,-1.66533333],[36.99211268,-2.27466667],[48.21457746,-1.97000000],[-26.09633803,-2.57933333],[-52.48429577,-4.40733333],
                          [58.22380282,-4.40733333],[2.41478873,-5.62600000],[-13.66063380,-6.23533333],[24.55640845,-6.23533333],[-45.20485915,-8.97733333],
                          [-35.49894366,-11.71933333],[-27.00626761,-12.93800000],[47.30464789,-13.54733333],[9.99753521,-14.15666667],[0.29161972,-15.07066667],
                          [38.81197183,-15.37533333],[-19.72683099,-16.89866667],[-47.93464789,-18.42200000],[-39.13866197,-20.25000000],[-6.07788732,-20.25000000],
                          [-16.08711268,-22.07800000],[53.67415493,-22.38266667],[-30.64598592,-24.51533333],[6.66112676,-26.64800000],[16.97366197,-27.56200000],
                          [40.32852113,-28.17133333],[-43.38500000,-29.39000000],[-36.40887324,-30.91333333],[28.80274648,-31.52266667],[49.73112676,-31.21800000],
                          [-1.83154930,-32.43666667],[-14.57056338,-34.26466667],[-27.30957746,-35.78800000],[15.76042254,-36.09266667],[35.47556338,-37.61600000],
                          [5.75119718,-37.92066667],[24.55640845,-40.05333333],[-12.75070423,-42.49066667],[-1.22492958,-43.70933333],[-30.03936620,-44.62333333],
                          [33.95901408,-44.62333333],[13.33394366,-46.14666667],[-21.54669014,-48.58400000],[22.73654930,-49.19333333],[-3.65140845,-53.15400000],[8.48098592,-53.76333333] ])
    if (file_header.exper_date == '06May2021'): # 'mar-may-2021/BKGDimage-localCues_Letter_cropped.png'
        r = numpy.array([ [-3.65140845,57.74466667],[8.48098592,57.44000000],[26.37626761,53.47933333],[-17.60366197,52.87000000],[-8.50436620,50.12800000],
                          [35.17225352,49.51866667],[-28.82612676,47.99533333],[5.75119718,47.99533333],[17.58028169,47.08133333],[-19.72683099,44.03466667],
                          [-1.22492958,42.20666667],[-30.03936620,41.29266667],[-11.23415493,40.37866667],[32.13915493,40.37866667],[19.09683099,38.85533333],
                          [6.05450704,37.02733333],[41.54176056,35.50400000],[-23.97316901,35.50400000],[-43.68830986,34.59000000],[48.82119718,34.28533333],
                          [-34.89232394,31.84800000],[-12.44739437,31.84800000],[-2.43816901,30.93400000],[35.47556338,29.10600000],[20.31007042,26.66866667],
                          [-47.63133803,26.05933333],[10.30084507,24.84066667],[44.27154930,25.14533333],[53.67415493,23.01266667],[24.25309859,21.79400000],
                          [3.62802817,19.66133333],[-33.37577465,19.35666667],[-5.77457746,18.44266667],[-41.86845070,17.52866667],[31.83584507,17.52866667],
                          [40.32852113,16.31000000],[50.64105634,13.56800000],[-20.03014085,10.82600000],[18.18690141,10.52133333],[1.80816901,9.91200000],
                          [58.52711268,8.69333333],[-52.18098592,8.38866667],[30.62260563,7.47466667],[-31.85922535,6.56066667],[-42.77838028,5.95133333],
                          [42.45169014,6.25600000],[-14.87387324,4.42800000],[9.39091549,2.90466667],[-5.47126761,2.60000000],[19.09683099,1.07666667],
                          [-37.31880282,-0.75133333],[48.21457746,-1.05600000],[36.68880282,-1.36066667],[-26.09633803,-1.97000000],[58.22380282,-3.49333333],
                          [-52.48429577,-3.49333333],[2.41478873,-4.71200000],[-13.66063380,-5.01666667],[24.25309859,-5.32133333],[-44.90154930,-8.36800000],
                          [-35.49894366,-10.80533333],[-27.00626761,-12.02400000],[47.30464789,-12.63333333],[9.99753521,-13.24266667],[0.29161972,-14.15666667],
                          [38.50866197,-14.46133333],[-19.72683099,-16.28933333],[-47.93464789,-17.50800000],[-39.13866197,-19.33600000],[-6.07788732,-19.33600000],
                          [-16.08711268,-21.16400000],[53.67415493,-21.77333333],[-30.64598592,-23.60133333],[6.66112676,-25.73400000],[16.97366197,-26.64800000],
                          [40.32852113,-27.25733333],[-43.38500000,-28.47600000],[-36.40887324,-29.99933333],[49.42781690,-30.30400000],[28.80274648,-30.60866667],
                          [-1.83154930,-31.52266667],[-14.57056338,-33.35066667],[-27.30957746,-34.87400000],[15.76042254,-35.48333333],[35.47556338,-36.70200000],
                          [5.75119718,-37.00666667],[24.55640845,-39.13933333],[-12.75070423,-41.88133333],[-1.22492958,-42.79533333],[-30.03936620,-44.01400000],
                          [33.95901408,-43.70933333],[13.33394366,-45.23266667],[-21.54669014,-47.97466667],[22.73654930,-48.27933333],[-3.65140845,-52.54466667],[8.48098592,-52.84933333] ])
    if (file_header.exper_date == '26May2021'): # 'mar-may-2021/BKGDimage-localCues_LetterTex_cropped.png'
        r = numpy.array([ [-3.95471831,57.74466667],[8.17767606,57.44000000],[26.07295775,53.47933333],[-17.90697183,52.87000000],[-8.80767606,50.12800000],
                          [34.86894366,49.51866667],[-29.12943662,47.99533333],[5.75119718,47.99533333],[17.27697183,47.08133333],[-20.03014085,44.03466667],
                          [-1.52823944,42.20666667],[-30.34267606,40.98800000],[31.83584507,40.37866667],[-11.53746479,40.37866667],[19.09683099,38.85533333],
                          [6.05450704,37.02733333],[41.54176056,35.80866667],[-23.97316901,35.50400000],[-43.99161972,34.59000000],[48.51788732,34.28533333],
                          [-12.75070423,31.84800000],[-35.19563380,31.84800000],[-2.74147887,30.93400000],[35.47556338,29.10600000],[20.31007042,26.66866667],
                          [-47.93464789,26.05933333],[9.99753521,25.14533333],[44.27154930,25.14533333],[53.37084507,23.01266667],[24.25309859,21.79400000],
                          [3.62802817,19.66133333],[-33.67908451,19.35666667],[-5.77457746,18.44266667],[31.53253521,17.52866667],[-41.86845070,17.22400000],
                          [40.32852113,16.31000000],[50.33774648,13.87266667],[-20.03014085,10.52133333],[17.88359155,10.82600000],[1.80816901,9.91200000],
                          [58.22380282,8.99800000],[-52.18098592,8.38866667],[30.62260563,7.47466667],[-31.85922535,6.56066667],[42.14838028,6.25600000],
                          [-43.08169014,5.95133333],[-14.87387324,4.12333333],[9.39091549,2.90466667],[-5.47126761,2.60000000],[19.09683099,1.07666667],
                          [-37.31880282,-1.05600000],[48.21457746,-0.75133333],[36.68880282,-1.36066667],[-26.09633803,-1.97000000],[57.92049296,-3.18866667],
                          [-52.48429577,-3.79800000],[2.11147887,-4.40733333],[-13.96394366,-5.01666667],[24.25309859,-5.32133333],[-45.20485915,-8.36800000],
                          [-35.49894366,-11.11000000],[-27.00626761,-12.02400000],[47.00133803,-12.32866667],[9.99753521,-13.24266667],[0.29161972,-14.15666667],
                          [38.50866197,-14.46133333],[-20.03014085,-16.28933333],[-47.93464789,-17.50800000],[-39.13866197,-19.64066667],[-6.07788732,-19.33600000],
                          [-16.08711268,-21.16400000],[53.67415493,-21.46866667],[-30.64598592,-23.90600000],[6.66112676,-25.73400000],[16.97366197,-26.64800000],
                          [40.32852113,-27.25733333],[-43.38500000,-28.78066667],[-36.40887324,-30.30400000],[49.42781690,-30.30400000],[28.80274648,-30.60866667],
                          [-1.83154930,-31.52266667],[-14.57056338,-33.35066667],[-27.30957746,-34.87400000],[15.76042254,-35.48333333],[35.47556338,-36.70200000],
                          [5.75119718,-37.00666667],[24.55640845,-39.13933333],[-12.75070423,-41.88133333],[-1.22492958,-42.79533333],[33.95901408,-43.70933333],
                          [-30.03936620,-44.01400000],[13.33394366,-45.23266667],[-21.54669014,-47.97466667],[22.73654930,-48.27933333],[-3.65140845,-52.54466667],[8.48098592,-52.84933333] ])
    if (file_header.exper_date == '22Jun2021') or (file_header.exper_date == '19Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
        r = numpy.array([ [-5.92485944,56.96500000],[6.21168675,56.35586022],[-20.48871486,52.39645161],[24.11309237,51.17817204],[-11.08289157,49.04618280],
                          [-31.71502008,47.82790323],[32.91208835,47.21876344],[3.48096386,46.30505376],[15.31409639,45.39134409],[-22.61261044,43.25935484],
                          [-33.23208835,40.51822581],[-3.80096386,40.82279570],[-13.81361446,38.99537634],[29.57453815,38.08166667],[16.83116466,36.55881720],
                          [3.78437751,35.34053763],[-26.86040161,34.42682796],[-47.49253012,34.12225806],[38.98036145,32.90397849],[45.95887550,31.68569892],
                          [-38.39012048,31.07655914],[-15.33068273,30.46741935],[-5.01461847,29.24913978],[32.91208835,26.81258065],[-51.74032129,25.59430108],
                          [18.04481928,24.37602151],[-19.57847390,23.15774194],[7.72875502,22.85317204],[41.40767068,22.54860215],[50.51008032,20.41661290],
                          [-24.43309237,19.19833333],[21.68578313,19.50290323],[-36.87305221,18.28462366],[1.05365462,17.67548387],[-45.36863454,16.45720430],
                          [-8.35216867,16.45720430],[28.96771084,15.23892473],[37.46329317,13.71607527],[47.17253012,10.97494624],[-22.91602410,8.84295699],
                          [15.31409639,8.23381720],[-1.07024096,7.92924731],[-55.98811245,7.62467742],[54.75787149,6.10182796],[-35.05257028,5.18811828],
                          [-46.27887550,4.88354839],[27.75405622,4.88354839],[38.98036145,3.66526882],[-17.75799197,2.44698925],[6.51510040,0.61956989],
                          [-8.04875502,0.61956989],[16.22433735,-1.20784946],[-40.51401606,-2.42612903],[44.44180723,-3.33983871],[-29.28771084,-3.64440860],
                          [33.51891566,-3.64440860],[-56.59493976,-4.86268817],[54.45445783,-5.77639785],[-0.46341365,-6.69010753],[-16.54433735,-6.99467742],
                          [21.38236948,-7.90838710],[-48.70618474,-9.73580645],[-38.99694779,-12.47693548],[-30.50136546,-13.99978495],[43.53156627,-14.60892473],
                          [6.81851406,-15.21806452],[-2.58730924,-16.43634409],[35.03598394,-16.43634409],[22.89943775,-17.65462366],[-22.91602410,-18.26376344],
                          [-51.74032129,-18.87290323],[-42.63791165,-21.00489247],[-8.95899598,-21.30946237],[18.04481928,-21.30946237],[-19.27506024,-23.13688172],
                          [49.59983936,-23.44145161],[-34.14232932,-25.57344086],[3.48096386,-27.70543011],[13.79702811,-28.61913978],[36.24963855,-29.22827957],
                          [-46.88570281,-30.14198925],[-39.90718876,-31.66483871],[45.35204819,-32.27397849],[25.32674699,-32.27397849],[-5.01461847,-33.49225806],
                          [-18.06140562,-35.01510753],[-30.50136546,-36.53795699],[12.58337349,-37.14709677],[31.39502008,-38.36537634],[2.26730924,-38.97451613],
                          [21.07895582,-40.80193548],[-16.24092369,-43.23849462],[-4.71120482,-44.45677419],[30.18136546,-45.06591398],[-33.53550201,-45.37048387],
                          [9.85265060,-46.58876344],[-25.03991968,-49.32989247],[18.95506024,-49.63446237],[-7.13851406,-53.89844086],[4.99803213,-54.20301075] ])
    if (file_header.exper_date == '16Jul2021') or (file_header.exper_date == '15Nov2021'):
        if (file_header.exper_date == '15Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-Nov15_cropped.png'
            r = numpy.array([ [-1.98048193,56.05129032],[10.15606426,54.83301075],[-16.24092369,51.78731183],[28.05746988,49.35075269],[-7.13851406,48.43704301],
                              [-27.46722892,47.82790323],[7.12192771,45.69591398],[36.55305221,45.08677419],[18.65164659,43.86849462],[-18.97164659,43.25935484],
                              [-29.28771084,40.82279570],[-0.46341365,40.21365591],[-10.77947791,38.99537634],[32.91208835,36.25424731],[-43.24473896,35.34053763],
                              [19.86530120,35.34053763],[-23.82626506,34.73139785],[6.51510040,34.12225806],[-34.74915663,31.99026882],[42.31791165,30.77198925],
                              [-12.59995984,30.46741935],[-2.58730924,28.64000000],[49.29642570,28.94456989],[-47.79594378,26.81258065],[35.64281124,24.68059140],
                              [-17.15116466,23.15774194],[20.16871486,23.15774194],[9.85265060,21.93946237],[-22.30919679,19.80747312],[44.13839357,19.80747312],
                              [-34.14232932,19.19833333],[-42.33449799,17.98005376],[23.50626506,17.67548387],[2.87413655,17.06634409],[53.24080321,17.06634409],
                              [-6.53168675,16.45720430],[30.78819277,13.10693548],[39.58718876,11.27951613],[-53.25738956,9.45209677],[-21.39895582,9.75666667],
                              [49.59983936,7.92924731],[0.44682731,7.62467742],[16.83116466,7.01553763],[-44.15497992,6.71096774],[-33.23208835,6.40639785],
                              [-16.54433735,2.75155914],[28.96771084,2.75155914],[57.18518072,2.44698925],[40.80084337,0.92413978],[-7.13851406,0.31500000],
                              [7.42534137,0.01043011],[-38.99694779,-0.59870968],[-54.47104418,-2.42612903],[-28.07405622,-2.42612903],[17.13457831,-2.73069892],
                              [34.73257028,-6.08096774],[-16.24092369,-6.69010753],[46.26228916,-6.69010753],[-0.16000000,-7.29924731],[-47.49253012,-7.60381720],
                              [21.98919679,-9.43123656],[55.97152610,-9.73580645],[-38.08670683,-10.95408602],[-29.89453815,-12.78150538],[6.81851406,-16.43634409],
                              [-50.52666667,-16.43634409],[-2.58730924,-16.74091398],[-22.91602410,-17.35005376],[44.13839357,-18.26376344],[-42.33449799,-19.17747312],
                              [35.64281124,-19.48204301],[23.20285141,-19.78661290],[-9.26240964,-21.61403226],[-19.27506024,-22.52774194],[17.74140562,-23.13688172],
                              [-34.14232932,-24.05059140],[50.20666667,-27.70543011],[-46.88570281,-28.01000000],[2.87413655,-28.61913978],[-40.21060241,-29.83741935],
                              [13.19020080,-30.44655914],[36.55305221,-32.27397849],[-5.92485944,-34.10139785],[-18.97164659,-34.71053763],[24.71991968,-35.01510753],
                              [-31.41160643,-35.31967742],[45.35204819,-35.92881720],[11.36971888,-38.97451613],[1.05365462,-39.88822581],[30.78819277,-41.41107527],
                              [-17.45457831,-43.23849462],[19.86530120,-43.23849462],[-34.74915663,-44.15220430],[-6.22827309,-45.06591398],[8.03216867,-48.72075269],
                              [29.27112450,-48.41618280],[-26.55698795,-48.72075269],[17.43799197,-52.37559140],[-9.26240964,-54.50758065],[2.87413655,-56.03043011] ])
        else: # 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
            r = numpy.array([ [-6.53168675,56.96500000],[5.90827309,56.66043011],[-20.79212851,52.39645161],[23.80967871,51.78731183],[-11.38630522,49.35075269],
                              [-32.32184739,47.52333333],[32.30526104,47.52333333],[3.17755020,46.91419355],[14.70726908,45.39134409],[-22.91602410,43.25935484],
                              [-4.10437751,40.82279570],[-33.53550201,40.51822581],[-14.11702811,39.29994624],[29.57453815,38.69080645],[16.52775100,36.86338710],
                              [3.48096386,35.34053763],[-27.16381526,34.42682796],[-47.79594378,34.12225806],[38.67694779,33.51311828],[45.65546185,32.29483871],
                              [-38.69353414,31.07655914],[-15.63409639,30.46741935],[-5.31803213,29.24913978],[32.60867470,27.11715054],[-51.74032129,25.28973118],
                              [17.74140562,24.68059140],[-19.88188755,22.85317204],[7.42534137,22.85317204],[41.10425703,22.85317204],[50.20666667,20.72118280],
                              [21.38236948,19.50290323],[-24.73650602,19.19833333],[-36.87305221,17.98005376],[1.05365462,17.67548387],[-8.65558233,16.76177419],
                              [-45.36863454,16.15263441],[28.66429719,15.54349462],[37.15987952,14.02064516],[47.17253012,11.58408602],[-23.21943775,8.84295699],
                              [15.31409639,8.53838710],[-1.07024096,7.92924731],[-56.29152610,7.32010753],[54.75787149,6.71096774],[27.45064257,5.18811828],
                              [-35.35598394,4.88354839],[-46.58228916,4.57897849],[38.98036145,3.96983871],[-17.75799197,2.44698925],[6.51510040,0.92413978],
                              [-8.04875502,0.61956989],[16.22433735,-1.20784946],[-40.81742972,-2.42612903],[44.74522088,-3.03526882],[33.51891566,-3.33983871],
                              [-29.28771084,-3.64440860],[-56.59493976,-5.16725806],[54.15104418,-5.47182796],[-0.76682731,-6.69010753],[-16.84775100,-6.99467742],
                              [21.38236948,-7.29924731],[-48.70618474,-10.04037634],[-38.99694779,-12.78150538],[-30.50136546,-13.99978495],[43.53156627,-14.30435484],
                              [6.81851406,-15.21806452],[-2.58730924,-16.13177419],[35.03598394,-16.13177419],[22.89943775,-17.65462366],[-22.91602410,-18.26376344],
                              [-51.74032129,-19.48204301],[-42.63791165,-21.30946237],[18.04481928,-21.30946237],[-8.95899598,-21.30946237],[-18.97164659,-23.13688172],
                              [49.59983936,-23.13688172],[-33.83891566,-25.57344086],[3.48096386,-27.40086022],[13.79702811,-28.61913978],[36.55305221,-28.61913978],
                              [-46.88570281,-30.44655914],[45.35204819,-31.66483871],[-39.90718876,-31.66483871],[25.32674699,-32.27397849],[-5.01461847,-33.49225806],
                              [-17.75799197,-35.31967742],[-30.50136546,-36.53795699],[12.58337349,-37.14709677],[31.69843373,-38.06080645],[2.57072289,-38.66994624],
                              [21.38236948,-40.49736559],[-15.93751004,-43.23849462],[-4.40779116,-44.45677419],[30.48477912,-44.76134409],[-33.23208835,-45.37048387],
                              [9.85265060,-46.58876344],[-24.73650602,-49.32989247],[19.25847390,-49.32989247],[-6.83510040,-53.59387097],[5.30144578,-53.89844086] ])
    if file_header.exper_date == '30Jul2021': 
        if (file_header.is_reverse): # 'jun-jul-aug-nov-2021/BKGDimage_asymmRev_cropped.png'
            r = numpy.array([ [-4.10437751,58.18327957],[8.03216867,57.87870968],[26.23698795,53.61473118],[-18.06140562,53.00559140],[-8.95899598,50.26446237],
                              [35.03598394,49.65532258],[-29.28771084,48.13247312],[5.60485944,48.43704301],[17.13457831,47.21876344],[-20.18530120,44.17306452],[-1.67706827,42.65021505],
                              [-30.50136546,41.43193548],[32.00184739,40.82279570],[-11.68971888,40.51822581],[18.95506024,38.99537634],[5.90827309,37.16795699],[41.40767068,35.94967742],
                              [-24.43309237,35.64510753],[-44.15497992,34.73139785],[48.68959839,34.73139785],[-12.90337349,31.99026882],[-35.35598394,31.99026882],[-2.89072289,31.07655914],
                              [35.33939759,29.55370968],[-48.09935743,26.20344086],[44.13839357,25.28973118],[9.85265060,25.28973118],[-17.15116466,24.68059140],[53.54421687,23.15774194],
                              [-22.00578313,20.72118280],[3.48096386,19.80747312],[-33.83891566,19.19833333],[-5.92485944,18.58919355],[31.69843373,17.67548387],[-42.03108434,17.37091398],
                              [40.19401606,16.45720430],[50.51008032,13.71607527],[-20.18530120,10.67037634],[18.04481928,10.67037634],[1.66048193,10.06123656],[58.39883534,8.84295699],
                              [-52.65056225,8.23381720],[30.48477912,7.32010753],[-32.32184739,6.40639785],[42.01449799,6.40639785],[-43.24473896,5.79725806],[-15.02726908,4.27440860],
                              [9.24582329,3.05612903],[-5.62144578,2.44698925],[18.95506024,1.22870968],[-37.47987952,-0.90327957],[48.08277108,-0.59870968],[36.55305221,-1.20784946],
                              [-26.25357430,-2.12155914],[58.09542169,-3.33983871],[-52.95397590,-3.94897849],[1.96389558,-4.55811828],[-14.11702811,-5.16725806],[24.41650602,-5.47182796],
                              [-45.36863454,-8.51752688],[-35.65939759,-11.25865591],[-27.16381526,-12.17236559],[47.17253012,-12.47693548],[9.85265060,-13.39064516],[0.14341365,-14.30435484],
                              [38.37353414,-14.30435484],[-19.88188755,-16.43634409],[-48.09935743,-17.95919355],[-39.30036145,-19.78661290],[-6.22827309,-19.48204301],[-16.24092369,-21.30946237],
                              [53.54421687,-21.61403226],[-30.80477912,-24.05059140],[6.51510040,-25.57344086],[16.83116466,-26.79172043],[40.19401606,-27.40086022],[-43.54815261,-28.92370968],
                              [-36.56963855,-30.44655914],[49.59983936,-30.44655914],[28.66429719,-30.75112903],[-1.98048193,-31.66483871],[-14.72385542,-33.49225806],[-27.46722892,-35.01510753],
                              [15.92092369,-35.62424731],[35.33939759,-36.84252688],[5.60485944,-37.45166667],[24.71991968,-39.58365591],[-12.90337349,-42.02021505],[-1.37365462,-43.23849462],
                              [34.12574297,-43.84763441],[-30.19795181,-44.15220430],[13.19020080,-45.37048387],[-21.70236948,-48.11161290],[22.59602410,-48.41618280],[-3.80096386,-52.68016129],[8.33558233,-52.98473118] ])
        else:                        # 'jun-jul-aug-nov-2021/BKGDimage_asymm_cropped.png'
            r = numpy.array([ [-3.49755020,58.18327957],[8.63899598,57.87870968],[26.54040161,53.61473118],[-17.75799197,53.31016129],[-8.35216867,50.56903226],
                              [35.33939759,49.65532258],[-28.68088353,48.43704301],[5.90827309,48.43704301],[17.43799197,47.52333333],[-19.88188755,44.47763441],
                              [-1.37365462,42.65021505],[-30.19795181,41.73650538],[-11.38630522,40.82279570],[32.30526104,40.51822581],[19.25847390,38.99537634],
                              [6.21168675,37.16795699],[-24.12967871,35.94967742],[41.71108434,35.94967742],[-43.85156627,35.03596774],[48.99301205,34.42682796],
                              [-35.05257028,32.29483871],[-12.59995984,32.29483871],[-2.58730924,31.38112903],[35.64281124,29.24913978],[20.47212851,26.81258065],
                              [-47.79594378,26.50801075],[10.15606426,25.28973118],[44.44180723,25.28973118],[53.84763052,23.15774194],[24.11309237,21.93946237],
                              [3.78437751,19.80747312],[-33.53550201,19.80747312],[-5.92485944,18.89376344],[-41.72767068,17.98005376],[31.69843373,17.67548387],
                              [40.49742972,16.45720430],[50.51008032,13.71607527],[-20.18530120,10.97494624],[18.04481928,10.67037634],[1.66048193,10.36580645],
                              [-52.34714859,8.84295699],[58.70224900,8.84295699],[30.78819277,7.32010753],[-32.01843373,6.71096774],[-42.94132530,6.40639785],
                              [42.31791165,6.10182796],[-15.02726908,4.57897849],[9.24582329,3.05612903],[-5.31803213,2.75155914],[18.95506024,1.22870968],
                              [-37.47987952,-0.59870968],[48.38618474,-0.90327957],[36.55305221,-1.20784946],[-26.25357430,-1.81698925],[-52.65056225,-3.33983871],
                              [58.39883534,-3.64440860],[2.26730924,-4.55811828],[-13.81361446,-5.16725806],[24.41650602,-5.47182796],[-45.36863454,-7.90838710],
                              [-35.65939759,-10.64951613],[-27.16381526,-11.86779570],[47.17253012,-12.78150538],[9.85265060,-13.08607527],[0.14341365,-13.99978495],
                              [38.67694779,-14.60892473],[25.93357430,-15.52263441],[-48.09935743,-17.35005376],[-39.30036145,-19.48204301],[-6.22827309,-19.48204301],
                              [21.07895582,-19.48204301],[53.84763052,-21.91860215],[-30.80477912,-23.44145161],[6.51510040,-25.57344086],[16.83116466,-26.79172043],
                              [40.19401606,-27.40086022],[-43.54815261,-28.61913978],[-36.87305221,-29.83741935],[49.59983936,-30.75112903],[28.66429719,-30.75112903],
                              [-1.98048193,-31.66483871],[-15.02726908,-33.18768817],[-27.46722892,-35.01510753],[15.61751004,-35.62424731],[35.33939759,-37.14709677],
                              [5.60485944,-37.14709677],[24.41650602,-39.58365591],[-13.20678715,-41.71564516],[-1.67706827,-42.93392473],[-30.19795181,-43.84763441],
                              [33.82232932,-44.15220430],[12.88678715,-45.37048387],[-21.70236948,-48.11161290],[22.59602410,-48.72075269],[-3.80096386,-52.68016129],[8.03216867,-52.98473118] ])
    if file_header.exper_date == '11Aug2021': 
        if file_header.is_reverse: # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_Reverse_cropped.png'
            r = numpy.array([ [-5.92485944,56.96500000],[6.51510040,56.05129032],[-20.18530120,52.39645161],[24.41650602,51.17817204],[-10.77947791,49.04618280],
                              [-31.71502008,47.82790323],[32.91208835,46.91419355],[3.78437751,46.30505376],[15.31409639,45.08677419],[-22.61261044,43.25935484],
                              [-33.23208835,40.82279570],[-3.80096386,40.82279570],[-13.81361446,38.99537634],[29.87795181,37.77709677],[16.83116466,36.55881720],
                              [3.78437751,35.03596774],[-47.49253012,34.73139785],[-26.86040161,34.42682796],[38.98036145,32.90397849],[-38.39012048,31.38112903],
                              [45.95887550,31.38112903],[-15.33068273,30.46741935],[-5.01461847,29.24913978],[32.91208835,26.20344086],[-51.43690763,25.89887097],
                              [17.74140562,24.37602151],[7.72875502,22.54860215],[41.40767068,21.93946237],[50.51008032,19.80747312],[21.38236948,19.19833333],
                              [-36.87305221,18.28462366],[1.05365462,17.37091398],[-45.36863454,16.76177419],[-8.65558233,16.76177419],[28.96771084,14.62978495],
                              [47.17253012,10.36580645],[-23.21943775,9.14752688],[-1.07024096,7.92924731],[15.01068273,8.23381720],[-56.29152610,7.92924731],
                              [54.45445783,5.79725806],[-35.35598394,5.18811828],[-46.58228916,4.88354839],[27.45064257,4.57897849],[-18.06140562,2.44698925],
                              [-8.35216867,0.31500000],[6.21168675,0.61956989],[15.92092369,-1.51241935],[-41.12084337,-2.12155914],[-29.59112450,-3.33983871],[44.44180723,-3.94897849],
                              [-56.89835341,-4.25354839],[53.84763052,-6.38553763],[-1.07024096,-6.99467742],[-17.15116466,-6.99467742],[21.07895582,-8.21295699],[-49.00959839,-9.43123656],
                              [-39.30036145,-12.17236559],[-30.80477912,-13.69521505],[6.51510040,-15.52263441],[42.92473896,-15.21806452],[-2.89072289,-16.43634409],[34.42915663,-17.04548387],
                              [22.59602410,-18.26376344],[-52.34714859,-18.56833333],[-43.24473896,-20.70032258],[-9.56582329,-21.61403226],[17.43799197,-21.91860215],[48.99301205,-24.05059140],
                              [-34.74915663,-25.26887097],[2.87413655,-27.70543011],[13.19020080,-28.92370968],[35.94622490,-29.53284946],[-47.49253012,-30.14198925],[-40.51401606,-31.36026882],
                              [24.71991968,-32.88311828],[44.74522088,-32.57854839],[-5.62144578,-33.49225806],[-18.66823293,-35.01510753],[-31.10819277,-36.53795699],[11.67313253,-37.45166667],
                              [1.66048193,-38.97451613],[30.78819277,-38.66994624],[20.16871486,-41.10650538],[-16.84775100,-43.23849462],[-5.31803213,-44.45677419],[-34.44574297,-45.06591398],
                              [29.27112450,-45.37048387],[8.94240964,-46.89333333],[-25.64674699,-49.32989247],[18.04481928,-49.93903226],[-8.04875502,-53.89844086],[4.08779116,-54.20301075] ])
        else:                      # 'jun-jul-aug-nov-2021/BKGDimage_3LocalCues_cropped.png'
            r = numpy.array([ [-6.22827309,56.96500000],[5.90827309,56.35586022],[-20.79212851,52.39645161],[24.11309237,51.48274194],[-11.38630522,49.35075269],
                              [-32.01843373,47.82790323],[32.60867470,47.52333333],[3.17755020,46.60962366],[15.01068273,45.39134409],[-22.91602410,43.25935484],
                              [-3.80096386,40.82279570],[-33.53550201,40.82279570],[-14.11702811,39.29994624],[29.57453815,38.08166667],[16.52775100,36.86338710],
                              [3.48096386,35.34053763],[-47.49253012,34.42682796],[-27.16381526,34.73139785],[38.67694779,33.20854839],[45.65546185,31.99026882],
                              [-38.39012048,31.38112903],[-15.63409639,30.46741935],[-5.31803213,29.24913978],[32.60867470,26.81258065],[-51.74032129,25.59430108],
                              [-19.88188755,23.15774194],[7.42534137,22.85317204],[41.10425703,22.54860215],[50.20666667,20.41661290],[-24.73650602,19.19833333],
                              [-37.17646586,18.28462366],[0.75024096,17.67548387],[-8.65558233,16.76177419],[-45.36863454,16.45720430],[28.66429719,15.23892473],
                              [37.15987952,14.02064516],[46.86911647,11.27951613],[-23.21943775,9.14752688],[15.01068273,8.23381720],[-1.07024096,7.92924731],
                              [-56.29152610,7.62467742],[54.45445783,6.40639785],[-46.58228916,4.88354839],[27.45064257,4.88354839],[38.98036145,3.66526882],
                              [-18.06140562,2.75155914],[6.21168675,0.92413978],[-8.35216867,0.31500000],[15.92092369,-1.20784946],[44.44180723,-3.33983871],
                              [-29.59112450,-3.33983871],[33.21550201,-3.64440860],[-56.59493976,-4.55811828],[53.84763052,-5.77639785],[-0.76682731,-6.69010753],
                              [-17.15116466,-6.99467742],[21.07895582,-7.60381720],[-49.00959839,-9.73580645],[-30.50136546,-13.69521505],[43.22815261,-14.91349462],
                              [6.51510040,-15.21806452],[-2.89072289,-16.13177419],[34.73257028,-16.43634409],[-23.21943775,-17.95919355],[-52.04373494,-18.87290323],
                              [-42.94132530,-21.00489247],[-9.26240964,-21.30946237],[-19.27506024,-23.13688172],[49.29642570,-23.44145161],[-34.44574297,-25.26887097],
                              [3.17755020,-27.40086022],[13.49361446,-28.92370968],[36.24963855,-29.22827957],[-47.18911647,-30.14198925],[-40.21060241,-31.66483871],
                              [45.04863454,-31.96940860],[25.02333333,-32.57854839],[-5.31803213,-33.49225806],[-18.36481928,-35.01510753],[-31.10819277,-36.53795699],
                              [11.97654618,-37.14709677],[31.09160643,-38.06080645],[1.96389558,-38.66994624],[20.77554217,-40.80193548],[-16.54433735,-43.23849462],
                              [-5.01461847,-44.45677419],[-33.83891566,-45.37048387],[29.87795181,-45.06591398],[9.24582329,-46.58876344],[-25.34333333,-49.32989247],
                              [18.65164659,-49.63446237],[-7.44192771,-53.59387097],[4.69461847,-54.20301075] ])
    if file_header.exper_date == '12Aug2022': 
        r = numpy.array( [[-3.03628629,56.30028571],[9.39873874,55.69175000],[-17.29107107,51.73626786],[27.29304304,50.51919643],
                            [-8.19227227,48.69358929],[-28.51292292,47.47651786],[6.06251251,45.95517857],[35.78525526,45.95517857],
                            [17.58765766,44.43383929],[-19.71741742,43.21676786],[-30.33268268,40.78262500],[-1.21652653,40.47835714],
                            [-11.22520521,38.95701786],[32.14573574,37.13141071],[19.10412412,35.91433929],[-43.98088088,35.00153571],
                            [5.75921922,34.69726786],[-24.26681682,34.69726786],[-35.48866867,31.65458929],[41.54782783,31.95885714],
                            [-13.04496496,30.43751786],[48.82686687,30.13325000],[-3.03628629,28.91617857],[-48.22698699,26.48203571],
                            [35.17866867,25.56923214],[19.71071071,23.74362500],[-17.59436436,23.13508929],[9.39873874,22.52655357],
                            [43.97417417,21.00521429],[-22.75035035,19.48387500],[-34.57878879,18.87533929],[23.35023023,18.57107143],
                            [53.07297297,18.57107143],[-42.76770771,17.35400000],[2.72628629,17.35400000],[-6.97909910,16.74546429],
                            [30.62926927,14.00705357],[39.42477477,12.18144643],[-21.53717718,9.44303571],[49.43345345,9.13876786],
                            [-53.68626627,8.83450000],[0.29993994,7.92169643],[16.67777778,7.61742857],[-44.58746747,6.09608929],
                            [-33.66890891,6.09608929],[29.11280280,3.66194643],[57.31907908,3.96621429],[-16.68448448,2.74914286],
                            [40.94124124,2.14060714],[-7.28239239,0.61926786],[7.57897898,0.31500000],[-39.12818819,-1.20633929],
                            [17.28436436,-2.11914286],[-28.20962963,-2.72767857],[-54.59614615,-3.33621429],[34.87537538,-5.46608929],
                            [46.40052052,-5.46608929],[-16.07789790,-6.68316071],[-0.00335335,-6.98742857],[-47.31710711,-8.20450000],
                            [56.40919920,-8.50876786],[22.13705706,-8.81303571],[-37.91501502,-11.24717857],[-29.72609610,-13.07278571],
                            [7.27568569,-15.81119643],[-2.42969970,-16.41973214],[-50.65333333,-17.33253571],[44.88405405,-17.02826786],
                            [-22.75035035,-17.63680357],[36.08854855,-18.54960714],[23.65352352,-19.15814286],[-42.16112112,-19.76667857],
                            [-9.10215215,-21.59228571],[-19.11083083,-22.80935714],[18.19424424,-22.80935714],[-33.66890891,-24.33069643],
                            [50.94991992,-26.46057143],[3.33287287,-28.28617857],[-46.40722723,-28.59044643],[13.64484484,-29.80751786],
                            [-39.73477477,-30.41605357],[36.99842843,-31.63312500],[-5.46263263,-34.06726786],[25.47328328,-34.37153571],
                            [-18.20095095,-34.67580357],[46.40052052,-34.98007143],[-30.93926927,-35.89287500],[12.12837838,-38.63128571],
                            [1.81640641,-39.84835714],[31.84244244,-40.76116071],[20.92388388,-42.89103571],[-16.68448448,-43.19530357],
                            [-33.97220220,-44.41237500],[-5.46263263,-45.02091071],[30.02268268,-48.06358929],[9.09544545,-48.36785714],
                            [-25.78328328,-49.28066071],[18.49753754,-51.71480357],[-8.19227227,-54.75748214],[3.93945946,-55.67028571]] )
    if file_header.exper_date == '11Oct2022': 
        r = numpy.array( [ [-4.69405765,56.81134387],[7.74003326,56.20272727],[-19.25104213,52.55102767],[25.63299335,51.02948617],
                            [-9.84965632,49.50794466],[-30.77532151,48.29071146],[4.70732816,46.46486166],[34.12456763,46.46486166],
                            [16.53487805,44.94332016],[-21.67720621,43.42177866],[-32.29167406,41.29162055],[-2.57116408,40.98731225],
                            [-12.88236142,39.46577075],[30.78859202,37.33561265],[18.05123060,36.42268775],[-46.54538803,35.20545455],
                            [-26.22626386,34.90114625],[4.70732816,34.90114625],[39.88670732,32.16237154],[-37.75054324,31.85806324],
                            [-14.70198448,30.64083004],[46.86192905,30.64083004],[-4.08751663,29.42359684],[-50.79117517,26.68482213],
                            [33.82129712,25.77189723],[18.65777162,23.94604743],[-18.94777162,23.33743083],[8.34657428,22.42450593],
                            [42.31287140,21.51158103],[-24.10337029,19.68573123],[51.10771619,19.07711462],[-36.53746120,18.77280632],
                            [22.29701774,18.77280632],[-45.02903548,17.25126482],[1.67462306,17.25126482],[-8.03003326,16.64264822],
                            [29.57550998,14.20818182],[38.37035477,12.99094862],[47.77174058,9.94786561],[-22.89028825,9.33924901],
                            [-55.94677384,8.42632411],[-0.44827051,7.81770751],[15.92833703,7.81770751],[-46.24211752,5.68754941],
                            [-35.02110865,5.68754941],[55.05023282,4.77462451],[28.36242794,4.16600791],[-17.73468958,2.64446640],
                            [39.58343681,2.64446640],[-8.03003326,0.21000000],[6.83022173,0.21000000],[-40.78324834,-1.92015810],
                            [16.53487805,-1.92015810],[-29.25896896,-3.44169960],[-56.55331486,-3.74600791],[45.04230599,-4.65893281],
                            [33.82129712,-4.65893281],[-16.82487805,-7.09339921],[-0.44827051,-7.09339921],[54.44369180,-7.39770751],
                            [21.69047672,-8.61494071],[-48.97155211,-8.91924901],[-39.26689579,-11.96233202],[-30.77532151,-13.78818182],
                            [6.83022173,-16.22264822],[43.52595344,-15.91833992],[-2.87443459,-16.83126482],[35.03437916,-17.74418972],
                            [-52.00425721,-18.04849802],[-23.19355876,-18.04849802],[22.90355876,-18.65711462],[-43.20941242,-20.48296443],
                            [-9.24311530,-21.70019763],[17.74796009,-22.30881423],[-19.55431264,-23.22173913],[-34.71783814,-25.04758893],
                            [49.28809313,-24.74328063],[3.19097561,-28.39498024],[-47.45519956,-29.61221344],[13.50217295,-29.30790514],
                            [36.24746120,-30.22083004],[-40.47997783,-31.13375494],[24.72318182,-33.26391304],[44.73903548,-33.26391304],
                            [-5.60386918,-33.87252964],[-18.64450111,-35.08976285],[-31.38186253,-36.30699605],[11.98582040,-37.82853755],
                            [1.67462306,-39.35007905],[30.78859202,-39.35007905],[20.47739468,-41.78454545],[-16.82487805,-43.30608696],
                            [-34.41456763,-44.82762846],[-5.60386918,-44.82762846],[29.27223947,-46.04486166],[8.95311530,-47.26209486],
                            [-25.92299335,-49.08794466],[18.05123060,-50.30517787],[-8.03003326,-53.95687747],[4.10078714,-54.56549407] ])
    if (file_header.exper_date == '20Sept2022'):
        if _is_probe2_trial(file_header.trial): # fname = '2022/BKGDimage-20220920-probe2_cropped.png'
            r = numpy.array([[-3.12661043,55.94758065],[9.01388037,55.33887097],[-17.39168712,51.68661290],[26.92110429,50.16483871],
                            [-8.28631902,48.64306452],[-28.62164110,47.42564516],[5.97875767,45.90387097],[35.72296012,45.90387097],
                            [17.51222393,44.38209677],[-19.81978528,42.86032258],[-30.44271472,40.42548387],[-1.30553681,40.42548387],
                            [-11.62495399,38.59935484],[32.08081288,36.77322581],[19.02978528,35.86016129],[-44.10076687,34.64274194],
                            [5.67524540,34.64274194],[-24.37246933,34.64274194],[41.48969325,31.90354839],[-35.60242331,31.29483871],
                            [-13.44602761,30.38177419],[48.77398773,30.07741935],[-3.12661043,28.86000000],[-48.65345092,26.12080645],
                            [34.81242331,25.51209677],[19.63680982,23.38161290],[-17.69519939,23.07725806],[9.31739264,22.46854839],
                            [43.91779141,20.94677419],[-22.85490798,19.72935484],[-34.69188650,18.81629032],[23.27895706,18.51193548],
                            [53.02315951,18.51193548],[-43.19023006,17.29451613],[2.33661043,17.29451613],[-7.07226994,16.68580645],
                            [30.56325153,13.94661290],[39.36510736,12.42483871],[-21.64085890,9.38129032],[49.38101227,9.07693548],
                            [-53.81315951,8.77258065],[0.21202454,7.85951613],[16.29817485,7.55516129],[-44.70779141,5.72903226],
                            [-33.78134969,5.72903226],[57.27233129,3.90290323],[29.04569018,3.59854839],[-16.78466258,2.68548387],
                            [40.57915644,2.07677419],[-7.37578221,0.25064516],[7.49631902,0.25064516],[-39.24457055,-1.27112903],
                            [16.90519939,-2.18419355],[-28.31812883,-2.79290323],[-54.42018405,-3.09725806],[34.81242331,-5.22774194],
                            [46.34588957,-5.53209677],[-16.17763804,-6.74951613],[-0.09148773,-7.05387097],[-47.43940184,-8.27129032],
                            [56.36179448,-8.27129032],[22.06490798,-8.88000000],[-38.03052147,-11.31483871],[-29.83569018,-13.14096774],
                            [7.19280675,-15.88016129],[-2.51958589,-16.48887097],[44.82832822,-17.09758065],[-50.47452454,-17.40193548],
                            [-22.55139571,-17.40193548],[36.02647239,-18.61935484],[23.27895706,-19.22806452],[-41.97618098,-19.83677419],
                            [-9.19685583,-21.35854839],[-19.21276074,-22.88032258],[18.11924847,-22.88032258],[-33.78134969,-24.40209677],
                            [50.89857362,-26.53258065],[3.55065951,-28.35870968],[-46.52886503,-28.96741935],[13.56656442,-29.88048387],
                            [-39.85159509,-30.48919355],[36.93700920,-31.40225806],[-5.55470859,-33.83709677],[25.40354294,-34.44580645],
                            [-18.30222393,-34.75016129],[46.34588957,-35.05451613],[-30.74622699,-35.66322581],[12.04900307,-38.40241935],
                            [1.72958589,-39.61983871],[31.77730061,-40.53290323],[20.85085890,-42.66338710],[-16.78466258,-43.27209677],
                            [-34.08486196,-44.48951613],[-5.55470859,-45.09822581],[29.95622699,-47.83741935],[9.01388037,-48.14177419],
                            [-25.58651840,-49.05483871],[18.42276074,-51.48967742],[-8.28631902,-54.53322581],[4.15768405,-55.44629032]])
        else:                                   # fname = '2022/BKGDimage-20220920_cropped.png'
            r = numpy.array([[-8.58983129,55.94758065],[3.85417178,55.33887097],[-22.85490798,51.68661290],[21.76139571,50.16483871],
                         [-13.74953988,48.33870968],[-34.08486196,47.42564516],[30.25973926,45.90387097],[0.51553681,45.90387097],[12.35251534,44.38209677],
                         [-25.28300613,42.86032258],[-35.90593558,40.42548387],[-6.76875767,40.42548387],[-17.08817485,38.90370968],[26.61759202,37.07758065],
                         [13.56656442,35.86016129],[-49.56398773,34.64274194],[0.51553681,34.64274194],[-29.83569018,34.33838710],[36.32998466,31.90354839],
                         [-41.06564417,31.29483871],[-18.60573620,30.38177419],[43.31076687,30.07741935],[-8.58983129,28.86000000],[-53.81315951,26.42516129],
                         [29.65271472,25.51209677],[14.17358896,23.68596774],[-23.15842025,23.07725806],[3.85417178,22.16419355],[38.45457055,20.94677419],
                         [-28.01461656,19.72935484],[-40.15510736,18.51193548],[17.81573620,18.51193548],[47.86345092,18.51193548],[-48.65345092,17.29451613],
                         [-2.82309816,17.29451613],[-12.53549080,16.68580645],[25.10003067,13.94661290],[33.90188650,12.42483871],[-27.10407975,9.38129032],
                         [44.22130368,9.07693548],[-58.97286810,8.77258065],[-5.25119632,7.55516129],[11.13846626,7.55516129],[-49.86750000,6.03338710],
                         [-38.94105828,6.03338710],[51.80911043,3.90290323],[23.58246933,3.59854839],[-21.94437117,2.68548387],[35.41944785,2.07677419],
                         [-12.83900307,0.25064516],[2.03309816,0.25064516],[-44.70779141,-1.27112903],[11.74549080,-2.18419355],[-33.47783742,-2.79290323],
                         [-59.88340491,-3.09725806],[29.34920245,-5.22774194],[40.88266871,-5.53209677],[-21.64085890,-6.74951613],[-5.55470859,-7.05387097],
                         [-52.90262270,-8.27129032],[50.89857362,-8.27129032],[16.90519939,-8.88000000],[-43.19023006,-11.31483871],[-35.29891104,-12.83661290],
                         [1.72958589,-15.88016129],[-7.67929448,-16.48887097],[39.36510736,-17.09758065],[-55.93774540,-17.40193548],[-28.01461656,-17.40193548],
                         [30.56325153,-18.61935484],[18.11924847,-18.92370968],[-47.43940184,-19.83677419],[-14.35656442,-21.66290323],[-24.37246933,-22.57596774],
                         [12.95953988,-22.57596774],[-39.24457055,-24.40209677],[45.43535276,-26.53258065],[-1.91256135,-28.35870968],[-51.68857362,-28.66306452],
                         [8.40685583,-29.88048387],[-45.31481595,-30.48919355],[31.77730061,-31.40225806],[-10.71441718,-33.83709677],[19.94032209,-34.44580645],
                         [-23.76544479,-34.75016129],[40.88266871,-35.05451613],[-36.51296012,-35.96758065],[6.58578221,-38.40241935],[-3.43012270,-39.61983871],
                         [26.31407975,-40.53290323],[15.38763804,-42.96774194],[-22.24788344,-43.27209677],[-39.54808282,-44.48951613],[-10.71441718,-44.79387097],
                         [24.79651840,-47.53306452],[3.55065951,-48.14177419],[-31.04973926,-49.05483871],[12.95953988,-51.48967742],[-13.44602761,-54.22887097],
                         [-1.30553681,-55.44629032]])
    if (file_header.exper_date == '04Nov2022'): # fname = '2022/BKGDimage-20221104_cropped.png'
        r = numpy.array([[-3.40463739,56.02539162],[8.66544433,55.40466302],[-17.64114402,51.68029144],[26.61582227,50.43883424],[-8.97544433,48.57664845],
                         [-28.78275792,47.33519126],[5.57055158,46.09373406],[35.28152196,46.09373406],[17.33114402,44.54191257],[-20.11705822,42.99009107],
                         [-30.94918284,40.50717668],[-1.85719101,40.50717668],[-11.76084780,38.95535519],[31.87713994,37.09316940],[18.56910112,35.85171220],
                         [5.26106231,34.61025501],[-44.56671093,34.61025501],[-24.75939734,34.29989071],[41.16181818,31.81697632],[-35.90101124,31.50661202],
                         [-13.61778345,30.26515483],[48.28007150,29.95479053],[-3.40463739,29.02369763],[-48.89956078,26.23041894],[34.66254341,25.60969035],
                         [19.18807967,23.74750455],[-17.95063330,23.12677596],[8.97493361,22.50604736],[43.32824311,20.95422587],[-23.21195097,19.40240437],
                         [-34.97254341,18.78167577],[22.90195097,18.47131148],[52.61292135,18.47131148],[2.16616956,17.22985428],[-43.32875383,17.22985428],
                         [-7.42799796,16.60912568],[30.32969356,14.12621129],[38.99539326,12.26402550],[-21.97399387,9.47074681],[49.20853933,9.16038251],
                         [-54.16087845,8.53965392],[-0.00025536,7.91892532],[16.09318693,7.60856102],[-44.87620020,5.74637523],[-34.04407559,5.74637523],
                         [28.78224719,3.57382514],[56.94577120,3.88418944],[-17.02216547,2.64273224],[40.54283963,2.02200364],[-7.42799796,0.47018215],
                         [7.11799796,0.15981785],[-39.61488253,-1.39200364],[16.71216547,-2.32309654],[-28.47326864,-2.94382514],[-54.77985700,-3.56455373],
                         [34.35305414,-5.42673953],[46.11364658,-5.42673953],[-16.40318693,-6.66819672],[-0.30974464,-6.97856102],[-47.66160368,-8.53038251],
                         [56.01730337,-8.53038251],[21.97348315,-8.84074681],[-38.37692543,-11.63402550],[-30.02071502,-13.18584699],[6.80850868,-15.97912568],
                         [-2.78565884,-16.59985428],[44.25671093,-17.22058288],[-50.75649642,-17.53094718],[-22.90246170,-17.84131148],[35.59101124,-18.46204007],
                         [23.21144025,-19.08276867],[-42.40028601,-20.01386157],[-9.28493361,-21.56568306],[-19.18859040,-22.80714026],[17.95012257,-22.80714026],
                         [-34.04407559,-24.35896175],[50.44649642,-26.53151184],[3.09463739,-28.39369763],[-46.73313585,-29.01442623],[13.30778345,-30.25588342],
                         [-40.23386108,-30.87661202],[36.82896834,-31.49734062],[-5.57106231,-33.98025501],[25.06837589,-34.60098361],[-18.56961185,-35.22171220],
                         [46.11364658,-35.22171220],[-31.25867211,-36.15280510],[11.76033708,-38.63571949],[1.54719101,-39.87717668],[31.56765066,-40.80826958],
                         [20.42603677,-42.98081967],[-17.02216547,-43.29118397],[-34.35356486,-44.84300546],[-5.57106231,-45.15336976],[29.40122574,-47.94664845],
                         [8.97493361,-48.25701275],[-25.99735444,-49.18810565],[18.25961185,-51.98138434],[-8.35646578,-54.77466302],[3.71361593,-55.70575592]])
    return r

def get_2target_target_index(trial,flip_targets=False,experiment_date=None):
    experiment_date = experiment_date if misc.exists(experiment_date) else '22Jun2021' # defaults to this date experiment
    target_index    = 1
    if is_trial_of_type(trial,'numeric'):
        target_index = 1 if int(trial) <= 18 else 2 # still holds for '22Jun2021', '19Nov2021', '12Aug2022', '11Oct2022'
    if is_trial_of_type(trial,'probe'):
        if (get_named_trial_trialnumber(trial) == 1): # first probe trial
            target_index = 1
        else:
            target_index = 2
    if is_trial_of_type(trial,'reverse'):
        target_index = 2
    if flip_targets:
        target_index = 3-target_index
    #print('*****************************            target index == %d'%target_index)
    return target_index

def get_2target_experiment_target(entrance_label,experiment_date,target_index,is_probe2=False):
    assert target_index in [1,2,'A','B'],"'target' must be either 1 or A, or 2 or B"
    if target_index == 'A':
        target_index = 1
    elif target_index == 'B':
        target_index = 2
    get_coords = lambda cond,a,b: a if cond else b
    target_coords = (numpy.nan,numpy.nan)
    if (experiment_date == '22Jun2021') or (experiment_date == '19Nov2021'):
        if target_index == 1: # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
            if entrance_label == u'SW':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (-38.99694779,-12.47693548) #get_coords(experiment_date=='22Jun2021', (-38.64,-11.62), (-38.32, -13.51))
            elif entrance_label == u'SE':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (12.33199054, -36.39458349) #get_coords(experiment_date=='22Jun2021', (12.25, -37.07), (12.88, -37.54))
            elif entrance_label == u'NE':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (36.24963855,  14.93435483) #get_coords(experiment_date=='22Jun2021', (37.54, 13.19) , (37.54, 13.51))
            elif entrance_label == u'NW':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (-15.07929978, 38.85200285) #get_coords(experiment_date=='22Jun2021', (-13.19, 39.27), (-13.51, 38.80))
        else:
            if entrance_label == u'SW':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (21.38236948,  19.19833333) #get_coords(experiment_date=='22Jun2021', (22.15, 18.85)  , (21.99, 18.85))
            elif entrance_label == u'SE':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (-19.34327827, 23.98473378) #get_coords(experiment_date=='22Jun2021', (-19.01, 23.40) , (-19.32, 22.77))
            elif entrance_label == u'NE':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (-24.12967872,-16.74091398) #get_coords(experiment_date=='22Jun2021', (-22.93, -17.59), (-22.62, -18.53))
            elif entrance_label == u'NW':
                #               estimated by crop notebook after undistorting the second                 '22Jun2021'       '19Nov2021'
                target_coords = (16.59596903, -21.52731442) #get_coords(experiment_date=='22Jun2021', (17.91, -21.36) , (18.38, -21.83))
    if (experiment_date == '12Aug2022'):
        if target_index == 1: #target A
            if entrance_label == u'SW':
                target_coords = (39.12148148,12.18144643)
            if entrance_label == u'SE':
                target_coords = (-11.52849850,38.65275000)
            if entrance_label == u'NE':
                target_coords = (-37.91501502,-11.24717857)
            if entrance_label == u'NW':
                target_coords = (11.82508509,-38.63128571)
        else: #target B
            if entrance_label == u'SW':
                target_coords = (-22.44705706,-17.63680357)
            if entrance_label == u'SE':
                target_coords = (18.19424424,-22.80935714)
            if entrance_label == u'NE':
                target_coords = (23.35023023,18.57107143)
            if entrance_label == u'NW':
                target_coords = (-17.89765766,23.13508929)
    if (experiment_date == '11Oct2022'):
        if target_index == 1: #target A
            if entrance_label == u'SW':
                target_coords = (-39.26689579,-11.96233202)
            if entrance_label == u'SE':
                target_coords = (11.68254989,-37.82853755)
            if entrance_label == u'NE':
                target_coords = (38.06708426,12.68664032)
            if entrance_label == u'NW':
                target_coords = (-13.18563193,39.46577075)
        else: #target B
            if entrance_label == u'SW':
                target_coords = (22.29701774,18.77280632)
            if entrance_label == u'SE':
                target_coords = (-19.25104213,23.03312253)
            if entrance_label == u'NE':
                target_coords = (-23.49682927,-18.04849802)
            if entrance_label == u'NW':
                target_coords = (17.74796009,-22.30881423)
    if (experiment_date == '20Sept2022'):
        if is_probe2:
            if target_index == 1: #target A
                if entrance_label == u'SW':
                    target_coords = (39.06159509,12.12048387)
                if entrance_label == u'SE':
                    target_coords = (-11.62495399,38.59935484)
                if entrance_label == u'NE':
                    target_coords = (-38.33403374,-11.31483871)
                if entrance_label == u'NW':
                    target_coords = (11.74549080,-38.40241935)
            else: #target B
                if entrance_label == u'SW':
                    target_coords = (-22.85490798,-17.70629032)
                if entrance_label == u'SE':
                    target_coords = (18.42276074,-22.88032258)
                if entrance_label == u'NE':
                    target_coords = (23.27895706,18.20758065)
                if entrance_label == u'NW':
                    target_coords = (-17.69519939,22.77290323)
                
        else:
            if target_index == 1: #target A
                if entrance_label == u'SW':
                    target_coords = (33.90188650,12.12048387)
                if entrance_label == u'SE':
                    target_coords = (-16.78466258,38.59935484)
                if entrance_label == u'NE':
                    target_coords = (-43.79725460,-11.31483871)
                if entrance_label == u'NW':
                    target_coords = (6.58578221,-38.40241935)
            else: #target B
                if entrance_label == u'SW':
                    target_coords = (-28.31812883,-17.70629032)
                if entrance_label == u'SE':
                    target_coords = (12.95953988,-22.57596774)
                if entrance_label == u'NE':
                    target_coords = (17.81573620,18.20758065)
                if entrance_label == u'NW':
                    target_coords = (-23.15842025,22.77290323)
    return numpy.array(target_coords)

def get_arena_alt_target(file_header):
    entrance        = file_header.start_location
    trial_condition = file_header.trial
    experiment      = file_header.exper_date
    is_probe2       = _is_probe2_trial(trial_condition) # used only for experiment == '20Sept2022'
    target_coords   = (numpy.nan,numpy.nan)
    if (experiment == '22Jun2021') or (experiment == '19Nov2021') or (experiment == '12Aug2022') or (experiment == '11Oct2022') or (experiment == '20Sept2022'):
        target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=True),is_probe2=is_probe2))
    if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
        if (experiment == '12Aug2022') or (experiment == '11Oct2022') or (experiment == '20Sept2022'):
            # probe2 of these experiments is 180 degrees rotated; so we rotate 180 deg here to undo the experiment rotation and get the correct result
            # however, the entrance labels are already rotated in xlsx raw experimental files, so we actually do not rotate labels
            #target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=True),is_probe2=is_probe2))
            target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=True),is_probe2=is_probe2))
    return numpy.array(target_coords)

def get_arena_alt_reverse_target(file_header):
    entrance        = file_header.start_location
    trial_condition = file_header.trial
    experiment      = file_header.exper_date
    is_probe2       = _is_probe2_trial(trial_condition) # used only for experiment == '20Sept2022'
    target_coords   = (numpy.nan,numpy.nan)
    if (experiment == '22Jun2021') or (experiment == '19Nov2021') or (experiment == '12Aug2022') or (experiment == '11Oct2022') or (experiment == '20Sept2022'):
        # the alt reverse is the 180 deg rotation of the alternative target (alt == 2 if tgt == 1; or alt == 1 if tgt == 2)
        target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=True),is_probe2=is_probe2))
    if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
        if (experiment == '12Aug2022') or (experiment == '11Oct2022') or (experiment == '20Sept2022'):
            # probe2 of these experiments is 180 degrees rotated; so we rotate 180 deg here to undo the experiment rotation and get the correct result
            target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=True),is_probe2=is_probe2))
    return numpy.array(target_coords)

def _get_main_target_set_for_rotating_probes(entrance):
    if entrance == 'SW':
        target_coords = 11.07, -30.48
    elif entrance == 'SE':
        target_coords = 35.64, 2.58
    elif entrance == 'NE':
        target_coords = 2.88, 27.45
    elif entrance == 'NW':
        target_coords = -21.68, -5.61     
    return target_coords

def _get_closest_arena_hole_coord(file_header,r0):
    if not(type(r0) is numpy.ndarray):
        r0 = numpy.asarray(r0)
    r = get_arena_hole_coord(file_header)
    return r[numpy.argmin(numpy.linalg.norm(r - r0,axis=1))]

#manually sets food target coordinates based on experiment
def get_arena_target(file_header):
    entrance        = file_header.start_location
    trial_condition = file_header.trial
    experiment      = file_header.exper_date
    target_coords   = (numpy.nan,numpy.nan)
    
    if ('23May2019' == experiment) or ('Pilot' in experiment):
        target_coords = 20.47, -39.91 #default coordinates
    elif ('06Sept2019' == experiment) or ('07Oct2019' == experiment) or ('08Jul2019' == experiment) or ('18Jun2019' == experiment):
        #if trial_condition.isdigit() or ('probe' in trial_condition.lower()):
        target_coords  = _get_main_target_set_for_rotating_probes(entrance)
        rotation_angle = 0.0
        if trial_condition.startswith('R90'):
            rotation_angle = -numpy.pi/2.0
        if trial_condition.startswith('R180'):
            rotation_angle = -numpy.pi
        if trial_condition.startswith('R270'):
            rotation_angle = numpy.pi/2.0
        target_coords = _get_main_target_set_for_rotating_probes(_rotate_compass_label_default(entrance,rotation_angle))
    elif experiment == '11Dec2019': # 'dec-2019/BKGDimage-localCues_cropped.png'
        target_coords = 2.88, 27.45
    elif (experiment == '08Mar2021') or (experiment == '06May2021') or (experiment == '26May2021'):
        target_coords = 24.47, 21.80
    elif experiment == '16Jul2021': # 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
        target_coords = -29.28771084,-3.64440860
    elif experiment == '15Nov2021': # jun-jul-aug-nov-2021/BKGDimage-Nov15_cropped.png
        target_coords = 28.96771084,3.05612903
    elif (experiment == '22Jun2021') or (experiment == '19Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
        target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif (experiment == '12Aug2022'):
        target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False)))
        if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
            target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif (experiment == '11Oct2022'):
        target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False)))
        if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
            target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif experiment == '30Jul2021':
        target_coords = -2.2, 30.63
    elif experiment == '11Aug2021':
        target_coords = 28.12, 4.71
    elif (experiment == '20Sept2022'):
        target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False),is_probe2=False))
        if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
            target_coords = tuple(get_2target_experiment_target(entrance,experiment,get_2target_target_index(trial_condition,flip_targets=False),is_probe2=True))
            # I don't rotate the probe 2 in this case because the probe2 is treated differently in this experiment set compared to others
            #target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif (experiment == '04Nov2022'):
        if entrance == u'SW':
            target_coords = (38.68590398,12.26402550)
        if entrance == u'SE':
            target_coords = (-12.07033708,38.64499089)
        if entrance == u'NE':
            target_coords = (-38.68641471,-11.94438980)
        if entrance == u'NW':
            target_coords = (11.76033708,-38.94608379)
    else:
        raise ValueError('Unknown experiment date for file %s' % file_header.file_name)
    #return get_arena_to_arena_translate(file_header)(numpy.array(target_coords))
    # we return the arena hole that is closest to the target coordinates dictated above to avoid misplaced targets
    return _get_closest_arena_hole_coord(file_header,numpy.array(target_coords))

def get_arena_reverse_target(file_header):
    entrance = file_header.start_location
    trial_condition = file_header.trial
    experiment = file_header.exper_date

    reverse_target_coords = (numpy.nan,numpy.nan)
    
    if ('23May2019' == experiment) or ('Pilot' in experiment):
        #reverse_target_coords = -6.32, 36.62 #180deg rotated target coords
        reverse_target_coords = 20.47, -39.91 #actual target coordinates
        rotation_angle        = 0.0
        if trial_condition.startswith('R90'):
            rotation_angle = -numpy.pi/2.0
        elif trial_condition.startswith('R270'):
            rotation_angle = numpy.pi/2.0
        elif is_trial_of_type(trial_condition,'flip') or trial_condition.startswith('R180'):
            rotation_angle = numpy.pi
        reverse_target_coords = tuple(_get_closest_arena_hole_coord(file_header,misc.RotateTransf( None, None, get_arena_center(file_header), rotation_angle )(numpy.array(reverse_target_coords))))
    elif ('06Sept2019' == experiment) or ('07Oct2019' == experiment) or ('08Jul2019' == experiment):
        #if trial_condition.isdigit() or ('probe' in trial_condition.lower()):
        reverse_target_coords = _get_main_target_set_for_rotating_probes(entrance) # entrance label is already of the entrance that's rotated relative to main training trials
        # kelly original
        #if trial_condition.startswith('R90'):
        #    reverse_target_coords = _get_main_target_set_for_rotating_probes(_rotate_compass_label_default(entrance, numpy.pi/2.0))
        #rotation_angle = 0.0
        #if trial_condition.startswith('R90'):
        #    rotation_angle = -numpy.pi/2.0
        #if trial_condition.startswith('R180'):
        #    rotation_angle = -numpy.pi
        #if trial_condition.startswith('R270'):
        #    rotation_angle = numpy.pi/2.0
        #reverse_target_coords = _get_main_target_set_for_rotating_probes(_rotate_compass_label_default(entrance,rotation_angle))

    elif experiment == '11Dec2019': # 'dec-2019/BKGDimage-localCues_cropped.png'
        reverse_target_coords = 11.07, -30.48
    elif (experiment == '08Mar2021') or (experiment == '06May2021') or (experiment == '26May2021'):
        reverse_target_coords = -19.61, -16.63
    elif experiment == '16Jul2021': # 'jun-jul-aug-nov-2021/BDGDimage_arenaB_visualCues_cropped.png'
        reverse_target_coords = 27.75405622 , 7.32010752
    elif experiment == '15Nov2021': # jun-jul-aug-nov-2021/BKGDimage-Nov15_cropped.png
        reverse_target_coords = -25.64674698,  -3.03526881
    elif (experiment == '22Jun2021') or (experiment == '19Nov2021'): # 'jun-jul-aug-nov-2021/BKGDimage-arenaB_cropped.png'
        reverse_target_coords = tuple(get_2target_experiment_target(entrance,experiment,1))
    elif (experiment == '12Aug2022'): #normal target A
        # the reverse target is where the main target is supposed to be if the mouse had entered from a 180 rotation
        reverse_target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False)))
        if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
            reverse_target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif (experiment == '11Oct2022'): #normal target A
        # the reverse target is where the main target is supposed to be if the mouse had entered from a 180 rotation
        reverse_target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False)))
        if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
            reverse_target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif experiment == '30Jul2021':
        reverse_target_coords = 6.75, -26.23
    elif experiment == '11Aug2021':
        reverse_target_coords = -29.21, -3.77
    elif (experiment == '20Sept2022'):
        reverse_target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False),is_probe2=False))
        if ('probe' in trial_condition.lower()) and (get_named_trial_trialnumber(trial_condition) >= 2): # Probe2
            reverse_target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False),is_probe2=True))
            # I don't rotate the probe 2 in this case because the probe2 is treated differently in this experiment set compared to others
            #target_coords = tuple(get_2target_experiment_target(_rotate_compass_label_default(entrance, numpy.pi),experiment,get_2target_target_index(trial_condition,flip_targets=False)))
    elif (experiment == '04Nov2022'):
        # I do not need to rotate the entrance label because
        # the entrance label is already "rotated" in the raw experimental files (i.e., the SE R90 is already SW)
        #########if is_trial_of_type(trial_condition,'rotation'):
        #########    entrance = _rotate_compass_label_default(entrance, numpy.pi)
        if entrance == u'SW':
            reverse_target_coords = (11.76033708,-38.94608379)
        if entrance == u'SE':
            reverse_target_coords = (38.68590398,12.26402550)
        if entrance == u'NE':
            reverse_target_coords = (-12.07033708,38.64499089)
        if entrance == u'NW':
            reverse_target_coords = (-38.68641471,-11.94438980)
    else:
        if not ('Pilot' in experiment):
            raise ValueError('Unknown experiment date for file %s' % file_header.file_name)
    #return get_arena_to_arena_translate(file_header)(numpy.array(reverse_target_coords))
    return _get_closest_arena_hole_coord(file_header,numpy.array(reverse_target_coords)) #get_arena_hole_coord(file_header)[numpy.argmin(numpy.linalg.norm(get_arena_hole_coord(file_header) - numpy.array(reverse_target_coords),axis=1))]


def get_intertarget_distance_static_entrance():
    r_tgt = numpy.array([get_arena_target(misc.structtype(exper_date='06Sept2019',start_location='SW',trial='1')),
                         get_arena_target(misc.structtype(exper_date='06Sept2019',start_location='SE',trial='1')),
                         get_arena_target(misc.structtype(exper_date='06Sept2019',start_location='NW',trial='1')),
                         get_arena_target(misc.structtype(exper_date='06Sept2019',start_location='NE',trial='1'))])
    d_intgt = numpy.array([ numpy.linalg.norm(r1-r2) for r1 in r_tgt for r2 in r_tgt ])
    d_intgt = d_intgt[d_intgt>0.0]
    return misc.calc_lower_upper_mean(d_intgt)