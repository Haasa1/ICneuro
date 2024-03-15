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

import copy
import numpy
import pandas
import random
import scipy.io
import scipy.sparse
import scipy.signal
import scipy.spatial
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import modules.io as io
import modules.helper_func_class as misc
import modules.process_mouse_trials_lib as plib
import warnings

def calc_self_intersections(all_trials_rt,trial_labels_rt,all_trials_ft,trial_labels_ft,output_dir='.',output_filename='self_intersection_fixtgt_reltgt',mouse_part='center',save_data=True):
    import time
    import datetime

    start_time = time.monotonic()
    # calculating number of self-intersections of the trajectories
    selfint_st_rt = [ find_self_intersections(all_mice,mouse_part=mouse_part,return_intersec_position=False,verbose=True) for all_mice in all_trials_rt ]
    selfint_st_ft = [ find_self_intersections(all_mice,mouse_part=mouse_part,return_intersec_position=False,verbose=True) for all_mice in all_trials_ft ]
    exper_info_rt = [ [ dict(exper_date=m.exper_date,file_name=m.file_name,mouse_number=m.mouse_number,trial=m.trial,day=m.day,start_location=m.start_location) for m in all_mice ] for all_mice in all_trials_rt ]
    exper_info_ft = [ [ dict(exper_date=m.exper_date,file_name=m.file_name,mouse_number=m.mouse_number,trial=m.trial,day=m.day,start_location=m.start_location) for m in all_mice ] for all_mice in all_trials_ft ]
    end_time = time.monotonic()
    print("* End of calculations... Total time: {}".format(datetime.timedelta(seconds=end_time - start_time)))
    
    if save_data:
        io.save_selfintersection_sim_data(output_dir,output_filename,selfint_st_rt,exper_info_rt,trial_labels_rt,selfint_st_ft,exper_info_ft,trial_labels_ft,mouse_part)
    
    return selfint_st_rt,selfint_st_ft

def find_trajectory(dir_or_loaded_file_list,points,point_radius,find_any_point=False,points_in_temporal_order=True,t0_frac=0.0,dt_frac=1.0):
    """
    finds every trajectory passing close enough to every point in points (considering mouse nose coordinates)

    dir_or_loaded_file_list  -> either a string with the directory name where trajectory files (.mat) are or a list of files returned by load_trial_file
                               if a directory is provided, load all .mat files in it
    points                   -> numpy.ndarray N x 2 (N rows, 2 cols; col 0 -> x, col 1 -> y), 1 point per row
    point_radius             -> tolerance around each point
    find_any_point           -> returns any trajectory that contains at least 1 point of points
    points_in_temporal_order -> if true, then restrict this function to find trajectories that
                                visit each entry in points in the order that they appear
                                in the points list (i.e., points[0] is visited before points[1], etc)
    t0_frac,dt_frac          -> track.time has T elements, so the analysis will be made from T0=floor(t0_frac*T) T0:min(T0+ceil(dt_frac*T),T)

    returns
        tr_found -> list of mouse tracks that match the criteria
        p_found -> points that were found for each tr_found
        kAB -> index of the first and last points found for each tr_found
        tAB -> time interval between first and last points found  for each tr_found
        rAB -> trajectory between first and last points found for each tr_found
        trial -> trial of each tr_found, trial[i] = tr_found[i].trial
    """
    if type(dir_or_loaded_file_list) is str:
        tracks = io.load_trial_file(dir_or_loaded_file_list, '*.mat', load_only_training_sessions_relative_target=False, skip_15_relative_target=False, use_extra_trials_relative_target=True, sort_by_trial=True, fix_nan=True, t0_frac=0.0,dt_frac=1.0)
    else:
        tracks = dir_or_loaded_file_list
    if not(type(tracks) is list):
        tracks = [tracks]
    if not(type(points) is numpy.ndarray):
        points = numpy.asarray(points)
    tr_found = []
    p_found = []
    kAB = []
    tAB = []
    rAB = []
    trial = []
    for tr in tracks:
        k = []
        for i,p in enumerate(points):
            if points_in_temporal_order:
                if i == 0:
                    k_start = 0
                else:
                    k_start = k[-1] if k[-1] > 0 else 0 # start from the last found point, if any
            else:
                k_start = 0
            k.append(find_first_point_in_trajectory(tr, p, point_radius, k_start=k_start))
        k = numpy.asarray(k)
        has_points = (find_any_point and numpy.any(k>=0)) or numpy.all(k>=0)
        if has_points:
            k_found = k[k>=0]
            kmm = numpy.array((numpy.min(k_found),numpy.max(k_found)))
            tr_found.append(tr)
            p_found.append(points[k>=0])
            kAB.append(kmm)
            tAB.append( numpy.abs(numpy.diff(tr.time[kmm]))[0] )
            rAB.append(tr.r_nose[kmm[0]:kmm[1]])
            trial.append(tr.trial)
    return tr_found,p_found,kAB,tAB,rAB,trial
        
def find_first_point(r,p,p_radius,k_start=0,time=None):
    """
    finds the point p = numpy.array((x,y)) in track.r_nose[k_start:]
    with tolerance p_radius
    """
    d = numpy.linalg.norm(r[k_start:] - p,axis=1) # distance from each point in trajectory to p, starting at k_start
    k = misc.find_first(d < p_radius)
    if k == -1: # no element found
        if type(time) is type(None):
            time = numpy.arange(r.shape[0])
        # we try to interpolate around the minimum distance to p, because maybe we missed the point due to low sampling around p (i.e., the mouse was too fast there)
        k_min = d.argmin() + k_start
        k1 = numpy.max((0,k_min-20)) # looking back 20 time steps
        k2 = numpy.min((r.shape[0],k_min+20)) # looking forward 20 time steps
        spline = scipy.interpolate.interp1d(time[k1:k2],r[k1:k2],kind='linear',axis=0,copy=False)
        nTimeSteps = int(float(k2-k1) / ( 0.1*(time[1] - time[0]))) # increasing time precision by a factor of 10
        d_inter = numpy.linalg.norm(spline(numpy.linspace(time[k1],time[k2-1],nTimeSteps)) - p,axis=1) # distance from each point in trajectory to p, starting at k_start
        kk = misc.find_first(d_inter < p_radius)
        k = k_min if kk >= 0 else -1
    else:
        k = k + k_start
    return k if k>=k_start else -1

def find_first_point_in_trajectory(track,p,p_radius,k_start=0,mouse_part='nose'):
    """
    finds the point p = numpy.array((x,y)) in track.r_nose[k_start:]
    with tolerance p_radius
    """
    mouse_part = mouse_part.lower()
    assert mouse_part in ['nose','center','tail'],"mouse_part must be one of 'nose','center','tail'"
    return find_first_point(track['r_'+mouse_part],p,p_radius,k_start=k_start,time=track.time)

def intersect_trajectory_arena_holes(track,hole_radius,r_arena_holes=None,t1_ind=0,t2_ind=None,ignore_entrance_positions=False):
    """
    intersects the trajectory defined by r (x,y coords) with all the arena holes, within a radius hole_radius of each hole
    
    returns 
        tind_inter -> list of indices where intersection happens (one arena hole per list item)
        dt_inter   -> list of the amount of time spent hanging around each hole (one arena hole per list item)
        r_inter    -> list of intersected coordinates of the trajectory (one arena hole per list item)
        r_hole     -> hole coordinate for each intersection (ndarray with 1 hole coord vector per row)
    """
    if not misc.exists(t1_ind):
        t1_ind = 0
    if type(r_arena_holes) is type(None):
        r_arena_holes = plib.get_arena_hole_coord(track)
    if not misc.exists(t2_ind):
        t2_ind = track.time.size
    r_ignore = numpy.array([ v for _,v in plib.get_arena_entrance_coord(track).items() ]) if ignore_entrance_positions else None
    tind_inter,dt_inter,r_inter,r_hole = intersect_trajectory_holes(track.r_nose[t1_ind:t2_ind,:],hole_radius,r_arena_holes,time=track.time[t1_ind:t2_ind],r_ignore=r_ignore)
    if t1_ind > 0:
        tind_inter = list(numpy.array(tind_inter)+t1_ind)
    return tind_inter,dt_inter,r_inter,r_hole

def intersect_trajectory_holes(r,hole_radius,r_holes,time=None,return_none_if_not_found=False,r_ignore=None):
    """
    intersects the trajectory defined by r (x,y coords) with all the holes, within a radius hole_radius of each hole
    
    returns 
        tind_inter -> list of indices where intersection happens (one arena hole per list item)
        dt_inter   -> list of the amount of time spent hanging around each hole (one arena hole per list item)
        r_inter    -> list of intersected coordinates of the trajectory (one arena hole per list item)
        r_hole     -> hole coordinate for each intersection (ndarray with 1 hole coord vector per row)
    """
    assert (type(r) is numpy.ndarray) and (r.ndim == 2),"r must be a 2-dim ndarray, each row being a different position, col1 == x(t), col2 == y(t)"
    assert type(r_holes) is numpy.ndarray,"r must be a 2-dim ndarray, each row being a different position, col1 == x(t), col2 == y(t)"
    has_r_ignore = misc.exists(r_ignore)
    if has_r_ignore:
        assert misc._is_numpy_array(r_ignore),"r_ignore is either None or ndarray, 1 position per row, or a single position"
        if r_ignore.ndim == 1:
            r_ignore = r_ignore[numpy.newaxis,:]
        ind_inter,_,_,_ = intersect_trajectory_holes(r,hole_radius,r_ignore,time=time,return_none_if_not_found=False,r_ignore=None)
        if len(ind_inter) > 0:
            k_ignore         = numpy.fromiter(misc.flatten_list(ind_inter),dtype=int)
            ind              = numpy.ones(r.shape[0],dtype=bool)
            ind[k_ignore]    = False
            r                = r[ind]
    if r_holes.ndim == 1:
        r_holes = r_holes.reshape((1,r_holes.size))
    if type(time) is type(None):
        time = numpy.arange(r.shape[0])
    tind_inter = []
    dt_inter   = []
    r_inter    = []
    k_hole     = numpy.zeros(r_holes.shape[0],dtype=bool)
    for i,r0 in enumerate(r_holes): # for each hole
        tind = numpy.nonzero(numpy.linalg.norm(r - r0,axis=1) < hole_radius)[0]
        if len(tind) > 0:
            tind_inter.append(tind)
            dt_inter.append(time[tind[-1]]-time[tind[0]])
            r_inter.append(r[tind])
            k_hole[i] = True
        else:
            if return_none_if_not_found:
                tind_inter.append(None)
                dt_inter.append(None)
                r_inter.append(None)
    #k_hole = numpy.asarray(k_hole)
    if numpy.any(k_hole):
        r_hole = r_holes[k_hole,:]
    else:
        r_hole = numpy.array([[numpy.nan,numpy.nan]])
    return tind_inter,dt_inter,r_inter,r_hole

def is_in_hole_horizon(r,hole_radius,r_arena_holes=None,return_pos_from='mouse',return_true_or_false=False):
    """
     check if r is close to any hole by a distance less than hole_radius
     if yes, then return hole position
     otherwise, returns false
    
     r_arena_holes -> position of each hole (1 per row); this is contained in the file structure converted by this lib
    
     return_pos_from -> 'mouse' or 'hole'
                        if 'mouse': returns positions of the mouse where the slowing down happened close to hole
                        if 'hole': returns positions of the holes where the slowing down happened
    
     if r is a list of vectors, or a numpy.ndarray
     then, each row of r is a new position to be checked against the holes' positions
     and returns a list containing either False if a given r is not close to any hole, or the requested coordinates (either mouse or hole position)
    """
    if type(r_arena_holes) is type(None):
        r_arena_holes = plib.get_arena_hole_coord()
    if not (type(r) is numpy.ndarray):
        r = numpy.asarray(r)
    if r.ndim > 1:
        return [ is_in_hole_horizon(r0,hole_radius,r_arena_holes=r_arena_holes,return_pos_from=return_pos_from,return_true_or_false=return_true_or_false) for r0 in r ]
    else:
        return_mouse_pos = return_pos_from.lower() == 'mouse'
        is_close_to_hole = numpy.linalg.norm(r - r_arena_holes,axis=1) < hole_radius
        k = numpy.nanargmax(is_close_to_hole)
        if (k == 0) and (not is_close_to_hole[0]):
            return False
        if return_true_or_false:
            return True
        else:
            return r if return_mouse_pos else r_arena_holes[k]

def calc_perp_dist_vs_velocity_corr(track,window_size,absolute_food_vec=False):
    """
    calculates the Pearson correlation between perp dist to food line vs. velocity
    within a rolling time window of size window_size (in seconds)

    absolute_food_vec -> if True, then the food vector is fixed between entrance and target position;
                         otherwise the food vector is relative to mouse center
    """
    if type(track) is list:
        return [ calc_perp_dist_vs_velocity_corr(tr,window_size,absolute_food_vec=absolute_food_vec) for tr in track ]
    else:
        d = calc_mouse_perp_dist_to_food_line(track,return_abs_value=True)
        return calc_rolling_corr(d,track.velocity,window_size,time=track.time)

def calc_deviation_angle_vs_velocity_corr(track,window_size,absolute_food_vec=False):
    """
    calculates the Pearson correlation between target deviation angle vs. velocity
    within a rolling time window of size window_size (in seconds)

    absolute_food_vec -> if True, then the food vector is fixed between entrance and target position;
                         otherwise the food vector is relative to mouse center
    """
    if type(track) is list:
        return [ calc_deviation_angle_vs_velocity_corr(tr,window_size,absolute_food_vec=absolute_food_vec) for tr in track ]
    else:
        d = numpy.arccos(calc_mouse_deviation(track,absolute_food_vec=absolute_food_vec,return_angle=False)) # absolute value of the angle (deviations to the right or left are always positive)
        return calc_rolling_corr(d,track.velocity,window_size,time=track.time)

def calc_rolling_corr(x,y,window_size,time=None,**pandasRollingArgs):
    """
    calculates the rolling (or "sliding" or "moving") Pearson correlation between x and y
    this is similar to the "moving average" filter", in the sense that
    given two functions x(t) and y(t) and a window_size (in units of t),
    then we calculate the r(x,y,t0) = Pearson Corr between x and y defined in [t0-window_size/2; t0+window_size/2],
    for every t0 in time

    x,y -> functions of time that will be used to calculate the Pearson correlation
    window_size -> amount of time (or number of time steps) of each window for the correlation
    time -> time vector (if None, then assumed to be integer time steps time=numpy.arange(x.size) )
            if time is given, then window_size is measured in the units of time dt = time[1] - time[0]
            such that window_size = window_size / dt
    pandasRollingArgs -> extra arguments that may be passed to pandas.DataFrame.rolling used to create the rolling structure for correlation
                         https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
    """
    dt = 1.0
    if not(type(time) is type(None)):
        k = misc.find_first(numpy.logical_and(numpy.logical_not(numpy.isnan(time[:-1]+time[1:])), numpy.logical_not(numpy.isnan(time[:-1])) )   )
        dt = time[k+1] - time[k]
    window_size = int( numpy.ceil(window_size / dt) )
    pandasRollingArgs = misc.set_default_kwargs(pandasRollingArgs,center=True,closed='left')
    return pandas.Series(x).rolling(window_size,**pandasRollingArgs).corr(pandas.Series(y)).to_numpy()

def find_velocity_minima(track,min_velocity_below=None,min_velocity_prominence=None,t1_ind=None,t2_ind=None,**find_peaks_args):
    return find_minima(track.velocity,prominence=min_velocity_prominence,minima_below=min_velocity_below,t1_ind=t1_ind,t2_ind=t2_ind,**find_peaks_args)

def find_minima(x,prominence=None,minima_below=None,t1_ind=None,t2_ind=None,**find_peaks_args):
    """
    x is inverted by taking y=max(x) - x;
    we then find the peaks of y (equivalently finding the minima of x)

    prominence   -> minimum vertical distance to the closest local maximum between minima
    minima_below -> hard threshold below which the minima must be located
    
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    for more parameters
    """
    t1_ind = t1_ind if misc.exists(t1_ind) else 0
    t2_ind = t2_ind if misc.exists(t2_ind) else x.size
    xx = x[t1_ind:t2_ind]
    x_max = misc.nanmax(xx)
    minima_below = minima_below if misc.exists(minima_below) else x_max
    k_min,_ = scipy.signal.find_peaks(x_max-xx,**misc._get_kwargs(find_peaks_args,prominence=prominence,height=x_max-minima_below)) # default
    k_min = k_min + t1_ind
    return k_min,xx[k_min]

def calc_threshold(x,gamma,threshold_method='ampv'):
    """
     threshold_method -> 'meanv': the threshold is given by a percent of the mean x_th = gamma*mean(x)
                         'ampv' : the threshold is given by a percent of the amplitude x_th = min(x) + gamma * (max(x) - min(x))
                         'abs'  : then, x_th = gamma (absolute threshold)
    """
    threshold_method = threshold_method.lower()
    assert (threshold_method in ['meanv','ampv','abs']),"threshold_method must be one of 'meanv','ampv', or 'abs'"
    if threshold_method == 'ampv':
        min_x = misc.nanmin(x)
        x_th = min_x + gamma * (misc.nanmax(x) - min_x)
    elif threshold_method == 'meanv':
        x_th = gamma * misc.nanmean(x)
    elif threshold_method == 'abs':
        x_th = gamma
    else:
        raise ValueError('unknown threshold method... use ampv, meanv, or abs')
    return x_th

def calc_threshold_crossings(x,x_th,only_downward_crossing=False,return_first_crossing_type=False,t1_ind=0,t2_ind=None):
    if not misc.exists(t1_ind):
        t1_ind = 0
    if not misc.exists(t2_ind):
        t2_ind = len(x)
    result = _calc_threshold_crossings_internal(x[t1_ind:t2_ind],x_th,only_downward_crossing=only_downward_crossing,return_first_crossing_type=return_first_crossing_type)
    if return_first_crossing_type:
        tind_crossing,tt = result
        return numpy.asarray(tind_crossing)+t1_ind,tt
    else:
        tind_crossing    = result
        return numpy.asarray(tind_crossing)+t1_ind

def _calc_threshold_crossings_internal(x,x_th,only_downward_crossing=False,return_first_crossing_type=False):
    """
     returns all the indices of x when x crosses x_th
     if only_downward_crossing == True, then returns only the crossings in which x is decreasing
    """
    f = (x[:-1]-x_th)*(x[1:]-x_th) # f<=0 -> crossing of threshold;
    if only_downward_crossing:
        tind_crossing = numpy.nonzero(numpy.logical_and( f <= 0 , x[1:]<=x_th))[0] # index of the slowing down instant (down crossing), since f<=0 and v[n] < v_th < v[n-1]
    else:
        #t_cross,_ = find_inter_func(track.time,track.velocity,v_th) # finds all crossings of velocity and v_th using Newton's method
        tind_crossing = numpy.nonzero(f <= 0)[0] # index of all crossings
    if return_first_crossing_type:
        k = misc.find_first(numpy.logical_and( f <= 0 , x[1:]<=x_th))
        if tind_crossing.size == 0:
            tt = 'none'
        else:
            tt = 'going_up'
            if k == tind_crossing[0]: # k is the index of the first going down
                tt = 'going_down'
        return tind_crossing,tt
    else:
        return tind_crossing

def calc_position_crossings(track,coord='x',threshold_method='ampv',gamma=0.2,mouse_part='nose'):
    """
     track -> data structure returned by the load_trial_file function
     coord -> 'x' or 'y', calculates the threshold crossings of either 'x' or 'y' coord of the respective mouse_part
     mouse_part -> 'nose', 'center', 'tail'
     threshold_method -> 'meanv': the threshold is given by a percent of the mean x_th = gamma*mean(x)
                         'ampv': the threshold is given by a percent of the amplitude x_th = min(x) + gamma * (max(x) - min(x))
                         'abs': then, x_th = gamma (absolute threshold)
    
     returns
       tind_crossing -> index of the time and position variables where the crossing happens
       t_cross -> instants when v crosses the threshold
       IEI     -> interevent interval; time interval between every consecutive crossing; IEI[n] = t_cross[n] - t_cross[n-1]
       x_th -> the threshold calculated by this method
    """
    mouse_part = mouse_part.lower()
    coord = coord.lower()
    if not ( mouse_part in ['nose','center','tail'] ):
        raise ValueError('mouse_part must be nose, tail or center')
    if not ( coord in ['x', 'y'] ):
        raise ValueError('coord must be x or y')
    k = 0 if coord == 'x' else 1
    pos_label = 'r_' + mouse_part
    #t_nan = numpy.nonzero(numpy.isnan(numpy.prod(track[pos_label],axis=1)))[0] # first check for nan in the position and fix it
    #if t_nan.size > 0:
    if misc.contains_nan(track[pos_label][:,k]):
        track[pos_label] = interp_trajectory(track.time,track[pos_label])
    x_th = calc_threshold(track[pos_label][:,k],gamma,threshold_method)
    tind_crossing = calc_threshold_crossings(track[pos_label][:,k],x_th,only_downward_crossing=False)
    t_cross = (track.time[tind_crossing] + track.time[tind_crossing+1])/2.0
    IEI = t_cross[1:] - t_cross[:-1]
    return tind_crossing,t_cross,IEI,x_th

def find_self_intersections(track,mouse_part='nose',return_intersec_position=False, verbose=False): #,interpolate=False,n_points_interp=100
    """
    determines the self-initersections of the trajectory in track determined by mouse part

    we basically try to find the zeros (within precision) of the difference between the trajectory and itself

    returns
        * k -> indices of all self-intersections
        * r[k,:] -> positions of all self-intersections
    """
    mouse_part = mouse_part.lower()
    assert (mouse_part in ['nose','center','tail']),"mouse_part must be one of the following: 'nose','center','tail'"
    if type(track) is list:
        k  = misc.get_empty_list(len(track))
        r0 = misc.get_empty_list(len(track))
        for i,tr in enumerate(track):
            k[i],r0[i] = find_self_intersections(tr,mouse_part=mouse_part,return_intersec_position=True, verbose=verbose) # interpolate=interpolate,n_points_interp=n_points_interp,
    else:
        if verbose:
            print(' ... processing experiment: %s      mouse %s     trial %s'%(track.exper_date,track.mouse_number,track.trial))
        lab = 'r_'+mouse_part
        k,r0 = misc.find_self_intersection_jit(track[lab],track.time)#,interpolate=interpolate,n_points=n_points_interp)
    if return_intersec_position:
        return k,r0
    else:
        return k

def calc_unitary_mouse_vec(track):
    mouse_vec = track.r_nose - track.r_center
    return mouse_vec / numpy.linalg.norm(mouse_vec,axis=1)

def calc_mean_velocity(track):
    if type(track) is list:
        return [calc_mean_velocity(tr) for tr in track]
    else:
        return misc.nanmean(track.velocity)

def calc_velocity_vector(track):
    return track.velocity*calc_unitary_mouse_vec(track)

def calc_acceleration(track):
    a = numpy.diff(track.velocity)/numpy.diff(track.time)
    return numpy.append(a,a[-1]) # we repeat the last value to make a the same size as time

def calc_acceleration_vector(track):
    v = calc_velocity_vector(track)
    a = numpy.diff(v,axis=0)/(numpy.diff(track.time)[:,numpy.newaxis])
    return numpy.append(a,a[-1][numpy.newaxis,:]) # we repeat the last value to make a the same size as time

def calc_position_integral(track):
    """
    performs the line integral (with Simpson method) of the vector field r[t,:],
    where r[t,:] is the mouse (x,y) nose position at time t

    scipy_simpson_args:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html


    https://en.wikipedia.org/wiki/Line_integral#Definition_2
    """
    if type(track) is list:
        return [calc_position_integral(tr) for tr in track]
    else:
        drdt = calc_velocity_vector(track)
        return scipy.integrate.simpson(track.r_center[:,0]*drdt[:,0]+track.r_center[:,1]*drdt[:,1],x=track.time,even='last')

def calc_partial_traveled_dist(track,mouse_part='center',use_integral=False):
    """
    returns d(t) -> total distance traveled up until time t
    """
    mouse_part = mouse_part.lower()
    assert mouse_part in ['center','nose','tail'],"mouse_part must be one of 'center','nose','tail'"
    if type(track) is list:
        return [ calc_partial_traveled_dist(tr,mouse_part=mouse_part,use_integral=use_integral) for tr in track ]
    else:
        if use_integral:
            get_distance = lambda tr,t1=None,t2=None: calc_traveled_dist(tr,t1_ind=t1,t2_ind=t2,use_integral=use_integral,mouse_part=mouse_part)
            d_cumul = numpy.zeros(track.time.size)
            D = 0.0
            for t_ind in range(1,track.time.size):
                D             += get_distance(track,t_ind-1,t_ind+1)
                d_cumul[t_ind] = D
        else:
            d_cumul = calc_traveled_dist_pyth(track['r_'+mouse_part],cumulative=True)
        return d_cumul

def calc_traveled_dist_time_interval(track,dt=3.0,t0=None,t1=None,use_integral=False,mouse_part='center'):
    """
    calls "calc_traveled_dist" to calculate traveled distance in a given time interval
     
    if t0 and t1 are given    -> distance between times [t0,t1]
    if t0 is given (no t1)    -> distance between times [t0,t0+dt]
    if t1 is given (no t0)    -> distance between times [t1-dt,t1]
    if dt is given (no t0,t1) -> positive dt: distance between times [end-dt,end]
                                 negative dt: distance between times [0,end-abs(dt)]
    """
    mouse_part = mouse_part.lower()
    assert mouse_part in ['center','nose','tail'],"mouse_part must be one of 'center','nose','tail'"
    if type(track) is list:
        return [ calc_traveled_dist_time_interval(tr,dt=dt,t0=t0,t1=t1,use_integral=use_integral,mouse_part=mouse_part) for tr in track ]
    else:
        if not ( misc.exists(t0) or misc.exists(t1) ):
            # neither t0 nor t1 are given
            if not misc.exists(dt):
                raise ValueError('at least one of t1,t0,dt must be given')
            if dt > 0:
                t1 = track.time[-1]
                t0 = t1 - dt
            else:
                t0 = track.time[0]
                t1 = track.time[-1] - numpy.abs(dt)
        else:
            # either t0 or t1 (or both) are given
            if not misc.exists(t0):
                t0 = t1 - dt
            if not misc.exists(t1):
                t1 = t0 + dt
        t0,t1 = sorted((t0,t1))
        k     = numpy.nonzero(numpy.logical_and(track.time>=t0,track.time<=t1))[0]
        return calc_velocity_abs_integral(track.velocity[k],track.time[k]) if use_integral else calc_traveled_dist_pyth(track['r_'+mouse_part][k,:])


def calc_traveled_dist(track,r0=None,Tmax=None,t1_ind=None,t2_ind=None,use_integral=False,mouse_part='center'):
    """
    performs the line integral (with Simpson method) of the vector field r[t,:],
    where r[t,:] is the mouse (x,y) nose position at time t

    r0            -> initial condition, must be a point in the trajectory of track
    Tmax          -> time instant of the right-hand integral limit: r(Tmax)
    t1_ind,t2_ind -> time indices for r0 and r[Tmax]; if given, take precedence relative to r0 and Tmax (i.e., r0 and Tmax are ignored)
                     only calculates distance from r[t1_ind,:] to r[t2_ind,:]

    scipy_simpson_args:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html

    https://en.wikipedia.org/wiki/Arc_length#Formula_for_a_smooth_curve
    """
    mouse_part = mouse_part.lower()
    assert mouse_part in ['center','nose','tail'],"mouse_part must be one of 'center','nose','tail'"
    if type(track) is list:
        return [ calc_traveled_dist(tr,r0=r0,Tmax=Tmax,t1_ind=t1_ind,t2_ind=t2_ind,use_integral=use_integral,mouse_part=mouse_part) for tr in track ]
    else:
        if misc.exists(t1_ind):
            k0 = t1_ind
        else:
            k0 = 0
            if misc.exists(r0):
                k0 = find_first_point_in_trajectory(track,r0,1e-8,k_start=0,mouse_part='center')
                if k0 == -1:
                    k0 = find_first_point_in_trajectory(track,r0,1e-8,k_start=0,mouse_part='nose')
                if k0 == -1:
                    k0 = find_first_point_in_trajectory(track,r0,1e-8,k_start=0,mouse_part='tail')
                if k0 == -1:
                    raise ValueError('initial condition r0 not found in track: trial %s; mouse %s'%(track.trial,track.mouse_number))
        if misc.exists(t2_ind):
            k1 = t2_ind
        else:
            k1 = track.time.size 
            if misc.exists(Tmax):
                k1 = misc.find_first(numpy.abs(track.time-Tmax)<1e-8)
                if k1 == -1:
                    raise ValueError('Tmax not found for track: trial %s; mouse %s'%(track.trial,track.mouse_number))
                k1 = _get_second_index((k1,),track.time.size)
        return calc_velocity_abs_integral(track.velocity[k0:k1],track.time[k0:k1]) if use_integral else calc_traveled_dist_pyth(track['r_'+mouse_part][k0:k1,:])

def calc_traveled_dist_pyth(r,cumulative=False):
    """
    r[t,:] = x[t],y[t]

    sums the hypotenuse of the displacements between every time point
    """
    if cumulative:
        return numpy.cumsum(numpy.linalg.norm(numpy.diff(numpy.insert(r,0,r[0,:],axis=0),axis=0),axis=1))
    else:
        return numpy.sum(numpy.linalg.norm(numpy.diff(numpy.insert(r,0,r[0,:],axis=0),axis=0),axis=1))

def calc_velocity_abs_integral(v,t=None):
    if not misc._is_numpy_array(v):
        v = numpy.asarray(v)
    if not misc.exists(t):
        t = numpy.arange(v.size)
    return scipy.integrate.simpson(numpy.abs(v),x=t,even='last')

def calc_number_checked_holes_per_dist(track,hole_horizon,threshold_method='ampv',gamma=0.2,divide_by_total_distance=True,cumulative=False,ignore_entrance_positions=False,use_velocity_minima=False,velocity_min_prominence=None,velmin_find_peaks_args=None):
    """
    if cumulative, then returns the number of checked holes up to time t, n(t)
    """
    if type(track) is list:
        return [ calc_number_checked_holes_per_dist(tr,hole_horizon,threshold_method          = threshold_method          ,
                                                                    gamma                     = gamma                     ,
                                                                    divide_by_total_distance  = divide_by_total_distance  ,
                                                                    cumulative                = cumulative                ,
                                                                    ignore_entrance_positions = ignore_entrance_positions ,
                                                                    use_velocity_minima       = use_velocity_minima       ,
                                                                    velocity_min_prominence   = velocity_min_prominence   ,
                                                                    velmin_find_peaks_args    = velmin_find_peaks_args    ) for tr in track ]
    else:
        t_holes = find_slowing_down_close_to_hole(track,hole_horizon,threshold_method          = threshold_method          ,
                                                                     gamma                     = gamma                     ,
                                                                     ignore_entrance_positions = ignore_entrance_positions ,
                                                                     use_velocity_minima       = use_velocity_minima       ,
                                                                     velocity_min_prominence   = velocity_min_prominence   ,
                                                                     velmin_find_peaks_args    = velmin_find_peaks_args    )[0]
        if cumulative:
            times = numpy.zeros(track.time.size,dtype=int)
            times[t_holes] = 1
            D = calc_partial_traveled_dist(track) if divide_by_total_distance else 1.0
            return numpy.cumsum(times) / D
        else:
            return float(t_holes.size)/calc_traveled_dist(track)

def calc_number_of_checks_entropy(track,return_error_estimate=False,**n_checks_histogram_args):
    """
    calls calc_number_of_checks_histogram_tracks to generate a normalized histogram
    then calculates the -sum(P*log(P)) on the generated histogram

    entropy error estimated according to the formulas derived in
    MS Roulston (1999): Estimating the errors on measured entropy and mutual information. Physica D 125: 285-294. https://doi.org/10.1016/S0167-2789(98)00269-3.
    (eqs 39 and 40)

    returns
        - entropy
        - entropy stddev (if return_error_estimate == True)
        - entropy max bias (if return_error_estimate == True; true entropy = entropy + max bias; if my interpretation of Roulston's is correct)
    """
    n_checks_histogram_args              = misc._get_kwargs(n_checks_histogram_args,normalize=True)
    n_checks_histogram_args['normalize'] = True
    dist             = calc_number_of_checks_histogram_tracks(track,**n_checks_histogram_args)
    log_c            = lambda xx: numpy.array([numpy.log(x) if x>0.0 else 0.0 for x in xx]) # corrected log to avoid inf
    calc_entropy_obs = lambda P: -numpy.sum(P * numpy.log(P)) # H_obs in Roulston paper
    B_star           = lambda P: numpy.nonzero(P>0)[0].size   # number of nonzero probability states (B* in Roulston paper)
    stddev_entropy   = lambda N,P,S: numpy.sqrt((1.0/N) * numpy.sum( ((log_c(P) + S)**2)*P*(1-P) ) ) # eq (40) in Roulston stddev of entropy sigma_H
    if type(dist) is list:
        N         = [ len(tr) for tr in track ] # track is a list of lists -- N is the number of mice in each trial
        S         = numpy.array([ calc_entropy_obs(d.P[d.P>0]) for d in dist ])
        S_std     = numpy.array([ stddev_entropy(n,d.P,s) for n,d,s in zip(N,dist,S) ])
        S_maxbias = numpy.array([(B_star(d.P)-1)/(2.0*n) for d,n in zip(dist,N)]) + S_std
    else:
        N         = len(track)
        S         = calc_entropy_obs(dist.P[dist.P>0])
        S_std     = stddev_entropy(N,dist.P,S)
        S_maxbias = ((B_star(dist.P)-1)/(2.0*N)) + S_std
    result = S
    if return_error_estimate:
        result = (S,S_std,S_maxbias)
    return result


def calc_number_of_checks_histogram_tracks(track,d_checks_threshold=20.0,less_than_threshold=True,normalize=True,hole_horizon=3.0,**args_for_calc_dist_func):
    """
    number of checks that happened within a radius of d_checks_threshold of the reference point (usually the target, or the TEV-target)
    that led to the target

    if track is a list of trackfile:
        track[i] -> data for mouse i; all mice within the same trial
    if track is a list of lists:
        track[j][i] -> mouse i in trial j
    this function will try and sort track list of lists to have the above structure

    The d_checks (distance of the checks towards the target) are calculated by calc_dist_checked_holes_target
    d_checks           -> list of numpy array; only used if track is None
                          d_checks[i] -> array containing all distances of hole checks from mouse i
                                         the distance is calculated outside this function, and is relative to a reference point
                                         (usually the target, or the TEV-target)
                              i.e., d_checks[i][j] == 0.0 means that the hole check j of mouse i happened exactly at the reference point (e.g., target)

    main input is either track or d_checks
    track              -> (list of) input track files
    hole_horizon       -> (cm) distance around each hole within which a hole check is captured (refer to calc_dist_checked_holes_target)
    d_checks_threshold -> (cm) distance threshold starting from the reference point where the data will be cutoff

    all other arguments are forwarded to the function calc_dist_checked_holes_target to calculate d_checks

    returns structtype s with fields s.n and s.P as a misc.structtype
            s.n -> number of hole checks within d_checks_threshold of the reference (target)
            s.P -> number of times the food was found with n checks within a radius of d_checks_threshold from the target
        if track is a list of lists, then returns a list of structtypes (one for each item in track list)
    """
    if misc.is_list_of_list(track):
        track = io.group_track_list(track, group_by='trial',return_group_keys=False) # track[j][i] -> mouse i of trial j
        return [ calc_number_of_checks_histogram_tracks(tr,d_checks_threshold=d_checks_threshold,less_than_threshold=less_than_threshold,normalize=normalize,hole_horizon=hole_horizon,**args_for_calc_dist_func) for tr in track ]

    d_checks = calc_dist_checked_holes_target(track,hole_horizon,**misc._get_kwargs(args_for_calc_dist_func,include_target=True,remove_target_check_duplicates=True,force_input_target=True))
    #if estimate_error:
    #    if (type(track) is list):
    #        n_samples = len(track)
    #        if n_samples > 10: # uses bootstrap
    #            pass
    #        else: # uses jackknife
    #            pass
    #    else:
    #        warnings.warn('calc_number_of_checks_histogram_tracks :: estimate_error :: cannot estimate error because n_samples = 1')
    return calc_number_of_checks_histogram(d_checks,d_checks_threshold=d_checks_threshold,less_than_threshold=less_than_threshold,normalize=normalize)

def calc_number_of_checks_histogram(d_checks,d_checks_threshold=20.0,less_than_threshold=True,normalize=True):
    """
    number of checks that happened within a radius of d_checks_threshold of the reference point (usually the target, or the TEV-target)
    that led to the target

    d_checks           -> list of numpy array; only used if track is None
                          d_checks[i] -> array containing all distances of hole checks from mouse i
                                         the distance is calculated outside this function, and is relative to a reference point
                                         (usually the target, or the TEV-target)
                              i.e., d_checks[i][j] == 0.0 means that the hole check j of mouse i happened exactly at the reference point (e.g., target)
    d_checks_threshold -> (cm) distance threshold starting from the reference point where the data will be cutoff


    returns structtype s with fields s.n and s.P as a misc.structtype
        s.n -> number of hole checks within d_checks_threshold of the reference (target)
        s.P -> number of times the food was found with n checks within a radius of d_checks_threshold from the target
    """
    if not(type(d_checks) is list):
        d_checks = [d_checks]
    assert type(d_checks[0]) is numpy.ndarray, "d_checks must be list of numpy.ndarray"
    if less_than_threshold:
        check_condition = lambda A,B: A<=B
    else:
        check_condition = lambda A,B: A>B
    n_checks = [ numpy.nonzero(check_condition(d,d_checks_threshold))[0].size for d in d_checks ] # number of checks within distance threshold for each mouse
    n_max    = misc.nanmax(n_checks)
    n_edges  = numpy.arange(n_max+1)+1
    S,_      = numpy.histogram(n_checks,bins=n_edges,density=normalize)#misc.calc_distribution(n_checks,x_edges=n_edges,return_as_struct=True,binning='linear',replace_Peq0_by_nan=True,remove_Peq0=False)
    return misc.structtype(n=n_edges[:-1],P=S)


def calc_time_checked_holes_before_target(track,hole_horizon,include_target=False,remove_target_check_duplicates=False,threshold_method='ampv',gamma=0.2,
                                                use_alt_target=False,use_reverse_target=False,
                                                r_target=None,ignore_entrance_positions=False,
                                                use_velocity_minima=False,velocity_min_prominence=None,velmin_find_peaks_args=None,force_input_target=False):
    """
    returns the distance from every checked hole to the target of each track
    """
    if type(track) is list:
        return [ calc_time_checked_holes_before_target(tr,hole_horizon,
                                                    include_target                 = include_target                 ,
                                                    remove_target_check_duplicates = remove_target_check_duplicates ,
                                                    threshold_method               = threshold_method               ,
                                                    gamma                          = gamma                          ,
                                                    use_reverse_target             = use_reverse_target             ,
                                                    r_target                       = r_target                       ,
                                                    ignore_entrance_positions      = ignore_entrance_positions      ,
                                                    use_velocity_minima            = use_velocity_minima            ,
                                                    velocity_min_prominence        = velocity_min_prominence        ,
                                                    velmin_find_peaks_args         = velmin_find_peaks_args         ,
                                                    force_input_target             = force_input_target             ) for tr in track ]
    else:
        _,t,_,_ = find_slowing_down_close_to_hole(track,hole_horizon,threshold_method          = threshold_method          ,
                                                                     gamma                     = gamma                     ,
                                                                     return_pos_from           = 'hole'                    ,
                                                                     ignore_entrance_positions = ignore_entrance_positions ,
                                                                     use_velocity_minima       = use_velocity_minima       ,
                                                                     velocity_min_prominence   = velocity_min_prominence   ,
                                                                     velmin_find_peaks_args    = velmin_find_peaks_args    )
        if (t.size > 0):
            r_target = r_target if misc.exists(r_target) else track.r_target
            if not force_input_target:
                if use_alt_target:
                    r_target = track.r_target_alt
                elif use_reverse_target:
                    r_target = track.r_target_reverse 
            t_to_food = calc_time_to_food(track,r_target=r_target)
            t         = t_to_food - t
            if not(0.0 in t):
                t = numpy.append(t,0.0)
            return t
        else:
            return numpy.zeros(1)


def calc_dist_checked_holes_target(track,hole_horizon,include_target=False,remove_target_check_duplicates=False,threshold_method='ampv',gamma=0.2,return_pos_from='hole',
                                         use_alt_target=False,use_reverse_target=False,use_closest_target=False,
                                         r_target=None,ignore_entrance_positions=False,
                                         use_velocity_minima=False,velocity_min_prominence=None,velmin_find_peaks_args=None,force_input_target=False):
    """
    returns the distance from every checked hole to the target of each track
    """
    if type(track) is list:
        return [ calc_dist_checked_holes_target(tr,hole_horizon,include_target=include_target,remove_target_check_duplicates=remove_target_check_duplicates,
                                                   threshold_method=threshold_method,gamma=gamma,
                                                   return_pos_from=return_pos_from,use_reverse_target=use_reverse_target,
                                                   use_closest_target=use_closest_target,r_target=r_target,
                                                   ignore_entrance_positions = ignore_entrance_positions,
                                                   use_velocity_minima       = use_velocity_minima,
                                                   velocity_min_prominence   = velocity_min_prominence,
                                                   velmin_find_peaks_args    = velmin_find_peaks_args,
                                                   force_input_target        = force_input_target) for tr in track ]
    else:
        _,_,r,_ = find_slowing_down_close_to_hole(track,hole_horizon,threshold_method          = threshold_method          ,
                                                                     gamma                     = gamma                     ,
                                                                     return_pos_from           = return_pos_from           ,
                                                                     ignore_entrance_positions = ignore_entrance_positions ,
                                                                     use_velocity_minima       = use_velocity_minima       ,
                                                                     velocity_min_prominence   = velocity_min_prominence   ,
                                                                     velmin_find_peaks_args    = velmin_find_peaks_args    )
        if (r.shape[0] > 0):
            r_target = r_target if misc.exists(r_target) else track.r_target
            if not force_input_target:
                if use_alt_target:
                    r_target = track.r_target_alt
                elif use_reverse_target:
                    r_target = track.r_target_reverse 
            if use_closest_target:
                warnings.warn('calc_dist_checked_holes_target ::: r_target parameter is ignored; using the closest target from track', RuntimeWarning)
                d1 = numpy.linalg.norm(r - track.r_target    ,axis=1)
                d2 = numpy.linalg.norm(r - track.r_target_alt,axis=1)
                d  = numpy.min(numpy.row_stack((d1,d2))    ,axis=0).flatten()
            else:
                d  = numpy.linalg.norm(r - r_target,axis=1)
            d = _include_target_in_number_of_checks(d,include_target,remove_target_check_duplicates)
            return d
        else:
            return numpy.zeros(1)

def calc_number_checked_holes(track,hole_horizon,threshold_method='ampv',gamma=0.2,ignore_entrance_positions=False,
                                    use_velocity_minima=False,velocity_min_prominence=None,velmin_find_peaks_args=None):
    return_scalar = False
    if not(type(track) is list):
        track = [track]
        return_scalar = True
    res = [ float(find_slowing_down_close_to_hole(tr,hole_horizon,threshold_method          = threshold_method          ,
                                                                  gamma                     = gamma                     ,
                                                                  ignore_entrance_positions = ignore_entrance_positions ,
                                                                  use_velocity_minima       = use_velocity_minima       ,
                                                                  velocity_min_prominence   = velocity_min_prominence   ,
                                                                  velmin_find_peaks_args    = velmin_find_peaks_args    )[0].size) for tr in track ]
    if return_scalar:
        return res[0]
    else:
        return res

def _is_valid_normalize_by(normalize_by):
    if (type(normalize_by) is str):
        normalize_by = normalize_by.lower()
        res = normalize_by in ['max','sum','none']
    else:
        res = False
    return res

def calc_number_of_checkings_near_position(track,r0,r0_horizon,hole_horizon,include_r0=True,remove_r0_duplicates=True,threshold_method='ampv',gamma=0.2,
                                           ignore_entrance_positions=False,use_velocity_minima=False,velocity_min_prominence=None,velmin_find_peaks_args=None,
                                           randomize_check_positions=False):
    """
    finds all the hole-checking happening inside the circle centered in r0 with radius r0_horizon

    randomize_check_positions -> if True, then before counting, the checks are uniformly spread across every arena hole

    returns:
        n_checks -> number of checks detected for each track
                    (list if track is list; otherwise scalar) 
    """
    if type(track) is list:
        n_checks = misc.get_empty_list(len(track)) # index where a slowing down happened inside any hole event horizon
        n_checks = [ calc_number_of_checkings_near_position(tr,r0,r0_horizon,hole_horizon,include_r0                 = include_r0                ,
                                                                                          remove_r0_duplicates       = remove_r0_duplicates      ,
                                                                                          threshold_method           = threshold_method          ,
                                                                                          gamma                      = gamma                     ,
                                                                                          ignore_entrance_positions  = ignore_entrance_positions ,
                                                                                          use_velocity_minima        = use_velocity_minima       ,
                                                                                          velocity_min_prominence    = velocity_min_prominence   ,
                                                                                          velmin_find_peaks_args     = velmin_find_peaks_args    ,
                                                                                          randomize_check_positions  = randomize_check_positions ) for tr in track ]
    else:
        # first find all the positions where a hole-checking event happened
        _,_,r_checks,_ = find_slowing_down_close_to_hole(track,hole_horizon,threshold_method          = threshold_method          ,
                                                                            gamma                     = gamma                     ,
                                                                            return_pos_from           = 'hole'                    ,
                                                                            ignore_entrance_positions = ignore_entrance_positions ,
                                                                            use_velocity_minima       = use_velocity_minima       ,
                                                                            velocity_min_prominence   = velocity_min_prominence   ,
                                                                            velmin_find_peaks_args    = velmin_find_peaks_args    )
        r0 = numpy.array(r0).flatten()
        n_checks = 0
        if r_checks.size > 0:
            if randomize_check_positions:
                # selecting random holes for each detected hole check
                r_checks = track.r_arena_holes[numpy.random.randint(0,high=track.r_arena_holes.shape[0],size=r_checks.shape[0]),:]
            d        = _include_target_in_number_of_checks(numpy.linalg.norm(r_checks - r0,axis=1),include_r0,remove_r0_duplicates)
            n_checks = numpy.nonzero(d <= r0_horizon)[0].size
    return n_checks

def _include_target_in_number_of_checks(d,include_r0,remove_r0_duplicates):
    r0_checks   = numpy.nonzero(d==0.0)[0]
    has_r0      = r0_checks.size>0
    has_r0_dupl = r0_checks.size>1
    if has_r0_dupl:
        if remove_r0_duplicates:
            d = numpy.delete(d,r0_checks[1:]) # this leaves only one check in r0
    else:
        if include_r0 and (not has_r0):
            d = numpy.append(d,0.0)
    return d


def calc_number_of_checkings_per_hole_from_pos(r,normalize_by='max',
                                            grouping_hole_horizon=None,sort_result=True):
    """
    r                  -> numpy.ndarray with (x,y) position in each row
    r0                 -> (x,y) position of interest around which this function counts checkings per hole
    radius_of_interest -> radius around p0
    2d dispersion:
    https://en.wikipedia.org/wiki/Covariance_matrix#/media/File:GaussianScatterPCA.svg

    returns:
        r_un          -> r_un[k,:] == (x,y) coordinates of the hole
        r_count       -> r_count[k] number of times the hole k appeared (may be normalized by max r_count, by sum of r_count, or just the count itself)
        r_cov         -> covariance between x and y coordinates
        r_dispersion  -> sqrt of eigenvalues of the covariance between x and y coordinates (2d-analogous of stddev, each in one of the eigendirections of the covariance matrix)
        r_eigdir      -> the two eigenvectors indicating the eigendirections of the covariance matrix (as a list, one vector per item)
                         r_dispersion[m] corresponds to the dispersion in the direction r_eigdir[m]
    """
   
    # calculate the distribution
    r_un,r_count = _find_unique_positions(r,grouping_hole_horizon)
    r_un         = numpy.array(r_un)
    r_count      = numpy.array(r_count,dtype=float)
    
    r_mean, r_cov, r_dispersion, r_eigdir = misc.calc_dispersion(r_un,r_count)
    if (r_count.size == 1):
        r_count  = r_count[0]*numpy.ones(5)
        r_un     = numpy.tile(r_un[0,:], (5,1))
    
    if _is_valid_normalize_by(normalize_by) and (normalize_by != 'none'):
        r_count  = r_count / (numpy.max(r_count) if (normalize_by == 'max') else numpy.sum(r_count))
    if sort_result:
        ind = numpy.argsort(r_count)
        r_un,r_count = r_un[ind],r_count[ind]
    return r_un,r_count,r_mean,r_cov,r_dispersion,r_eigdir

def calc_number_of_checkings_per_hole(track,hole_horizon,threshold_method='ampv',gamma=0.2,normalize_by='max',
                                            grouping_hole_horizon=None,sort_result=True,ignore_entrance_positions=False,
                                            use_velocity_minima=False,velocity_min_prominence=None,velmin_find_peaks_args=None,
                                            r0=None,radius_of_interest=None):
    """
    r0                 -> (x,y) position of interest around which this function counts checkings per hole
    radius_of_interest -> radius around p0
    2d dispersion:
    https://en.wikipedia.org/wiki/Covariance_matrix#/media/File:GaussianScatterPCA.svg

    returns:
        r_un          -> r_un[k,:] == (x,y) coordinates of the hole
        r_count       -> r_count[k] number of times the hole k appeared (may be normalized by max r_count, by sum of r_count, or just the count itself)
        r_cov         -> covariance between x and y coordinates
        r_dispersion  -> sqrt of eigenvalues of the covariance between x and y coordinates (2d-analogous of stddev, each in one of the eigendirections of the covariance matrix)
        r_eigdir      -> the two eigenvectors indicating the eigendirections of the covariance matrix (as a list, one vector per item)
                         r_dispersion[m] corresponds to the dispersion in the direction r_eigdir[m]
    """
    if not _is_valid_normalize_by(normalize_by):
        raise ValueError("calc_number_of_checkings_per_hole ::: normalize_by must be one of 'max','sum','none'")
    if type(track) is list:
        track = list(misc.flatten_list(track,only_lists=True))
    if not misc.exists(grouping_hole_horizon):
        grouping_hole_horizon = hole_horizon

    if misc.exists(r0):
        r0 = numpy.asarray(r0)

    if misc.exists(r0) and misc.exists(radius_of_interest):
        track = remove_outside_of_radius(track,r0,radius_of_interest,mouse_part='nose',copy_track=True)


    # first find all the positions where a hole-checking event happened
    _,_,r_ls,_ = find_slowing_down_close_to_hole(track,hole_horizon,threshold_method          = threshold_method          ,
                                                                    gamma                     = gamma                     ,
                                                                    return_pos_from           = 'hole'                    ,
                                                                    ignore_entrance_positions = ignore_entrance_positions ,
                                                                    use_velocity_minima       = use_velocity_minima       ,
                                                                    velocity_min_prominence   = velocity_min_prominence   ,
                                                                    velmin_find_peaks_args    = velmin_find_peaks_args    )

    # sample all tracks together
    if type(r_ls) is list:
        r = numpy.vstack([(rr if rr.ndim==2 else rr[numpy.newaxis,:]) for rr in r_ls if rr.size>0]) # all positions where a hole was checked
    else:
        r = r_ls
    
    # calculate the distribution
    r_un,r_count = _find_unique_positions(r,grouping_hole_horizon)
    r_un         = numpy.array(r_un)
    r_count      = numpy.array(r_count,dtype=float)
    
    r_mean, r_cov, r_dispersion, r_eigdir = misc.calc_dispersion(r_un,r_count)
    if (r_count.size == 1):
        r_count  = r_count[0]*numpy.ones(5)
        r_un     = numpy.tile(r_un[0,:], (5,1))
    
    if _is_valid_normalize_by(normalize_by) and (normalize_by != 'none'):
        r_count  = r_count / (numpy.max(r_count) if (normalize_by == 'max') else numpy.sum(r_count))
    if sort_result:
        ind = numpy.argsort(r_count)
        r_un,r_count = r_un[ind],r_count[ind]
    return r_un,r_count,r_mean,r_cov,r_dispersion,r_eigdir

def _find_unique_positions(r,horizon=1e-8,return_indices=False):
    """
    r       -> numpy 2d array; 1 position per row
    horizon -> precision, such that if a position is closer to another than horizon, they are considered the same
    """
    if not misc._is_numpy_array(r):
        raise ValueError('r must be a 2d numpy array')
    checked = numpy.zeros(r.shape[0],dtype=bool)
    r_un    = []
    count   = []
    ind     = []
    for i in range(r.shape[0]):
        if checked[i]:
            continue
        r0 = r[i,:]
        j  = numpy.nonzero(numpy.linalg.norm(r[(i+1):,:] - r0,axis=1) < horizon)[0] # indices of all the elements that are within the boundaries of r0
        r_un.append(r0)
        count.append(1+j[numpy.logical_not(checked[j])].size)
        ind.append(i)
        checked[i] = True
        checked[j] = True
    if return_indices:
        return r_un,count,ind
    else:
        return r_un,count


def find_slowing_down_close_to_hole(track,hole_horizon,threshold_method          = 'ampv'  ,
                                                       gamma                     = 0.2     ,
                                                       return_pos_from           = 'mouse' ,
                                                       t1_ind = None, t2_ind     = None    ,
                                                       ignore_entrance_positions = False   ,
                                                       use_velocity_minima       = False   ,
                                                       velocity_min_prominence   = None    ,
                                                       velmin_find_peaks_args    = None    ,
                                                       join_vmin_vthresh_output  = True    ):
    """
     returns all slowing downs that happened inside any hole horizon
     if none is found, then returns an empty vector

     hole_horizon (cm) -> radius around the hole where mouse nose is close enough
     threshold_method,gamma,return_pos_from -> see function calc_velocity_crossings parameters
    
     return_pos_from -> 'mouse' or 'hole'
                        if 'mouse': returns positions of the mouse where the slowing down happened close to hole
                        if 'hole': returns positions of the holes where the slowing down happened

    t1_ind,t2_ind    -> time indices defining the portion of the track where this function will look for slowing downs

    returns
        * k -> index where a slowing down happened inside any hole event horizon
        * t -> time where a slowing down happened inside any hole event horizon
        * r -> position of the mouse or hole where the slowing down happened
        * v_th -> velocity threshold used for the calculations
    """
    assert (return_pos_from.lower() in ['mouse','hole']),"return_pos_from must be either 'mouse' or 'hole'"
    if type(track) is list:
        k_ls = misc.get_empty_list(len(track)) # index where a slowing down happened inside any hole event horizon
        t_ls = misc.get_empty_list(len(track)) # time where a slowing down happened inside any hole event horizon
        r_ls = misc.get_empty_list(len(track)) # position of the mouse or hole where the slowing down happened
        v_th_ls = numpy.zeros(len(track))
        for k,tr in enumerate(track):
            k_ls[k],t_ls[k],r_ls[k],v_th_ls[k] = find_slowing_down_close_to_hole(tr,hole_horizon,threshold_method = threshold_method, gamma = gamma,
                                                                                    return_pos_from = return_pos_from, t1_ind = t1_ind, t2_ind = t2_ind,
                                                                                    ignore_entrance_positions = ignore_entrance_positions  ,
                                                                                    use_velocity_minima       = use_velocity_minima        ,
                                                                                    velocity_min_prominence   = velocity_min_prominence    ,
                                                                                    velmin_find_peaks_args    = velmin_find_peaks_args     ,
                                                                                    join_vmin_vthresh_output  = join_vmin_vthresh_output   )
        return k_ls,t_ls,r_ls,v_th_ls
    else:
        return_mouse_pos            = return_pos_from.lower() == 'mouse'
        tind_inter,_,r_inter,r_hole = intersect_trajectory_arena_holes(track,hole_horizon,r_arena_holes=track.r_arena_holes,t1_ind=t1_ind,t2_ind=t2_ind,ignore_entrance_positions=ignore_entrance_positions)
        if use_velocity_minima:
            v_th                      = calc_threshold(track.velocity, gamma, threshold_method=threshold_method)
            tind_crossing,_           = find_velocity_minima(track, min_velocity_below=v_th, min_velocity_prominence=velocity_min_prominence, t1_ind=t1_ind, t2_ind=t2_ind, **misc._get_kwargs(velmin_find_peaks_args))
            tind_crossing_extra,_,_,_ = calc_velocity_crossings(track,threshold_method=threshold_method,gamma=gamma,only_slowing_down=True,t1_ind=t1_ind,t2_ind=t2_ind)
            #tind_crossing             = numpy.unique(numpy.concatenate((tind_crossing.flatten(),tind_crossing_extra.flatten())))
        else:
            tind_crossing,_,_,v_th    = calc_velocity_crossings(track,threshold_method=threshold_method,gamma=gamma,only_slowing_down=True,t1_ind=t1_ind,t2_ind=t2_ind)
            tind_crossing_extra       = numpy.array([])
        if (misc._is_numpy_array(v_th) and (v_th.size == 0)):
            v_th = 0.0
        k,t,r                         = _find_slowing_down_close_to_hole_internal(track,tind_crossing      ,tind_inter,return_mouse_pos,r_inter,r_hole)
        if tind_crossing_extra.size > 0:
            k_extra,t_extra,r_extra   = _find_slowing_down_close_to_hole_internal(track,tind_crossing_extra,tind_inter,return_mouse_pos,r_inter,r_hole)
            # check if each r_extra already exists in r
            if (k_extra.size > 0):
                if k.size>0:
                    ind = numpy.logical_not(numpy.array(is_in_hole_horizon(r_extra,hole_horizon,r_arena_holes=r,return_true_or_false=True)))
                    if numpy.any(ind): # if any r_extra is different from all r
                        if join_vmin_vthresh_output:
                            k = numpy.append(k,k_extra[ind])
                            t = numpy.append(t,t_extra[ind])
                            r = numpy.append(r,r_extra[ind],axis=0)
                        else:
                            k = [k,k_extra]
                            t = [t,t_extra]
                            r = [r,r_extra]
                else:
                    k = k_extra
                    t = t_extra
                    r = r_extra
        return k,t,r,v_th

def _find_slowing_down_close_to_hole_internal(track,tind_crossing,tind_inter,return_mouse_pos,r_inter,r_hole):
    k = [] # index where a slowing down happened inside any hole event horizon
    t = [] # time where a slowing down happened inside any hole event horizon
    r = [] # position of the mouse or hole where the slowing down happened
    for tind in tind_crossing:
        k_hole,t_hole = _find_hole(tind,tind_inter)
        if k_hole >= 0:
            if return_mouse_pos:
                r.append(r_inter[k_hole][t_hole])
            else:
                r.append(r_hole[k_hole])
            t.append(track.time[tind])
            k.append(tind)
    return numpy.asarray(k),numpy.asarray(t),numpy.asarray(r)

def _find_hole(tt,tind_inter):
    for k,tind in enumerate(tind_inter):
        if tt in tind:
            return k,numpy.nonzero(tind==tt)[0][0]
    return -1,None

#def find_slowing_down_close_to_hole(track,hole_horizon,threshold_method='ampv',gamma=0.2,return_pos_from='mouse'):
#    """
#     returns all slowing downs that happened inside any hole horizon
#     if none is found, then returns an empty vector
#     
#     this method calculates a single position for the slowing down, and then checks if this position is inside a hole horizon
#
#     a better method may be the find_slowing_down_close_to_hole_inter
#    
#     return_pos_from -> 'mouse' or 'hole'
#                        if 'mouse': returns positions of the mouse where the slowing down happened close to hole
#                        if 'hole': returns positions of the holes where the slowing down happened
#    """
#    tind_crossing,_,_,_ = calc_velocity_crossings(track,threshold_method=threshold_method,gamma=gamma,only_slowing_down=True)
#    if contains_nan(track.r_nose[:,0]):
#        track.r_nose = interp_trajectory(track.time,track.r_nose)
#    if tind_crossing.size > 0:
#        rr = is_in_hole_horizon(track.r_nose[tind_crossing], hole_horizon, track.r_arena_holes,return_pos_from=return_pos_from)
#        k = [] # index where a slowing down happened inside any hole event horizon
#        t = [] # time where a slowing down happened inside any hole event horizon
#        r = [] # position of the mouse or hole where the slowing down happened
#        for i,r0 in enumerate(rr):
#            if isinstance(r0,numpy.ndarray):
#                k.append(tind_crossing[i])
#                t.append(track.time[tind_crossing[i]])
#                r.append(r0)
#        k = numpy.asarray(k)
#        t = numpy.asarray(t)
#        r = numpy.asarray(r)
#    else:
#        r = tind_crossing
#        k = tind_crossing
#        t = tind_crossing
#    return k,t,r

def calc_velocity_crossings(track,threshold_method='ampv',gamma=0.2,only_slowing_down=False,return_first_crossing_type=False,t1_ind=None,t2_ind=None):
    """
     track -> data structure returned by the load_trial_file function
     only_slowing_down -> if True, returns only instants of down crossings (slowing down); otherwise, returns all instants of crossings (slowing down and speeding up)
     threshold_method -> 'meanv': the threshold is given by a percent of the mean x_th = gamma*mean(x)
                         'ampv': the threshold is given by a percent of the amplitude x_th = min(x) + gamma * (max(x) - min(x))
                         'abs': then, x_th = gamma (absolute threshold)
    
     returns
       tind_crossing -> index of the time and position variables where the crossing happens
       t_cross -> instants when v crosses the threshold
       IEI     -> interevent interval; time interval between every consecutive crossing; IEI[n] = t_cross[n] - t_cross[n-1]
       v_th -> the threshold calculated by this method
    """
    #t_nan = numpy.nonzero(numpy.isnan(track.velocity))[0] # first check for nan in the velocity and fix it
    #if t_nan.size > 0:
    if misc.contains_nan(track.velocity):
        track.velocity = interp_velocity(track.time,track.velocity,track.r_center)
    v_th = calc_threshold(track.velocity,gamma,threshold_method)
    tind_crossing,cross_type = calc_threshold_crossings(track.velocity,v_th,only_downward_crossing=only_slowing_down,return_first_crossing_type=True,t1_ind=t1_ind,t2_ind=t2_ind)
    t_cross = (track.time[tind_crossing] + track.time[tind_crossing+1])/2.0
    IEI = t_cross[1:] - t_cross[:-1]
    if return_first_crossing_type:
        return tind_crossing,t_cross,IEI,v_th,cross_type
    else:
        return tind_crossing,t_cross,IEI,v_th

def remove_path_after_food(track,r_target=None,return_t_to_food=False,force_main_target=False,hole_horizon=None,time_delay_after_food=0.0,copy_tracks=False,use_reverse_targets=False):
    """
    removes all the trajectory points from track that happen after the minimum distance between food and r_nose
    if return_t_to_food==True, then returns track.time[t_food_idx] as the second argument, as the minimum distance to r_target time

    track             -> a track file or a list of track files imported by load_trial_file
    r_target          -> target we use to trim the tracks (if None, then uses track.r_target)
    force_main_target -> if true, only considers r_target;
                         if false, then trims after the visit to the latest between r_target and track.r_target_alt

    if r_target is provided, then excludes all the trajectory after the minimum distance between r_target and r_nose
                if you set r_target, either set a unique target for all tracks (if more than one), or 1 target per track file
    """
    if type(track) is list:
        n_tracks = len(track)
        if n_tracks == 0:
            raise ValueError('remove_path_after_food ::: track is an empty list')
        #r_target = numpy.asarray(r_target)
        #if len(r_target.shape) == 1:
        #    r_target = numpy.tile(r_target,(n_mice,1))
        #else:
        #    if r_target.shape[0] != n_mice:
        #        raise ValueError('you must provide either a single target, or one target per track file')
        t_to_food  = misc.get_empty_list(n_tracks)
        track_copy = misc.get_empty_list(n_tracks) if copy_tracks else track
        for k,tr in enumerate(track):
            result = remove_path_after_food(tr,misc.get_element_or_none(r_target,k),return_t_to_food=True,force_main_target=force_main_target,hole_horizon=hole_horizon,time_delay_after_food=time_delay_after_food,copy_tracks=copy_tracks,use_reverse_targets=use_reverse_targets)
            track_copy[k] = result[0]
            t_to_food[k]  = result[1]
        if numpy.isscalar(t_to_food[0]):
            t_to_food = numpy.asarray(t_to_food)
    else:
        time_delay_after_food = time_delay_after_food if misc.exists(time_delay_after_food) else 0.0
        track_copy            = copy.deepcopy(track) if copy_tracks else track
        r_tgt                 = track_copy.r_target_reverse     if use_reverse_targets else track_copy.r_target
        r_tgt_alt             = track_copy.r_target_alt_reverse if use_reverse_targets else track_copy.r_target_alt
        skip = False
        if track_copy.IsField('remove_after_food') and track_copy.remove_after_food:
            skip = True
            t_to_food = track_copy.time[-1]
        if not skip:
            if type(r_target) is type(None):
                r_target = r_tgt
            if misc.contains_nan(track_copy.r_nose[:,0]):
                track_copy = fill_trajectory_nan_gaps(track_copy)
            if type(hole_horizon) is type(None):
                t1 = numpy.argmin(numpy.linalg.norm(track_copy.r_nose-r_target,axis=1))
                t2 = 0
                if not force_main_target:
                    if not numpy.any(numpy.isnan(r_tgt_alt)):
                        t2 = numpy.argmin(numpy.linalg.norm(track_copy.r_nose-r_tgt_alt,axis=1)) # + 1# + 1
                tind_inter = (t1,t2)
            else:
                if not force_main_target:
                    r_target = numpy.array((r_target,r_tgt_alt))
                tind_inter = find_first_intersection_index(track_copy.r_nose,r_target,time=track_copy.time,hole_horizon=hole_horizon)
            t_ind = _get_second_index(tind_inter,track_copy.time.size)
            if t_ind <= 0:
                t_ind = track_copy.time.size
            if (time_delay_after_food > 0.0):
                t_ind = _get_delayed_time_index(track_copy.time,time_delay_after_food,t_ind_start=t_ind-1)
            t_ind = _get_second_index((t_ind,),track_copy.time.size)
            track_copy.time      = track_copy.time[:t_ind]
            track_copy.r_nose    = track_copy.r_nose[:t_ind,:]
            track_copy.r_center  = track_copy.r_center[:t_ind,:]
            track_copy.r_tail    = track_copy.r_tail[:t_ind,:]
            track_copy.velocity  = track_copy.velocity[:t_ind]
            track_copy.direction = track_copy.direction[:t_ind]
            t_to_food = track_copy.time[-1]
    if return_t_to_food:
        return track_copy, t_to_food
    else:
        return track_copy

def _get_second_index(index_list,max_size):
    return numpy.min((numpy.max(index_list)+1,max_size))

def find_first_intersection_index(r,r_target,time=None,hole_horizon=None):
    """
    returns the index of time of the first intersection of the trajectory r(t) with each of the r_targets

    returns a list with the same len as the amount of targets passed to this function

    a list entry of -1 means that no intersection is found between a given target and the trajectory
    """
    assert (type(r) is numpy.ndarray) and (r.ndim == 2),'r must be 2d ndarray'
    assert (type(r_target) is numpy.ndarray),'r_target must be an ndarray, one target per row, or a single target'
    if type(time) is type(None):
        time = numpy.arange(r.shape[0])
    if r_target.ndim == 1:
        r_target = r_target.reshape((1,r_target.size))
    found_inter = lambda tind: (len(tind)>0) and (len(tind[0])>0)
    t_inter = [ -1 for _ in range(r_target.shape[0]) ]
    for k,r_tgt in enumerate(r_target):
        t_temp = intersect_trajectory_holes(r,hole_horizon,r_tgt,time=time)[0]
        if found_inter(t_temp):
            t_inter[k] = t_temp[0][0]
    return t_inter

def keep_path_between_targets(track,return_t_in_targets=False,hole_horizon=None,time_delay_after_food=0.0,copy_tracks=True,use_reverse_targets=False):
    """
    removes all the trajectory points from track that happen after the minimum distance between food and r_nose
    if return_t_to_food==True, then returns track.time[t_food_idx] as the second argument, as the minimum distance to r_target time

    track -> a track file or a list of track files imported by load_trial_file

    if r_target is provided, then excludes all the trajectory after the minimum distance between r_target and r_nose
                if you set r_target, either set a unique target for all tracks (if more than one), or 1 target per track file
    """
    if type(track) is list:
        n_tracks = len(track)
        #r_target = numpy.asarray(r_target)
        #if len(r_target.shape) == 1:
        #    r_target = numpy.tile(r_target,(n_mice,1))
        #else:
        #    if r_target.shape[0] != n_mice:
        #        raise ValueError('you must provide either a single target, or one target per track file')
        t_targets  = misc.get_empty_list(n_tracks)
        track_copy = misc.get_empty_list(n_tracks) if copy_tracks else track
        for k,tr in enumerate(track):
            track_copy[k],t_targets[k] = keep_path_between_targets(tr,return_t_in_targets=True,hole_horizon=hole_horizon,time_delay_after_food=time_delay_after_food,use_reverse_targets=use_reverse_targets)
    else:
        time_delay_after_food = time_delay_after_food if misc.exists(time_delay_after_food) else 0.0
        track_copy            = copy.deepcopy(track) if copy_tracks else track
        r_tgt                 = track_copy.r_target_reverse     if use_reverse_targets else track_copy.r_target
        r_tgt_alt             = track_copy.r_target_alt_reverse if use_reverse_targets else track_copy.r_target_alt
        skip = False
        if track_copy.IsField('keep_between_targets') and track_copy.keep_between_targets:
            skip = True
            t_targets = track_copy.time[[0,-1]]
        if not skip:
            if misc.contains_nan(track_copy.r_nose[:,0]):
                track_copy = fill_trajectory_nan_gaps(track_copy)
            if type(hole_horizon) is type(None):
                t1_ind,t2_ind = sorted([ numpy.argmin(numpy.linalg.norm(track_copy.r_nose-r_tgt,axis=1)), numpy.argmin(numpy.linalg.norm(track_copy.r_nose-r_tgt_alt,axis=1)) ]) # + 1# + 1
            else:
                tind_inter = find_first_intersection_index(track_copy.r_nose,numpy.array((r_tgt,r_tgt_alt)),time=track_copy.time,hole_horizon=hole_horizon)
                t1_ind,t2_ind = sorted(tind_inter)
                t1_ind = t1_ind if t1_ind > 0 else 0
                t2_ind = t2_ind if t2_ind > 0 else track_copy.time.size
            t2_ind = numpy.min((t2_ind+1,len(track_copy.time)))
            if time_delay_after_food > 0.0:
                t1_ind = _get_delayed_time_index(track_copy.time,time_delay_after_food,t_ind_start=t1_ind)
                t2_ind = _get_delayed_time_index(track_copy.time,time_delay_after_food,t_ind_start=t2_ind-1)
            track_copy.time      = track_copy.time[t1_ind:t2_ind]
            track_copy.r_nose    = track_copy.r_nose[t1_ind:t2_ind,:]
            track_copy.r_center  = track_copy.r_center[t1_ind:t2_ind,:]
            track_copy.r_tail    = track_copy.r_tail[t1_ind:t2_ind,:]
            track_copy.velocity  = track_copy.velocity[t1_ind:t2_ind]
            track_copy.direction = track_copy.direction[t1_ind:t2_ind]
            t_targets      = track_copy.time[[0,-1]]
    if return_t_in_targets:
        return track_copy, t_targets
    else:
        return track_copy

def _get_delayed_time_index(time,t_delay,t_ind_start=0):
    """
    finds index of time vector after a t_delay, starting at time[t_ind_start]
    """
    if t_ind_start == len(time):
        return t_ind_start - 1
    if t_ind_start == (len(time)-1):
        return t_ind_start
    t0 = time[t_ind_start]
    return t_ind_start + numpy.argmin(numpy.abs(time[t_ind_start:] - (t0 + t_delay)))

def fill_trajectory_nan_gaps(track):
    """
    # fills the nan gaps in r_nose, r_tail and r_center with linear interpolation
    # fills the velocity nan gaps with the average speed (constant) between the two positions before and after the nan gap
    # the constant speed is tied to the linear interpolation of r
    # track is the data returned by the plib.load_trial_file function
    """
    if type(track) is list:
        track = [ fill_trajectory_nan_gaps(tr) for tr in track ]
    else:
        track.r_nose   = interp_trajectory(track.time,track.r_nose)
        track.r_center = interp_trajectory(track.time,track.r_center)
        track.r_tail   = interp_trajectory(track.time,track.r_tail)
        track.velocity = interp_velocity(  track.time,track.velocity,track.r_center)
    return track

def interp_velocity(t,v,r):
    """
     t -> time
     v -> velocity of the mouse
     r -> position of the center (i.e. r_center)
     this function replaces the nan's in the v time series by constant velocity intervals (estimated based on average velocity = |delta r / delta t| )
    """
    v = v.copy()
    #t_nan = numpy.nonzero(numpy.isnan(numpy.prod(r,axis=1)))[0] # first check for nan in the trajectory
    #if t_nan.size > 0:
    if misc.contains_nan(r[:,0]):
        r = interp_trajectory(t,r) # if any nan in trajectory is found, then eliminate them by interpolation
    idx = misc.get_nan_chuncks(v,return_type='firstlast')
    if len(idx) > 0:
        for k in idx:
            # we replace the nan chuncks by the derivative of the position,
            # since we assume that the position varies linearly with time inside each chunck
            if (k[0] > 0) and (k[1] < len(v)):
                v[k[0]:k[1]] = numpy.linalg.norm(r[k[1],:] - r[k[0],:]) / ( t[k[1]] - t[k[0]] ) # |dr/dt|
            else:
                v[k[0]:k[1]] = numpy.zeros(k[1]-k[0]) # if v ends in nan, then the velocity in the end is zero (since position is assumed constant)
    return v

def interp_trajectory(t,r):
    """
     t -> time
     r -> position (either r_nose, r_tail or r_center)
    """
    r = r.copy()
    t_nan = numpy.nonzero(numpy.isnan(numpy.prod(r,axis=1)))[0] # first check for nan in the trajectory
    if t_nan.size > 0:
        t_fix = numpy.delete(t,t_nan)
        r_fix = numpy.delete(r,t_nan,axis=0)
    else:
        return r # no nan found
    interpolate_r = scipy.interpolate.interp1d(t_fix,r_fix,kind='linear',axis=0,copy=False)
    idx = misc.get_nan_chuncks(numpy.prod(r,axis=1),return_type='slice')
    if len(idx) > 0:
        for k in idx:
            if (k.start > 0) and (k.stop < len(t)): # if r ends in nan, the code won't be able to interpolate over the last chunck (it would need to extrapolate)
                r[k] = interpolate_r(t[k])
            else:
                if k.start == 0:
                    r0 = r[k.stop+1].reshape((1,r.shape[1])) # first known value when the recording starts with nan's
                else:
                    r0 = r[k.start-1].reshape((1,r.shape[1])) # last known value when the recording ends with nan's
                r[k] = numpy.repeat(r0,k.stop-k.start,axis=0) # repeating the last known value of r when the last nan chunck corresponds to the end of the recording
    return r

def calc_mouse_deviation(track,absolute_food_vec=True,return_angle=False,use_target_closest_to_end_position=False):
    """
     calculates the mouse deviation by defining the mouse and food vectors
     mouse_vec = r_nose - r_center
     the food vector depends on the parameter absolute_food_vec
     if absolute_food_vec is True, then the food vector is absolute:
          food_vec = r_target - r_entrance
     otherwise, the food vector is always the vector directly connecting the mouse center to the food
          food_vec = r_target - r_center
    
     RETURNS
      either the deviation given by
        mouse_vec dot food_vec / (|mouse_vec| * |food_vec|)
         this is the cosine of the angle between mouse and food vectors
     
     OR the actual angle if return_angle is True
        arccos( mouse_vec dot food_vec / (|mouse_vec| * |food_vec|) )
        (angle is negative if food is to the right of mouse; positive otherwise)
    """
    if type(track) is list:
        return [calc_mouse_deviation(tr,absolute_food_vec=absolute_food_vec,return_angle=return_angle,use_target_closest_to_end_position=use_target_closest_to_end_position) for tr in track]
    else:
        r_target = track.r_target
        if use_target_closest_to_end_position:
            d1 = numpy.linalg.norm(track.r_nose[-1,:]-track.r_target)
            d2 = numpy.linalg.norm(track.r_nose[-1,:]-track.r_target_alt)
            if d2 < d1:
                r_target = track.r_target_alt
        return calc_mouse_deviation_from_target(track,r_target,absolute_food_vec=absolute_food_vec,return_angle=return_angle)


def calc_mouse_deviation_from_target(track,r_target,absolute_food_vec=True,return_angle=False):
    """
     calculates the mouse deviation by defining the mouse and food vectors
     mouse_vec = r_nose - r_center
     the food vector depends on the parameter absolute_food_vec
     if absolute_food_vec is True, then the food vector is absolute:
          food_vec = r_target - r_entrance
     otherwise, the food vector is always the vector directly connecting the mouse center to the food
          food_vec = r_target - r_center
    
     RETURNS
      either the deviation given by
        mouse_vec dot food_vec / (|mouse_vec| * |food_vec|)
         this is the cosine of the angle between mouse and food vectors
     
     OR the actual angle if return_angle is True
        arccos( mouse_vec dot food_vec / (|mouse_vec| * |food_vec|) )
        (angle is negative if food is to the right of mouse; positive otherwise)
    """
    if type(track) is list:
        x = misc.get_empty_list(len(track))
        for k,tr in enumerate(track):
            x[k] = calc_mouse_deviation(tr,r_target,absolute_food_vec=absolute_food_vec,return_angle=return_angle)
        return x
    else:
        mouse_vec = track.r_nose - track.r_center
        if absolute_food_vec:
            food_vec = r_target - track.r_start
        else:
            food_vec = r_target - track.r_center
        c = misc.cos_uv(mouse_vec,food_vec,axis=1)
        if return_angle:
            return (2.0*misc.is_to_the_right(mouse_vec,food_vec,axis=1) - 1.0) * numpy.arccos(c)
        else:
            return c

def calc_time_to_food(track,r_target=None):
    """
    returns the time (in seconds) that the mouse took to get to the minimum distance to the food (or r_target if given)
    """
    _,t_food = remove_path_after_food(track,return_t_to_food=True,r_target=r_target,copy_tracks=True)
    return t_food

def calc_mouse_perp_dist_to_food_line(track,return_abs_value=False):
    """
    the food line is defined as the line that connects the entrance to the food

    this function calculates the perpendicular distance of the mouse to that line
    """
    if type(track) is list:
        return [calc_mouse_perp_dist_to_food_line(tr,return_abs_value=return_abs_value) for tr in track]
    else:
        return calc_mouse_perp_dist_to_line(track,track.r_start,track.r_target,return_abs_value=return_abs_value)

def calc_mouse_perp_dist_to_2target_line(track,return_abs_value=False):
    """
    the food line is defined as the line that connects r_target to r_target_alt

    this function calculates the perpendicular distance of the mouse to that line

    it is recommended to use only absolute values from this function, since the sign only tells about which side of the vector the mouse is
    """
    if type(track) is list:
        return [calc_mouse_perp_dist_to_2target_line(tr,return_abs_value=return_abs_value) for tr in track]
    else:
        return calc_mouse_perp_dist_to_line(track,track.r_target,track.r_target_alt,return_abs_value=return_abs_value)

def calc_mouse_perp_dist_to_line(track,p0,p1,return_abs_value=False):
    """
    the food line is defined as the line that connects p0 to p1

    this function calculates the perpendicular distance of the mouse to that line
    """
    p0 = p0 if misc._is_numpy_array(p0) else numpy.asarray(p0).flatten()
    p1 = p1 if misc._is_numpy_array(p1) else numpy.asarray(p1).flatten()
    if type(track) is list:
        d = misc.get_empty_list(len(track))
        for k,tr in enumerate(track):
            d[k] = calc_mouse_perp_dist_to_food_line(tr,return_abs_value)
        return d
    else:
        n = track.r_nose - p0 # this is the n vector from the calculation I sent you
        T = p1 - p0 # the T vector
        theta = misc.angle_uv(n,T)
        d = numpy.linalg.norm(n,axis=1) * numpy.sin(theta)
        return numpy.abs(d) if return_abs_value else d

def remove_outside_of_straightline(track,p0,p1,distance,mouse_part='nose',copy_track=True):
    """
    removes trajectory parts outside of "distance" perpendicular to the line defined by the p0,p1 points

    p0,p1    -> (2,) or (1,2) numpy.ndarray, containing (x,y) coordinates of the points that define the straight line
    distance -> perpendicular distance around the straight line to keep (in units of track.unit_r)

    returns
        trimmed track(s)
    """
    if type(track) is list:
        return [ remove_outside_of_straightline(tr,p0,p1,distance,mouse_part=mouse_part,copy_track=copy_track) for tr in track  ]
    else:
        r_field = 'r_'+mouse_part.lower()
        if not track.IsField(r_field):
            raise ValueError('unknown mouse_part')
        d    = calc_mouse_perp_dist_to_line(track,p0,p1,return_abs_value=True)
        k    = numpy.nonzero(d <= distance)[0]
        if copy_track:
            track = copy.deepcopy(track)
        return _trim_trajectory(track,k)

def remove_outside_foodline(track,distance,use_main_target=False,use_reverse_target=False,use_between_targets=False,use_between_reverse_targets=False,mouse_part='nose',copy_track=True):
    """
    removes trajectory outside of the food line

    use_main_target             -> if True (default), foodline defined by r_start          and r_target
    use_reverse_target          -> if True,           foodline defined by r_start          and r_target_reverse
    use_between_targets         -> if True,           foodline defined by r_target         and r_target_alt
    use_between_reverse_targets -> if True,           foodline defined by r_target_reverse and r_target_alt_reverse

    only 1 of these four parameters must be true

    returns
        trimmed track(s)
    """
    if type(track) is list:
        return [ remove_outside_foodline(tr,distance,use_main_target=use_main_target,use_reverse_target=use_reverse_target,use_between_targets=use_between_targets,use_between_reverse_targets=use_between_reverse_targets,mouse_part=mouse_part,copy_track=copy_track) for tr in track ]
    else:
        if not any((use_main_target,use_reverse_target,use_between_targets,use_between_reverse_targets)):
            use_main_target = True # set default
        if not _has_single_true(use_main_target,use_reverse_target,use_between_targets,use_between_reverse_targets):
            raise ValueError('only of the following can be True: use_main_target,use_reverse_target,use_between_targets,use_between_reverse_targets')
        if use_main_target:
            p0 = track.r_start
            p1 = track.r_target
        elif use_reverse_target:
            p0 = track.r_start
            p1 = track.r_target_reverse
        elif use_between_targets:
            p0 = track.r_target
            p1 = track.r_target_alt
        elif use_between_reverse_targets:
            p0 = track.r_target_reverse
            p1 = track.r_target_alt_reverse
        return remove_outside_of_straightline(track,p0,p1,distance,mouse_part=mouse_part,copy_track=copy_track)

def _has_single_true(*v):
    a = numpy.array(v,dtype=bool)
    return numpy.nonzero(a)[0].size == 1

def remove_slow_parts(track,threshold_method='ampv',gamma=0.2,return_threshold=False,copy_track=True):
    """
    removes parts of the trajectory where velocity is below a certain threshold
    """
    if type(track) is list:
        return [ remove_slow_parts(tr,threshold_method=threshold_method,gamma=gamma,return_threshold=return_threshold,copy_track=copy_track) for tr in track ]
    else:
        v_th = calc_threshold(track.velocity,gamma,threshold_method)
        k    = numpy.nonzero(track.velocity >= v_th)[0]
        if copy_track:
            track = copy.deepcopy(track)
        if return_threshold:
            return _trim_trajectory(track,k),v_th
        else:
            return _trim_trajectory(track,k)

def remove_outside_of_radius(track,r0,radius,mouse_part='nose',copy_track=True):
    """
    removes parts of the trajectory outside of the circle centered in r0 = (x0,y0), with the given radius
    """
    if type(track) is list:
        return [ remove_outside_of_radius(tr,r0,radius,mouse_part=mouse_part,copy_track=copy_track) for tr in track ]
    else:
        r_field = 'r_'+mouse_part.lower()
        if not track.IsField(r_field):
            raise ValueError('unknown mouse_part')
        r0   = numpy.asarray(r0)
        d    = numpy.linalg.norm(track[r_field] - r0,axis=1)
        k    = numpy.nonzero(d <= radius)[0]
        if copy_track:
            track = copy.deepcopy(track)
        return _trim_trajectory(track,k)


def _get_start_end_time_slice(time,t0_frac=0.0,dt_frac=1.0,return_idx=False):
    """
    t0_frac,dt_frac -> track.time has T elements, so the analysis will be made from T0=floor(t0_frac*T) T0:min(T0+ceil(dt_frac*T),T)
    return_idx      -> if true, returns a tuple of (T0,Tmax) integer indices; otherwise returns a slice object
    """
    if type(time) is list:
        return [ _get_start_end_time_slice(tt,t0_frac=t0_frac,dt_frac=dt_frac,return_idx=return_idx) for tt in time ]
    if not(type(time) is numpy.ndarray):
        time = numpy.asarray(time)
    T    = float(time.size)
    T0   = int(numpy.floor(t0_frac * T))
    Tmax = numpy.min(( int(T), T0+int(numpy.ceil(dt_frac * T)) ))
    if return_idx:
        return T0,Tmax
    else:
        return slice(T0,Tmax)

def slice_track_by_time(track,dt=None,t0=None,t1=None,copy_track=True):
    """
    slice time by time interval

    if t0 and t1 are given    -> time between [t0,t1]
    if t0 is given (no t1)    -> time between [t0,t0+dt]
    if t1 is given (no t0)    -> time between [t1-dt,t1]
    if dt is given (no t0,t1) -> positive dt: time between [end-dt,end];
                                 negative dt: time between [0,end-abs(dt)]
    """
    if type(track) is list:
        return [ slice_track_by_time(tr,dt=dt,t0=t0,t1=t1,copy_track=copy_track) for tr in track ]
    else:
        if not ( misc.exists(t0) or misc.exists(t1) ):
            # neither t0 nor t1 are given
            if not misc.exists(dt):
                raise ValueError('at least one of t1,t0,dt must be given')
            if dt > 0:
                t1 = track.time[-1]
                t0 = t1 - dt
            else:
                t0 = track.time[0]
                t1 = track.time[-1] - numpy.abs(dt)
        else:
            # either t0 or t1 (or both) are given
            if not misc.exists(t0):
                t0 = t1 - dt
            if not misc.exists(t1):
                t1 = t0 + dt
        t0,t1 = sorted((t0,t1))
        t0_ind = numpy.argmin(numpy.abs(track.time-t0))
        t1_ind = numpy.argmin(numpy.abs(track.time-t1))
        return slice_track(track,slice(t0_ind,t1_ind),copy_track=copy_track)

def slice_track_by_time_frac(track,t0_frac=0.0,dt_frac=1.0,copy_track=True):
    assert (t0_frac >= 0.0) and (t0_frac <= 1.0), 't0_frac must be a value between 0 and 1'
    assert (dt_frac >= 0.0) and (dt_frac <= 1.0), 't0_frac must be a value between 0 and 1'
    if (t0_frac == 0.0) and (dt_frac >= 1.0):
        return track
    return slice_track(track,_get_start_end_time_slice(_get_track_time(track),t0_frac,dt_frac,return_idx=False),copy_track=copy_track)

def _get_track_time(track):
    if type(track) is list:
        return [ _get_track_time(tr) for tr in track ]
    else:
        return track.time

def slice_track(track,slices,copy_track=True):
    """
    slices time indices of every trajectory variable in track
    according to slices

    slices is either a slice() object or a list of slice() objects
    """
    if type(track) is list:
        slc = slices
        if not(type(slc) is list):
            slc = [ slices for _ in track ]
        return [ slice_track(tr,sl,copy_track=copy_track) for tr,sl in zip(track,slc) ]
    else:
        if type(slices) is slice:
            if (slices.start == 0) and (slices.stop == track.time.size):
                return track
        if copy_track:
            track = copy.deepcopy(track)
        k = numpy.r_[slices]
        return _trim_trajectory(track,k)

def _trim_trajectory(track,ind_to_keep):
    track.time      =      track.time[ind_to_keep]
    track.direction = track.direction[ind_to_keep]
    track.velocity  =  track.velocity[ind_to_keep]
    track.r_tail    =    track.r_tail[ind_to_keep,:]
    track.r_center  =  track.r_center[ind_to_keep,:]
    track.r_nose    =    track.r_nose[ind_to_keep,:]
    return track

def calc_curvature(track,mouse_part='nose',eps=1.0e-10):
    """
    calculates the curvature of the trajectory given by track.r_PART, where PART is the mouse_part
    https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization
    https://web.ma.utexas.edu/users/m408m/Display13-4-3.shtml

    eps -> derivative precision
    mouse_part -> 'nose', 'center', 'tail' (the tracked mouse part to use in this calculation)

    curvature is defined as
    K = |x'y''-y'x''|/(x'**2 + y'**2)**(3/2)
    where the prime refers to time derivative

    returns K(t)
    """
    if type(track) is list:
        K = misc.get_empty_list(len(track))
        for k,tr in enumerate(track):
            K[k] = calc_curvature(tr,mouse_part=mouse_part,eps=eps)
    else:
        track = fill_trajectory_nan_gaps(track)
        r = track['r_'+mouse_part]
        x1 = misc.derivative(track.time,r[:,0],interpolate=True,epsilon=eps) # first derivative
        x2 = misc.derivative(track.time,x1,interpolate=True,epsilon=eps) # second derivative
        y1 = misc.derivative(track.time,r[:,1],interpolate=True,epsilon=eps) # first derivative
        y2 = misc.derivative(track.time,y1,interpolate=True,epsilon=eps) # second derivative
        with numpy.errstate(all='ignore'): # ignore 0/0 type of error (for when the mouse is not moving, such that x1=y1=0)
            K = numpy.abs( x1 * y2 - y1 * x2 ) / numpy.sqrt( (x1**2.0 + y1**2.0)**3.0 )
            idx = numpy.nonzero(numpy.isnan(K))[0]
            if idx.size > 0:
                K[idx] = numpy.sqrt(x2[idx]**2+y2[idx]**2) # when the mouse is not moving, assuming that the curvature is the norm of the acceleration
    return K


def calc_distribution(x,n_bins=25,x_edges=None,return_as_struct=False,binning='linear',join_samples=False,replace_Peq0_by_nan=False,remove_Peq0=False):
    """
    calculates the distribution (histogram) of x

    if x is list, then calculates the average distribution of x, all within the same x edges

    returns:
        * x mid points
        * P(x)
        * P(x) std dev
        * P(x) std err
    """
    return misc.calc_distribution(x,n_bins=n_bins,x_edges=x_edges,
                                    return_as_struct=return_as_struct,
                                    binning=binning,join_samples=join_samples,
                                    replace_Peq0_by_nan=replace_Peq0_by_nan,remove_Peq0=remove_Peq0)

def avg_feature(x,backwards=False,cos_to_avg_angle=False):
    """
    averages all x together from initial time
    
    backwards        :: if True, then align all features from the last to first before averaging
    cos_to_avg_angle :: if True, then x is a list of cosines, and the return will be the average angle

    returns
        - the average x(t)
        - the number of valid x for each t
    """
    if not(type(x) is list):
        return x,numpy.zeros(x.size),numpy.ones(x.size,dtype=int)
    flip_func = lambda s: s
    if backwards:
        flip_func = numpy.flip
        x = [flip_func(xx) for xx in x]
    if cos_to_avg_angle:
        x_avg,x_std,x_err,n_valid=misc.avg_angle_from_cos(misc.asarray_nanfill(x),axis=0,in_radians=False)
    else:
        x_avg,x_std,x_err,n_valid=misc.avg_count_nan(misc.asarray_nanfill(x),axis=0)
    return flip_func(x_avg),flip_func(x_std),flip_func(x_err),flip_func(n_valid)

def avg_velocity(track,from_food=False):
    """
    averages all velocities in track together from initial time or from food
    
    from_food :: if True, then align the food find times,
                 then do the averages backwards in time from there

    returns
        - the average v(t)
        - the stddev of v(t)
        - the number of valid v (not nan's) for each t
    """
    if not(type(track) is list):
        return track.velocity,numpy.zeros(track.velocity.size),numpy.ones(track.time.size,dtype=int),remove_path_after_food(track,return_t_to_food=True)[1]
    flip_func = lambda s: s
    if from_food:
        track = remove_path_after_food(track)
        flip_func = numpy.flip
    N = len(track)
    v = misc.get_empty_list(N)
    t_to_food = numpy.zeros(N)
    for k,tr in enumerate(track):
        v[k] = flip_func(tr.velocity)
        t_to_food[k] = remove_path_after_food(tr,return_t_to_food=True)[1]
    v,v_std,v_err,n_v_valid = misc.avg_count_nan(misc.asarray_nanfill(v),axis=0)
    return flip_func(v),flip_func(v_std),flip_func(v_err),flip_func(n_v_valid),t_to_food

def calc_trajectory_range(track,mouse_part='nose',from_food=False,relative_to_food=False,use_polar_coord=False,use_err=False):
    """
    averages all trajectories in track list, then calculates a trajectory range around the average trajectory

    the leftmost and rightmost trajectories are calculated based on the standard deviation of the averaged trajectory

    mouse_part       :: 'nose', 'center', 'tail' (the tracked mouse part to use in this calculation)
    from_food        :: if True, then align the food find times,
                        then do the averages backwards in time from there
    relative_to_food :: if True, considers r_new = r-r_target, such that food is at r_new = 0 (vector)
    use_polar_coord  :: if True, converts to polar coordinates before averaging, then converts back to cartesian

    returns 3 trajectories (cartesian coord), the average, the left-most and right-most trajectories,
    and the number of valid points for each time, and the time to food for each:
        r_avg,r_left,r_right,n_valid,t_to_food
    """
    lab = 'r_' + mouse_part
    if not(type(track) is list):
        return track[lab],track[lab],track[lab],numpy.ones(track.time.size,dtype=int),remove_path_after_food(track,return_t_to_food=True)[1]
    rth,rth_std,rth_err,n_valid,t_to_food = avg_trajectory(track,mouse_part=mouse_part,from_food=from_food,relative_to_food=relative_to_food,use_polar_coord=use_polar_coord)
    r1 = rth.copy()
    r2 = rth.copy()
    #r1[:,0] -= rth_std[:,0]
    #r2[:,0] += rth_std[:,0]
    if use_err:
        r1 -= rth_err
        r2 += rth_err
    else:
        r1 -= rth_std
        r2 += rth_std
    if use_polar_coord:
        return misc.to_cartesian(rth),misc.to_cartesian(r1),misc.to_cartesian(r2),n_valid,t_to_food
    else:
        return rth,r1,r2,n_valid,t_to_food

def avg_trajectory(track,mouse_part='nose',from_food=False,relative_to_food=False,use_polar_coord=False):
    """
    averages all trajectories in track together from initial time or from food
    
    mouse_part       :: 'nose', 'center', 'tail' (the tracked mouse part to use in this calculation)
    from_food        :: if True, then align the food find times,
                        then do the averages backwards in time from there
    relative_to_food :: if True, considers r_new = r-r_target, such that food is at r_new = 0 (vector)
    use_polar_coord  :: if True, converts to polar coordinates before averaging, then converts back to cartesian

    returns
        - the average r(t), x==r[:,0] and y==r[:,1]; or if use_polar_coord, x->radius,y->theta
        - the stddev of r(t)
        - the number of valid x (not nan's) for each t
    """
    lab = 'r_' + mouse_part
    if not(type(track) is list):
        return track[lab],numpy.zeros(track[lab].shape),numpy.ones(track.time.size,dtype=int),remove_path_after_food(track,return_t_to_food=True)[1]
    flip_func = lambda s: s
    if from_food:
        track = remove_path_after_food(track)
        flip_func = numpy.flipud
    N = len(track)
    x = misc.get_empty_list(N)
    y = misc.get_empty_list(N)
    t_to_food = numpy.zeros(N)
    for k,tr in enumerate(track):
        r_food = numpy.zeros(2)
        if relative_to_food:
            r_food = tr.r_target
        if use_polar_coord:
            # x is the radius
            # y is the angle
            x[k],y[k]=[ flip_func(val) for val in misc.to_polar(tr[lab]-r_food) ]
        else:
            x[k] = flip_func(tr[lab][:,0]) - r_food[0]
            y[k] = flip_func(tr[lab][:,1]) - r_food[1]
        t_to_food[k] = remove_path_after_food(tr,return_t_to_food=True)[1]
    x,x_std,x_err,n_x_valid = misc.avg_count_nan(misc.asarray_nanfill(x),axis=0)
    y,y_std,y_err,_         = misc.avg_count_nan(misc.asarray_nanfill(y),axis=0)
    return numpy.column_stack((flip_func(x),flip_func(y))),numpy.column_stack((flip_func(x_std),flip_func(y_std))),numpy.column_stack((flip_func(x_err),flip_func(y_err))),flip_func(n_x_valid),t_to_food

def avg_step_displacement(track):
    """
    returns the average distance traveled between consecutive time steps
    """
    if type(track) is list:
        return [ avg_step_displacement(tr) for tr in track ]
    else:
        return misc.nanmean(track.velocity[:-1] * numpy.diff(track.time))

def avg_value_over_displacement(track,x,d,r0=None,n_r0_sample=1,mouse_part='nose',return_angle_from_cos=False,subtract_x_in_r0=False):
    """
    this function averages x over the pair of points (r,r0) in the trajectory that are separated by a distance d

    e.g., d = track.r_target - track.r_start, giving say |d| = 10 cm
    and mouse_part == nose, giving track['r_'+mouse_part] = track.r_nose
    then, all the time points in the trajectory r=track.r_nose
    such that the condition M |r(t_n)-r0| = d (within precision) is satisfied
    will be used to calculate the average of x:
    <x> = (1/T) sum_n x(t_n)
    t0 is t for r==r0
    and T == number of points that satisfy the condition M; the sum is over the time interval in which the condition is valid

    track       -> a track file or list
    x           -> a time series variable that will be averaged
                   x is a numpy.ndarray such that
                   x.shape == track.time.shape
                   if track is a list, x must a list that matches the length of track
    d           -> displacement vector (or distance scalar) defining the distance over which to take the average of x
                   or distance (i.e., norm of the displacement vector)
                   measured in track.unit_r
    r0          -> reference point in the trajectory to measure d (e.g., r0=track.r_nose[0,:] )
                   if track is a list, r0 must be either None or a list with the same length as track
    n_r0_sample -> 1 (only uses r0=track.r_..[0,:]); if > 1, then selects n_r0 random r0, and averages the averages over the r0
    mouse_part  -> 'nose','center','tail'

    returns
        * average of x (for single track) or list of averages of each x (for a list of track)
        * distance or list of distances that best matched d and were used to calculate average
        * other statistical parameters (stddev, stderr, min, max of x for dist)
    """
    if type(track) is list:
        n_tracks = len(track)
        assert ((type(x) is list) and (len(x) == n_tracks)), 'x must be a list when track is a list of the same length as track'
        if type(r0) is list:
            assert len(r0) == n_tracks, 'r0 list must be the same length as the track list'
        else:
            r0 = [r0 for _ in range(n_tracks)]
        x_avg        = misc.get_empty_list(n_tracks)
        dist_for_avg = misc.get_empty_list(n_tracks)
        x_std        = misc.get_empty_list(n_tracks)
        x_err        = misc.get_empty_list(n_tracks)
        x_min        = misc.get_empty_list(n_tracks)
        x_max        = misc.get_empty_list(n_tracks)
        for k,(tr,xx,rr) in enumerate(zip(track,x,r0)):
            x_avg[k],dist_for_avg[k],x_std[k],x_err[k],x_min[k],x_max[k] = avg_value_over_displacement(tr,xx,d,r0=rr,n_r0_sample=n_r0_sample,mouse_part=mouse_part,return_angle_from_cos=return_angle_from_cos,subtract_x_in_r0=subtract_x_in_r0)
        return x_avg,dist_for_avg,x_std,x_err,x_min,x_max
    else:
        assert x.size == track.time.size,'x (or x elements) must be the same size as the corresponding track.time'
        mouse_part = mouse_part.lower()
        assert mouse_part in ['nose','center','tail'],"mouse_part must be one of 'nose','center','tail'"
        lab = 'r_' + mouse_part
        #if type(precision) is type(None):
        #    precision = avg_step_displacement(track)
        if type(r0) is type(None):
            r0 = track[lab][0,:]
            k0_ind = 0
        else:
            r0 = numpy.asarray(r0)
            k0_ind = find_minimum_distance(track[lab],r0) #find_first_point_in_trajectory(track,r0,precision,0,mouse_part)
        if type(d) is numpy.ndarray:
            d = numpy.linalg.norm(d.flatten())
        assert ( (x.ndim == 1) and (x.shape == track.time.shape) ), "x must be the same shape as track.time"
        if n_r0_sample > 1:
            r0 = track[lab][random.sample(range(track.time.size),n_r0_sample),:] # selecting n random r0
            x_avg,dist,x_std,x_err,x_min,x_max = misc.unpack_list_of_tuples([ avg_value_over_displacement(track,x,d,r0=rr0,n_r0_sample=1,mouse_part=mouse_part,return_angle_from_cos=False,subtract_x_in_r0=subtract_x_in_r0) for rr0 in r0 ])
            d_avg = misc.nanmean(numpy.array( dist ))
            if return_angle_from_cos:
                x_avg,x_std,x_err,_,x_min,x_max = misc.avg_angle_from_cos(x_avg,return_minmax=True)
            else:
                #x_avg,x_std,x_err,x_min,x_max = misc.avg_of_avg(x_avg,x_std,x_err,x_min=x_min,x_max=x_max)
                #x_avg, d_avg, x_std, x_err, x_min, x_max = misc.nanmean(numpy.array( x_avg )), misc.nanmean(numpy.array( dist )), numpy.sqrt(misc.nanmean(numpy.array(x_std)**2)), numpy.sqrt(misc.nanmean(numpy.array(x_err)**2)), misc.nanmin(numpy.array( x_min )), misc.nanmax(numpy.array( x_max ))
                x_avg,x_std,x_err,x_min,x_max   = misc.mean_std_err_minmax(x_avg)
            return x_avg, d_avg, x_std, x_err, x_min, x_max
        else:
            s = numpy.linalg.norm(track[lab] - r0,axis=1) # distance to r0
            #t = numpy.nonzero(numpy.abs(s - d) < precision)[0] # time steps in which the distance is of the order of d (within precision)
            k_min = numpy.argmin(numpy.abs(s - d))
            t0,t1 = sorted((k0_ind,k_min))
            t1 = _get_second_index((t1,),x.size)
            x0 = x[t0] if subtract_x_in_r0 else 0.0
            if return_angle_from_cos:
                x_avg,x_std,x_err,_,x_min,x_max = misc.avg_angle_from_cos(x[t0:t1] - x0,return_minmax=True)
            else:
                x_avg,x_std,x_err,x_min,x_max   = misc.mean_std_err_minmax(x[t0:t1] - x0)
            return x_avg, s[k_min], x_std, x_err, x_min, x_max

def find_minimum_distance(r,r0):
    """
    finds minimum distance between the 2d-ndarray r and the reference r0
    r has 1 vector per row

    if r0 is 2d-array, returns a list of minimum distances indices from r to each point in r0 (1 r0 per row)
    """
    if not(type(r0) is numpy.ndarray):
        r0 = numpy.asarray(r0)
    if not(type(r) is numpy.ndarray):
        r = numpy.asarray(r)
    if r.ndim == 1:
        r = r[numpy.newaxis,:]
    if r0.ndim == 1:
        r0 = r0[numpy.newaxis,:]
    ind = [ numpy.argmin(numpy.linalg.norm(r-r0[k,:],axis=1)) for k in range(r0.shape[0]) ]
    return misc._get_zero_or_same(ind)

def find_min_distance_to_alt_target(track,mouse_part='nose'):
    mouse_part = mouse_part.lower()
    assert mouse_part in ['nose','center','tail'],'Mouse part must be nose, center or tail'
    if type(track) is list:
        return [ find_min_distance_to_alt_target(tr,mouse_part=mouse_part) for tr in track ]
    else:
        rmin = track['r_'+mouse_part][find_minimum_distance(track['r_'+mouse_part],track.r_target_alt),:]
        return numpy.linalg.norm(rmin-track.r_target_alt)

def find_min_distance_to_random_point(track,mouse_part='nose',n_points=10):
    mouse_part = mouse_part.lower()
    assert mouse_part in ['nose','center','tail'],'Mouse part must be nose, center or tail'
    if type(track) is list:
        return [ find_min_distance_to_random_point(tr,mouse_part=mouse_part,n_points=n_points) for tr in track ]
    else:
        r0_many   = _random_point_in_circle(track.r_arena_center,track.arena_diameter/2.0,N=n_points)
        d = 0.0
        for r0 in r0_many:
            rmin  = track['r_'+mouse_part][find_minimum_distance(track['r_'+mouse_part],r0),:]
            d    += numpy.linalg.norm(rmin-r0)
        return d/float(n_points)

def _random_point_in_circle(c,R,N=None):
    """
    returns a random point (x,y) within a circle centered in c with radius R
    """
    c     = numpy.array([c]).flatten()
    r     = R * numpy.sqrt(numpy.random.random(N))
    theta = numpy.random.random(N) * 2.0 * numpy.pi
    rr    = numpy.array(( r*numpy.cos(theta), r*numpy.sin(theta) ))
    if misc.exists(N) and (N > 1):
        c  = c[numpy.newaxis,:]
        rr = rr.T
    return c+rr

def remove_mice_from_track_list(track_list,omit_mice):
    """
    removes the given mouse number from the given trials (both given in omit_mice dict)
    input_tracks -> list of input trackfile's
    omit_mice    -> dict   :: key   -> trial number (plib.trial_to_number or trial raw str from plib)
                           :: value -> list of mice to omit in the given trial
                    dict containing the list of mice to skip in each trial (if any)
    returns
        a filteres list of input_tracks
    """
    has_mice_to_skip = misc.exists(omit_mice) and (len(omit_mice) > 0)
    if not has_mice_to_skip:
        return track_list
    #omit_mice = { plib.trial_to_number(trial):[ int(mouse) for mouse in mice ] for trial,mice in omit_mice.items() } # making sure the trials and the mouse numbers are numbers
    omit_mice_trial_mouse_tup = [ (plib.trial_to_number(trial),int(mouse)) for trial,mice in omit_mice.items() for mouse in mice ]
    return _remove_mice_from_track_list_by_tuple(track_list,omit_mice_trial_mouse_tup)

def _remove_mice_from_track_list_by_tuple(track_list,omit_mice_trial_mouse):
    #skip_none  = lambda values: list(filter(lambda v: misc.exists(v),values)) if (type(v) is list) else v if
    if type(track_list) is list:
        return list(filter(lambda item: misc.exists(item),[ _remove_mice_from_track_list_by_tuple(track,omit_mice_trial_mouse) for track in track_list ]))
    else:
        #skip_mouse = lambda trial,mouse,track: (plib.trial_to_number(track.trial) == trial) and (int(track.mouse_number) == mouse)
        if (plib.trial_to_number(track_list.trial),int(track_list.mouse_number)) in omit_mice_trial_mouse:
            return None
        else:
            return track_list # track_list is not a list here