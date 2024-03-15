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

__HAS_WX__ = False

import os # path and file handling
import sys
import ast
import glob # name pattern expansion
import numpy
import numpy.core.records
import functools
import re as regexp
import scipy.io
import scipy.sparse
import scipy.spatial
import scipy.optimize
import scipy.interpolate
import warnings
import modules.traj_analysis as tran
import modules.helper_func_class as misc
import modules.traj_to_step_matrix as tstep
import modules.process_mouse_trials_lib as plib

def save_selfintersection_sim_data(output_dir,output_filename,selfint_st_rt,exper_info_rt,trial_labels_rt,selfint_st_ft,exper_info_ft,trial_labels_ft,mouse_part):
    if output_filename[-4:].lower() != '.mat':
        output_filename += '.mat'
    selfint_st_rt = _fix_list_of_empty_list(selfint_st_rt)
    selfint_st_ft = _fix_list_of_empty_list(selfint_st_ft)
    scipy.io.savemat(os.path.join(output_dir,output_filename),
        dict(selfint_st_rt   = list_of_arr_to_arr_of_obj(selfint_st_rt),  selfint_st_ft   = list_of_arr_to_arr_of_obj(selfint_st_ft),
             exper_info_rt   = list_of_arr_to_arr_of_obj(exper_info_rt),  exper_info_ft   = list_of_arr_to_arr_of_obj(exper_info_ft),
             trial_labels_rt = list_of_arr_to_arr_of_obj(trial_labels_rt),trial_labels_ft = list_of_arr_to_arr_of_obj(trial_labels_ft), mouse_part=mouse_part),
             long_field_names=True, do_compression=True )

def _fix_list_of_empty_list(L):
    return [ list_of_arr_to_arr_of_obj(x) for x in L ]

def get_img_extent_from_header(file_name,hchar='#'):
    header_txt = read_txt_header(file_name,hchar=hchar)
    return ast.literal_eval([ l for l in header_txt if 'cropped image extent (cm) = [' in l ][0].split('=')[1].strip())

def read_txt_header(file_name,hchar='#'):
    f_content = []
    with open(file_name,'r') as f:
        line = 'txt'
        while line:
            line = f.readline().strip()
            if (len(line) > 0) and (line[0] != hchar):
                continue
            l = line[1:].strip()
            if len(l) > 0:
                f_content.append(l)
        f.close()
    return f_content

def write_txt_header_from_dict(file_name,data_dict,append=False,replace=True,hchar='#',verbose=True):
    """
    writes the following lines in file_name for each key in data_dict:
    # ...
    # key=value
    # ...

    if data_dict is a list of dict, then adds two blank lines between each dict in the list
    """
    open_mode = 'w'
    if append:
        open_mode = 'a'
    else:
        if not replace:
            file_name = check_and_get_filename(file_name)
    f_content = []
    with open(file_name,open_mode,encoding='utf-8') as f:
        if type(data_dict) is list:
            data_dict_list = data_dict
        else:
            data_dict_list = [data_dict]
        for n,dd in enumerate(data_dict_list):
            for k,v in dd.items():
                line = '%s %s=%s\n'%(hchar,str(k),str(v))
                f_content.append(line)
                f.write(line)
            if (len(data_dict_list)>1) and (n < (len(data_dict_list)-1)):
                line = '\n\n'
                f_content.append(line)
                f.write(line)
        f.close()
    if verbose:
        print('*** file written: ', file_name)
    return file_name,f_content

def save_step_probability_file_struct(out_dir,step_file_struct,ntrials=None,filename_suffix=''):
    #step_file_struct.use_extra_trials = False                            if not step_file_struct.IsField('use_extra_trials') else step_file_struct.use_extra_trials
    #step_file_struct.prob_calc        = misc.ProbabilityType.independent if not step_file_struct.IsField('prob_calc'       ) else step_file_struct.prob_calc
    #param_names = ['mouse_part','n_stages','L_lattice','prob_calc','start_from_zero','use_extra_trials','stop_at_food']
    nstages = len(step_file_struct.P)
    if not ntrials:
        ntrials = 16 if step_file_struct.use_extra_trials else 14
    ptype                           = 'cstep' if step_file_struct.prob_calc == misc.ProbabilityType.cumulative_step else ('cprob' if step_file_struct.prob_calc == misc.ProbabilityType.cumulative_prob else 'indept')
    ic                              = '0.0' if step_file_struct.start_from_zero else '0.25'
    st                              = 'stopfood' if step_file_struct.stop_at_food else 'continuefood'
    align_method                    = 'ent' if (step_file_struct.align_method=='entrance') else 'tgt'
    step_file_struct.prob_calc      = str(step_file_struct.prob_calc)
    step_file_struct.arena_geometry = list_of_arr_to_arr_of_obj([struct_array_for_scipy('r_center,arena_radius,lattice_extent',*_get_step_prob_arena_geometry_fields(s)) for s in step_file_struct.arena_geometry])
    #traj          = struct_array_for_scipy('mouse_r,t_food,run,totalruns',list_of_arr_to_arr_of_obj(traj_mouse_r),list_of_arr_to_arr_of_obj(traj_t_food),traj_run,traj_totalruns)
    filename_suffix = '_' + filename_suffix if len(filename_suffix) > 0 else ''
    out_fname       = 'stepmat_%s_L_%d_nstages_%d_ntrials_%d_Pinit_%s_%s_%s_align_%s%s.mat'%(step_file_struct.mouse_part,step_file_struct.L_lattice,nstages,ntrials,ic,ptype,st,align_method,filename_suffix)
    out_fname       = os.path.join(out_dir,out_fname)
    print(' - saving ::: %s'%out_fname)
    scipy.io.savemat(out_fname,
                     dict(**step_file_struct),
                     long_field_names=True,do_compression=True)

def _get_step_prob_arena_geometry_fields(arena_geometry):
    # arena_geometry = misc.structtype(r_center=r_center,arena_radius=plib.get_arena_diameter_cm()/2.0,lattice_extent=misc.flatten_list([arena_dx,numpy.flip(arena_dy)]))
    r_center       = numpy.empty(len(arena_geometry),dtype=object)
    arena_radius   = numpy.empty(len(arena_geometry),dtype=object)
    lattice_extent = numpy.empty(len(arena_geometry),dtype=object)
    for k,s in enumerate(arena_geometry):
        r_center[k]        =    s.r_center
        arena_radius[k]    =    s.arena_radius
        lattice_extent[k]  =    s.lattice_extent
    return r_center,arena_radius,lattice_extent

def save_step_probability_file(out_dir,param_struct,P,N,G,stage_trials,n_trials,r_target,r_mouse,
                               t_to_food,mouse_id,trial_id,P_specific,mouse_id_spec,
                               r_target_trial,r_target_previous_trial,r_target_alt_trial,r_target_rev_trial,r_target_revalt_trial,
                               arena_geometry,ntrials=None):
    save_step_probability_file_struct(out_dir,
                                      tstep.get_step_probability_file_struct(param_struct,P,N,G,stage_trials,n_trials,r_target,r_mouse,
                                                                             t_to_food,mouse_id,trial_id,P_specific,mouse_id_spec,
                                                                             r_target_trial,r_target_previous_trial,r_target_alt_trial,r_target_rev_trial,r_target_revalt_trial,
                                                                             arena_geometry),
                                      ntrials=ntrials)

def fix_step_probability_output_dir_structure(out_dir,n_trials):
    script = _get_fix_step_probability_output_dir_structure_script(n_trials)
    prev_dir = os.getcwd()
    os.chdir(os.path.join(prev_dir,out_dir))
    os.system(script)
    os.chdir(prev_dir)

def _get_fix_step_probability_output_dir_structure_script(n_trials):
    if sys.platform == 'win32':
        script_txt = """PowerShell -c "$ntrials = %d;$folders = @('cumul_prob', 'cumul_step', 'independent');Write-Host 'Fixing dir structure ...';foreach ($folder in $folders){    if (-not (Test-Path -Path $folder))    {        Write-Host 'Creating dir ... ' $folder;        mkdir $folder;    };    $subdir = @(\\"$folder/ntrials_$ntrials\\");    foreach ($s in $subdir)    {        if (-not (Test-Path -Path $s))        {            Write-Host 'Creating dir ... ' $s;            mkdir $s;        };        $subsubdir = @(\\"$s/stopfood\\");        foreach ($ss in $subsubdir)        {            if (-not (Test-Path -Path $ss))            {                Write-Host 'Creating dir ... ' $ss;                mkdir $ss;            };            $subsubsubdir = @(\\"$ss/Pinit025\\");            foreach ($sss in $subsubsubdir)            {                if (-not (Test-Path -Path $sss))                {                    Write-Host 'Creating dir ... ' $sss;                    mkdir $sss;                };            };        };    };};Move-Item \\"*ntrials_$ntrials*Pinit_0.25*cprob*stopfood*.mat\\" \\"cumul_prob/ntrials_$ntrials/stopfood/Pinit025\\";Move-Item \\"*ntrials_$ntrials*Pinit_0.25*cstep*stopfood*.mat\\" \\"cumul_step/ntrials_$ntrials/stopfood/Pinit025\\";Move-Item \\"*ntrials_$ntrials*Pinit_0.25*indept*stopfood*.mat\\" \\"independent/ntrials_$ntrials/stopfood/Pinit025\\"" """
    else:
        script_txt = """ntrials=%d;folders="cumul_prob cumul_step independent";for folder in $folders; do    if [ ! -d $folder ]; then        echo Creating dir ...  $folder;        mkdir $folder;    fi;    $subdir = "$folder/ntrials_$ntrials";    for s in $subdir; do        if [ ! -d $s ]; then            echo Creating dir ...  $s;            mkdir $s;        fi;        $subsubdir = "$s/stopfood";        for ss in $subsubdir; do            if [ ! -d $ss ]; then                echo Creating dir ...  $ss;                mkdir $ss;            fi            $subsubsubdir = "$ss/Pinit025";            for sss in $subsubsubdir; do                if [ ! -d $sss ]; then                    echo Creating dir ...  $sss;                    mkdir $sss;                fi;            done;        done;    done;done;mv "*ntrials_$ntrials*Pinit_0.25*cprob*stopfood*.mat" "cumul_prob/ntrials_$ntrials/stopfood/Pinit025";mv "*ntrials_$ntrials*Pinit_0.25*cstep*stopfood*.mat" "cumul_step/ntrials_$ntrials/stopfood/Pinit025";mv "*ntrials_$ntrials*Pinit_0.25*indept*stopfood*.mat" "independent/ntrials_$ntrials/stopfood/Pinit025" """
    return script_txt%n_trials


def load_selfintersection_file(filename):
    selfint_file_struct = misc.structtype(**scipy.io.loadmat(filename,squeeze_me=True))
    selfint_file_struct.selfint_st_rt = [ [ (m if type(m) is numpy.ndarray else numpy.array([m])) for m in all_mice ] for all_mice in selfint_file_struct.selfint_st_rt ]
    selfint_file_struct.selfint_st_ft = [ [ (m if type(m) is numpy.ndarray else numpy.array([m])) for m in all_mice ] for all_mice in selfint_file_struct.selfint_st_ft ]
    return selfint_file_struct

def load_step_probability_file(file_path,file_name_expr=None):
    is_valid_dir = lambda f: ( (type(f) is str) and os.path.isdir(f))
    is_dir_wildcard = lambda f: ( (type(f) is str) and ('*' in f))
    if (type(file_path) is list) or is_valid_dir(file_path) or is_dir_wildcard(file_path):
        file_name_expr = 'stepmat_*.mat' if file_name_expr is None else file_name_expr
        if (type(file_path) is list):
            f_list = file_path
        else:
            if is_dir_wildcard(file_path):
                f_list = glob.glob(file_path)
            else:
                f_list = glob.glob(os.path.join(file_path,file_name_expr))
        if len(f_list) == 0:
            raise ValueError('No file found that match the pattern ::: %s'%os.path.join(file_path,file_name_expr))
        f_contents = [ load_step_probability_file(f,file_name_expr=file_name_expr) for f in f_list ]
        if (type(f_contents) is list):
            if (type(f_contents[0]) is list):
                f_contents = [ f for sub_f_list in f_contents for f in sub_f_list if not (f is None) ]
            else:
                f_contents = [ f for f in f_contents if not (f is None) ]
        L = [ int(numpy.sqrt(f.P[0].shape[0])) for f in f_contents ]
        f_contents = [ f for _,f in sorted(zip(L, f_contents), key=lambda pair: pair[0]) ]
        return f_contents
    else:
        if os.path.exists(file_path):
            S = misc.structtype(**scipy.io.loadmat(file_path,squeeze_me=True))
            S.help = lambda: print(S.help_param_definition)
            if not S.IsField('L'):
                S.L = int(numpy.sqrt(S.P[0].shape[0]))
            return S
        else:
            raise ValueError('The specified file does not exist ::: %s'%file_path)
    #return structtype(**{ k:v for k,v in scipy.io.loadmat(file_path,squeeze_me=True).items() if k[:2] != '__' } )

def replace_tracks_trial_label(track_dir,replacement_dict,verbose=True):
    """
    replace the trial labels of all tracks in track_dir according to the old:new values in the replacement dict
    """
    file_list  = get_all_files(track_dir,file_name_expr='*')
    track_list = load_trial_file(file_list,file_name_expr='') # loads all tracks in the directory
    for k,v in replacement_dict.items():
        tracks,ind = find_tracks_by(track_list,lambda track: k in track.trial)
        n_start = len(find_tracks_by(track_list,lambda track: v in track.trial)[0])
        tracks = sort_track_list(tracks,'trial_id')
        for track in tracks:
            n_start += 1 # we use this index to avoid conflict with any existing file having a similar trial type
            old_track_filename = get_track_output_filename('',track,join_with_output_dir=False)
            track.trial = v + str(n_start)
            new_track_filename = os.path.join(track_dir,get_track_output_filename('',track,join_with_output_dir=False))+'.mat'
            if verbose:
                print('   *** relabelling trial of ...  %s'%old_track_filename)
            save_trial_file(new_track_filename,track,verbose=verbose,rename_if_exists=True,ignore_directory=True)
        for f in numpy.array(file_list,dtype=str)[ind]:
            if verbose:
                print('   ... removing ::: %s'%str(f))
            os.remove(f)

def find_tracks_by(track_list,check_func,return_index=False):
    """
    returns all tracks in which
    check_func(track) is True
    """
    ind = [ k for k,tr in enumerate(track_list) if check_func(tr) ]
    tra = [ tr for tr in track_list if check_func(tr) ]
    return tra,ind

def get_all_files(file_path,file_name_expr=None):
    if type(file_path) is str:
        if os.path.isfile(file_path):
            return [ file_path ]
        else:
            if (type(file_name_expr) is str):
                file_path = os.path.join(file_path,file_name_expr)
            return glob.glob(file_path)
    else:
        if ((type(file_path) is list) or (type(file_path) is tuple)):
            return list(misc.flatten_list([ get_all_files(ff,file_name_expr=file_name_expr) for ff in file_path ]))
        else:
            raise ValueError('invalid file_path')

def group_track_list(track_list,group_by='trial',get_key_group_func=None,sortinsideby_label=None,get_key_sortinsideby_func=None,sortgroups_label=None,get_key_sortgroups_func=None,return_group_keys=True):
    """
    this function groups track_list according to group_by field label, evaluating each track[group_by] according to get_key_group_func(track[group_by])

    each group is internally sorted by track[sortinsideby_label], being evaluated by get_key_sortby_func(track[sortinsideby_label])

    returns:
        grouped_tracks: a list of lists, such that
                grouped_tracks[j][i] -> track i in group j; each group j corresponds to each unique track_list[group_by] value returned by get_key_group_func(track_list[group_by])
                                        group_tracks[j] list is sorted  by the values returned by get_key_sortby_func(track[sortinsideby_label])
                                        group_tracks is sorted according to get_key_sortgroups_func(track[sortgroups_label])
    
    for example: group_by == 'trial' and get_key_group_func == plib.trial_to_number, sortinsideby_label == 'mouse', get_key_sortby_func == int
    means that
        grouped_tracks[j] -> all mice in trial j, sorted according to the int(mouse_number) value
    """
    #from collections import OrderedDict
    group_by                = group_by.lower()
    sortinsideby_label      = sortinsideby_label.lower() if misc.exists(sortinsideby_label)      else None
    sortgroups_label        = sortgroups_label.lower()   if misc.exists(sortgroups_label)        else None
    if misc.exists(sortgroups_label) and (not plib.get_track_file().IsField(sortgroups_label)):
        raise ValueError('group_track_list ::: sortgroups_label must be a field of track file')
    #assert (group_by in ['mouse','trial']),"group_by must be 'mouse' or 'trial'"
    if misc.is_list_of_list(track_list):
        track_list = list(misc.flatten_list(track_list, only_lists=True))
    if group_by == 'trial':
        key_label    = 'trial'              # outer key
        get_key      = plib.trial_to_number # outer key
        key_sort     = 'mouse_number'       # inner key
        get_key_sort = int                  # inner key
    elif group_by == 'mouse':
        key_label    = 'mouse_number'       # outer key
        get_key      = int                  # outer key
        key_sort     = 'trial'              # inner key
        get_key_sort = plib.trial_to_number # inner key
    else:
        if not plib.get_track_file().IsField(group_by):
            raise ValueError('group_track_list ::: group_by must be a field of track file')
        if misc.exists(sortinsideby_label) and (not plib.get_track_file().IsField(sortinsideby_label)):
            raise ValueError('group_track_list ::: sortinsideby_label must be a field of track file')
        key_label    = group_by                                                                                      # outer key
        get_key      = get_key_group_func        if misc.exists(get_key_group_func)        else int                  # outer key
        key_sort     = sortinsideby_label        if misc.exists(sortinsideby_label)        else 'trial'              # inner key
        get_key_sort = get_key_sortinsideby_func if misc.exists(get_key_sortinsideby_func) else plib.trial_to_number # inner key
    
    sortgroups_label        = sortgroups_label.lower()   if misc.exists(sortgroups_label)        else key_label      # outer sort key
    get_key_sortgroups_func = get_key_sortgroups_func    if misc.exists(get_key_sortgroups_func) else get_key        # outer sort key

    all_keys_full,_ = misc.unpack_list_of_tuples(sorted([ (get_key(f[key_label]),get_key_sortgroups_func(f[sortgroups_label])) for f in track_list ],key=lambda item: item[1]))
    all_keys        = misc.unique_ordered(all_keys_full) #numpy.unique([ get_key(f[key_label]) for f in track_list ])
    track_list      = [ sorted([ f for f in track_list if get_key(f[key_label])==k ],key=lambda tr: get_key_sort(tr[key_sort])) for k in all_keys ]
    #if group_by == 'trial':
    all_keys        = [ m[0][key_label] for m in track_list ] # getting the original value of each key
    if return_group_keys:
        return track_list,all_keys
    else:
        return track_list

def sort_track_list(track_list,sort_by='trial'):
    """
    sort_by -> 'trial'    : sort by trial number;
               'mouse'    : sort by mouse number;
               'trial_id' : sort by trial id (excel file index)
    """
    sort_by = sort_by.lower()
    assert (sort_by in ['mouse','trial','trial_id']),"sort_by must be 'mouse';'trial' or 'trial_id'"
    if sort_by == 'trial':
        get_value = lambda f: float(plib.trial_to_number(f.trial))
    else: # sort_by == 'mouse' or 'trial_id'
        field = 'mouse_number' if sort_by == 'mouse' else 'trial_id'
        get_value = lambda f: int(f[field])
    value      = [ get_value(f) for f in track_list ]
    track_list = [ f for _,f in sorted(zip(value, track_list), key=lambda pair: pair[0]) ]
    return track_list

def select_numeric_trials(tracks,trial0=1,trial1=None,get_trial_number_func=None):
    """
    select every track[i] from tracks list such that
    get_trial_number_func(track[i]) >= trial0 and get_trial_number_func(track[i]) <= trial1

    if the get_trial_number_func is not given, then searches recursively in the tracks list
    and returns only tracks that have trial greater than trial0 and less than trial1
    """
    if misc.exists(get_trial_number_func):
        select_func = lambda tr: (get_trial_number_func(tr) >= trial0) and (get_trial_number_func(tr) <= trial1)
        return misc.select_from_list(tracks,select_func)
    else:
        return _select_numeric_trials_recursive(tracks,trial0=trial0,trial1=trial1)

def _select_numeric_trials_recursive(tracks,trial0=1,trial1=None):
    """
    returns all numeric trials such that trial >= trial0 and trial <= trial1
    """
    trial_max = _get_max_trial(tracks)
    trial1 = trial1 if misc.exists(trial1) else trial_max
    if (trial0 == 1) and (trial1 == trial_max):
        return tracks
    if type(tracks) is misc.structtype:
        if tracks.trial.isdigit():
            trial = int(tracks.trial)
            if (trial >= trial0) and (trial <= trial1):
                return tracks
        return None
    if misc.is_list_of_structtype(tracks):
        return [ tr for tr in tracks if ((int(tr.trial)>=trial0) and (int(tr.trial)<=trial1)) ]
    else:
        track_filtered = [ _select_numeric_trials_recursive(tr,trial0=trial0,trial1=trial1) for tr in tracks ]
        return [ tr for tr in track_filtered if len(tr)>0 ]

def _get_max_trial(tracks):
    return numpy.max([ int(tr.trial) for tr in misc.flatten_list(tracks,only_lists=True) if tr.trial.isdigit() ])

def _fix_return_tuple(result):
    if len(result) == 1:
        return result[0]
    else:
        return result

def load_track_simple(file_path,file_name_expr=None,return_file_path=False,fix_nan=True):
    """
    load a trial file without checking for any condition and without applying any function to the data...
    just pure raw plain loads the file contents and returns it
    if you want to filter files in the loading process, use load_trial_file (old)
    """
    file_name_expr     = 'mpos_*.mat' if (file_name_expr is None) else file_name_expr
    is_valid_dir       = lambda f: ( (type(f) is str) and os.path.isdir(f))
    is_dir_wildcard    = lambda f: ( (type(f) is str) and ('*' in f))
    f_list             = file_path
    if is_valid_dir(f_list) or is_dir_wildcard(f_list):
        f_list = get_all_files(file_path,file_name_expr=file_name_expr)
    if (type(f_list) is list):
        result = [ load_track_simple(f,return_file_path=False,fix_nan=fix_nan) for f in f_list ]
    else:
        if fix_nan:
            fix_nan_func = tran.fill_trajectory_nan_gaps
        else:
            fix_nan_func = lambda x: x
        result = fix_nan_func(misc.trackfile(**scipy.io.loadmat(f_list,squeeze_me=True)))
    if return_file_path:
        result = (result,f_list)
    return result

def load_trial_file(file_path,file_name_expr=None,
                    load_only_training_sessions_relative_target=False,skip_15_relative_target=False,use_extra_trials_relative_target=False,
                    sort_by_trial=False,fix_nan=True,remove_after_food=False,align_to_top=False,
                    group_by='none',return_group_by_keys=False,
                    is_relative_target=False,
                    t0_frac=0.0,dt_frac=1.0,
                    keep_between_targets=False,max_trial_number=1e8,
                    time_delay_after_food=None):
    """
    loads experiment MAT files from file path according to the parameters
    
    file_name_expr                              -> only file names matching this pattern will be loaded; default is 'mpos_*.mat'
    load_only_training_sessions_relative_target -> if True, skips all rotated trials (RXXX_Y trials), loads only standard 1 to 17 or 1 to 14 trials (only in relative target experiments)
    skip_15_relative_target                     -> if True, skips trial 15 (since it was a probe trial without food)
    use_extra_trials_relative_target            -> if True, returns all 1 to 17 trials (15 is treated according to parameter above); otherwise returns from 1 to 14
    sort_by_trial                               -> if True, then the return list of files is sorted according to the trial number
    fix_nan                                     -> if True, then fixes the missing data in velocity and trajectory
    remove_after_food                           -> if True, then removes all the data points after the minimum distance between nose and food
    align_to_top                                -> if True, aligns all files such that the entrance is at the top of the screen
    group_by                                    -> 'none', 'mouse', 'trial';
                                                    if 'none', then returns a simple list with each loaded file as an entry
                                                    if 'mouse', list of list, first index is mouse, second index is trial
                                                    if 'trial', list of list, first index is trial, second index is mouse
    return_group_by_keys                        -> if True and group_by != 'none', then returns the key values
                                                   corresponding to the first index of the list of lists
    t0_frac,dt_frac                             -> track.time has T elements, so the analysis will be made from T0=floor(t0_frac*T) T0:min(T0+ceil(dt_frac*T),T)
    """
    if type(group_by) == type(None):
        group_by = 'none'
    assert (group_by.lower() in ['none','mouse','trial']),"group_by must be 'none','mouse' or 'trial'"
    is_relative_target = is_relative_target or (skip_15_relative_target or use_extra_trials_relative_target)
    is_valid_dir       = lambda f: ( (type(f) is str) and os.path.isdir(f))
    is_dir_wildcard    = lambda f: ( (type(f) is str) and ('*' in f))
    if (type(file_path) is list) or is_valid_dir(file_path) or is_dir_wildcard(file_path):
        
        file_name_expr = 'mpos_*.mat' if file_name_expr is None else file_name_expr
        f_list = get_all_files(file_path,file_name_expr=file_name_expr)
        if len(f_list) == 0:
            raise ValueError('No file found that matches the pattern ::: %s'%str(os.path.join(file_path,file_name_expr)))

        f_contents = [ load_trial_file(f,load_only_training_sessions_relative_target=load_only_training_sessions_relative_target,
                                         skip_15_relative_target=skip_15_relative_target,use_extra_trials_relative_target=use_extra_trials_relative_target,
                                         fix_nan=fix_nan,remove_after_food=remove_after_food,align_to_top=align_to_top,group_by=group_by,return_group_by_keys=return_group_by_keys,
                                         is_relative_target=is_relative_target,t0_frac=t0_frac,dt_frac=dt_frac,keep_between_targets=keep_between_targets,
                                         max_trial_number=max_trial_number,time_delay_after_food=time_delay_after_food) for f in f_list ]
        f_contents = [f for f in f_contents if not(type(f) is type(None))]

        if len(f_contents) == 0:
            if return_group_by_keys:
                return f_contents,[]
            return f_contents

        if group_by != 'none':
            f_contents,all_keys = group_track_list(f_contents,group_by)
            if return_group_by_keys:
                return f_contents,all_keys
            else:
                return f_contents
        else:
            return sort_track_list(f_contents,'trial' if sort_by_trial else 'mouse')
    else: # file_path is a single file name
        if os.path.exists(file_path):
            track_file,tr_file_name = _load_trial_file(file_path,_get_import_functions_for_trial_file(fix_nan,remove_after_food,keep_between_targets,align_to_top,t0_frac,dt_frac,time_delay_after_food))
            trial_n                 = int(track_file.trial) if ((not (type(track_file) is type(None))) and track_file.trial.isdigit()) else None
            if _check_load_trial_file_conditions(trial_n,max_trial_number,load_only_training_sessions_relative_target,is_relative_target,use_extra_trials_relative_target,skip_15_relative_target):
                if type(track_file) is type(None):
                    warnings.warn('file has no trajectory data! -- {} coming from {}'.format(file_path,tr_file_name))
                return track_file
            return None
        else:
            raise ValueError('The specified file does not exist ::: %s'%file_path)

def _check_load_trial_file_conditions(trial_n,max_trial_number,load_only_training_sessions_relative_target,is_relative_target,use_extra_trials_relative_target,skip_15_relative_target):
    """
    returns True only if we want to load a particular file
    """
    if load_only_training_sessions_relative_target:
        if trial_n: # this is a numbered trial
            if is_relative_target and (((not use_extra_trials_relative_target) and (trial_n > 14)) or (skip_15_relative_target and (trial_n == 15))):
                return False
            if (trial_n > max_trial_number):
                return False
        else:
            return False
    else:
        if trial_n and (trial_n > max_trial_number):
            return False
    return True

def _add_track_fields(tr,**params):
    tr.Set(**params)
    return tr

def _get_import_functions_for_trial_file(fix_nan,remove_after_food,keep_between_targets,align_to_top,t0_frac,dt_frac,time_delay_after_food):
    if fix_nan:
        fix_nan_func = tran.fill_trajectory_nan_gaps
    else:
        fix_nan_func = lambda x: x
    if keep_between_targets:
        if remove_after_food:
            remove_after_food = False
            warnings.warn('keep_between_targets takes precedence from remove_after_food; keeping only between target and target_alt trajectories',UserWarning)
        rem_path_func = lambda x: _add_track_fields(tran.keep_path_between_targets(x,time_delay_after_food=time_delay_after_food),remove_after_food=remove_after_food,keep_between_targets=keep_between_targets)
    else:
        if remove_after_food:
            rem_path_func = lambda x: _add_track_fields(tran.remove_path_after_food(x,force_main_target=True,time_delay_after_food=time_delay_after_food),remove_after_food=remove_after_food,keep_between_targets=keep_between_targets)
        else:
            rem_path_func = lambda x: _add_track_fields(x,remove_after_food=remove_after_food,keep_between_targets=keep_between_targets)
    align_vector = numpy.array( (0,1) )
    if align_to_top:
        align_func = lambda x: plib.rotate_trial_file(x,align_vector,return_only_track=True)
    else:
        align_func = lambda x: x
    return lambda x: tran.slice_track_by_time_frac(align_func(rem_path_func(fix_nan_func(x))),t0_frac=t0_frac,dt_frac=dt_frac,copy_track=False)

def _get_trial_from_filename(file_path):
    d,fn = os.path.split(file_path)
    m = regexp.search('_trial_(\d+)_',fn)
    return int(m.group(1)) if m else None

def _load_trial_file(file_path,process_func=None):
    if type(process_func) is type(None):
        process_func = lambda x:x
    ff = misc.trackfile(**scipy.io.loadmat(file_path,squeeze_me=True))
    if has_trajectory_data(ff):
        return process_func(ff),ff.file_name
    else:
        return None,ff.file_name

def has_trajectory_data(track):
    has_data = lambda r: numpy.any(numpy.logical_not(numpy.isnan(numpy.prod(r,axis=1) if r.ndim>1 else r)))
    return has_data(track.r_nose) and has_data(track.r_center) and has_data(track.r_tail) and has_data(track.velocity)

def save_trial_file_txt(base_dir,track_mat,file_type='csv'):
    if not (file_type in ['csv','dat','txt']):
        raise ValueError('Invalid file type')
    out_fname = get_track_output_filename(base_dir,track_mat,ext='.'+file_type)
    if file_type == 'csv':
        col_delimiter = ','
    else:
        col_delimiter = '\t'
    create_output_dir(get_track_output_dir(base_dir,track_mat))
    print('     ... saving     %s'%out_fname)
    h = { k:v for k,v in track_mat.items() if not k in ['r_arena_holes','r_center','r_nose','r_tail','direction','velocity','time','__header__','__version__','__globals__']  }
    txt = functools.reduce(lambda a,b: a+b,[k+' = '+str(v)+'\n' for k,v in h.items()])
    fileHeader  = "****** Parameters:\n"
    fileHeader += txt
    fileHeader += "****** Data columns:\n"
    fileHeader += "time{0}r_center_x{0}r_center_y{0}r_nose_x{0}r_nose_y{0}r_tail_x{0}r_tail_y{0}velocity{0}direction".format(col_delimiter)
    try:
        data = numpy.column_stack((track_mat.time,track_mat.r_center,track_mat.r_nose,track_mat.r_tail,track_mat.velocity,track_mat.direction))
    except ValueError as verr:
        data = numpy.array([])
    numpy.savetxt(out_fname, data, fmt='%16.8E', delimiter=col_delimiter, header=fileHeader)

def save_track_simple(track_list,filename_list=None,base_dir_or_fname=None):
    """
    saves each track in the track_list with the corresponding filename in filename_list
    """
    is_valid_iterable   = lambda x: (type(x) is list) or (type(x) is tuple)
    is_valid_track_list = lambda x: is_valid_iterable(x) and misc.is_trackfile(x[0])
    base_dir_or_fname   = base_dir_or_fname if misc.exists(base_dir_or_fname) else '.'
    track_list          = track_list        if is_valid_track_list(track_list)  else [ track_list ]
    if not misc.exists(filename_list):
        filename_list = [ get_track_output_filename(base_dir_or_fname, track, join_with_output_dir=True) for track in track_list ]
    filename_list     = filename_list if is_valid_iterable(filename_list) else [ filename_list ]
    for out_fname,track in zip(filename_list,track_list):
        scipy.io.savemat(out_fname,dict(**track),long_field_names=True,do_compression=True)
    return filename_list

def save_trial_file(base_dir_or_fname,track,verbose=True,rename_if_exists=False,ignore_directory=False,save=True):
    if base_dir_or_fname.lower()[-4:] == '.mat':
        out_fname = base_dir_or_fname
    else:
        out_fname = get_track_output_filename(base_dir_or_fname, track, join_with_output_dir=not ignore_directory)
        if not ignore_directory:
            create_output_dir(get_track_output_dir(base_dir_or_fname,track))
    if rename_if_exists:
        if os.path.exists(out_fname):
            fn = check_and_get_filename(out_fname)
            if verbose:
                print('     ... renaming     %s   ...   %s'%(out_fname,fn))
            os.path.rename(out_fname,fn)
    if verbose:
        print('     ... saving     %s'%out_fname)
    if save:
        scipy.io.savemat(out_fname,dict(**track),long_field_names=True,do_compression=True)
    return out_fname

def clear_directory(path,verbose=False):
    if type(path) is list:
        for p in path:
            clear_directory(p,verbose=verbose)
    else:
        if os.path.isdir(path):
            objs = glob.glob(os.path.join(path,'*'))
            for obj in objs:
                clear_directory(obj)
            try:
                if verbose:
                    print('- Removing dir ::: %s'%path)
                os.rmdir(path)
            except:
                print('- Cannot remove dir ::: %s'%path)
        else:
            try:
                if verbose:
                    print('- Removing file ::: %s'%path)
                os.remove(path)
            except:
                print('- Cannot remove file ::: %s'%path)

def create_output_dir(dirname):
    if not os.path.exists(dirname):
        #print('- Directory %s already exists, deleting all files in it' % dirname)
        #files = glob.glob(os.path.join(dirname,'*'))
        #for f in files:
        #    os.remove(f)
        #else:
        print('- Creating directory ::: %s'%dirname)
        os.makedirs(dirname,exist_ok=True)

def get_filename_with_parentdir(path):
    return os.path.join(os.path.split(os.path.split(path)[0])[1],get_filename(path))

def get_filename(path):
    return os.path.split(path)[1]

def get_dirname(path):
    return os.path.split(path)[0]

def list_of_arr_to_arr_of_obj(X):
    n = len(X)
    Y = numpy.empty((n,),dtype=object)
    for i,x in enumerate(X):
        Y[i] = x
    return Y

def struct_array_for_scipy(field_names,*fields_data):
    """
    returns a data structure which savemat in scipy.io interprets as a MATLAB struct array
    the order of field_names must match the order in which the remaining arguments are passed to this function
    such that
    s(j).(field_names(i)) == fields_data[i][j], identified by field_names[i]

    field_names ->  comma-separated string listing the field names;
                        'field1,field2,...' -> field_names(1) == 'field1', etc...
    fields_data ->  each extra argument entry is a list with the data for each field of the struct
                        fields_data[i][j] :: data for field i in the element j of the struct array: s(j).(field_names(i))
    """
    fn_list = field_names.split(',')
    assert len(fn_list) == len(fields_data),'you must give one field name for each field data'
    return numpy.core.records.fromarrays([f for f in fields_data],names=fn_list,formats=[object]*len(fn_list))

def get_files_GUI(message='Select file...',path='',wildcard='*.npz',multiple=True,max_num_files=3):
    return None

def get_track_output_dir(base_dir,file_header):
    return os.path.join(base_dir,'mouse_%s'%file_header.mouse_number)

def get_track_output_filename(base_dir,file_header,ext='.mat',join_with_output_dir=True):
    fname = 'mpos_%s_trial_%s_startloc_%s_day_%s' % (file_header.exper_date,file_header.trial,file_header.start_location,file_header.day)
    if join_with_output_dir:
        return os.path.join(get_track_output_dir(base_dir,file_header),fname) + ext
    else:
        return fname + ext

def check_and_get_filename(fn):
    n = 0
    if os.path.exists(fn):
        old_fn,ext = os.path.splitext(fn)
        while True:
            n += 1
            new_fn = old_fn + '_' + str(n) + ext
            if not os.path.exists(new_fn):
                break
        return new_fn
    else:
        return fn

