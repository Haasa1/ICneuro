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

#from re import L
import os # path and file handling
import numpy
import pandas # load excel and data files
import networkx
import warnings
import functools
import itertools
import scipy.io
import scipy.stats
import scipy.sparse
import scipy.spatial
import scipy.optimize
import scipy.interpolate
import modules.io as io
import modules.traj_analysis as tran
import modules.helper_func_class as misc
import modules.process_mouse_trials_lib as plib

#import warnings

#warnings.filterwarnings('error')

def get_step_probability_file_struct(param_struct,P,N,G,stage_trials,n_trials,r_target,r_mouse,t_to_food,mouse_id,trial_id,P_specific,mouse_id_spec,r_target_trial,r_target_previous_trial,r_target_alt_trial,r_target_rev_trial,r_target_revalt_trial,arena_geometry):
    """
    returns a misc.structtype data structure with the simulation data resulting from calc_probability_step function
    """
    param_help = calc_step_probability(get_help=True)
    param_struct.KeepFields(*get_calc_step_probability_param_struct().keys())
    io.save_step_probability_file_struct
    return misc.structtype(**param_struct,
                             P                       = io.list_of_arr_to_arr_of_obj(P             )          ,
                             N                       = io.list_of_arr_to_arr_of_obj(N             )          ,
                             G                       = io.list_of_arr_to_arr_of_obj(G             )          ,
                             P_specific              = io.list_of_arr_to_arr_of_obj(P_specific    )          ,
                             mouse_id_spec           = io.list_of_arr_to_arr_of_obj(mouse_id_spec )          ,
                             stage_trials            = io.list_of_arr_to_arr_of_obj(stage_trials  )          ,
                             r_mouse                 = io.list_of_arr_to_arr_of_obj(r_mouse       )          ,
                             t_to_food               = io.list_of_arr_to_arr_of_obj(t_to_food     )          ,
                             mouse_id                = io.list_of_arr_to_arr_of_obj(mouse_id      )          ,
                             trial_id                = io.list_of_arr_to_arr_of_obj(trial_id      )          ,
                             n_trials                = n_trials                                              ,
                             r_target                = r_target                                              ,
                             r_target_trial          = io.list_of_arr_to_arr_of_obj(r_target_trial         ) ,
                             r_target_alt_trial      = io.list_of_arr_to_arr_of_obj(r_target_alt_trial     ) ,
                             r_target_rev_trial      = io.list_of_arr_to_arr_of_obj(r_target_rev_trial     ) ,
                             r_target_revalt_trial   = io.list_of_arr_to_arr_of_obj(r_target_revalt_trial  ) ,
                             r_target_previous_trial = io.list_of_arr_to_arr_of_obj(r_target_previous_trial) ,
                             L                       = int(numpy.sqrt(P[0].shape[0]))                        ,
                             arena_geometry          = arena_geometry                                        ,
                             help_param_definition   = param_help                                            )

def get_step_prob_input_param_config_list(**params):
    """
    this function gets the same arguments as calc_step_probability
    where each argument now receives a list of possible values

    returns a list of configurations that will be used to calculate the step probability matrices via calc_step_probability

    warning: this can take up a lot of memory
    """
    param_names = list(params.keys()) #['mouse_part','n_stages','L_lattice','prob_calc','start_from_zero','use_extra_trials','stop_at_food']
    param_values = itertools.product(*params.values()) #(mouse_part,n_stages,L,prob_calc,start_from_zero,use_extra_trials,stop_at_food)
    config = [ get_calc_step_probability_param_struct(**{ n:v for n,v in zip(param_names,val_combination) }) for val_combination in param_values ]
    return config

def get_calc_step_probability_param_struct(mouse_part=None,n_stages=None,L_lattice=None,prob_calc=None,start_from_zero=None,use_extra_trials=None,stop_at_food=None,use_latest_target=False,align_method='entrance',use_reverse_target=False):
    """
    returns a misc.structtype with the following fields:
        mouse_part         -> 'nose', 'center', 'tail' (the tracked mouse part to use in this calculation)
        n_stages           -> number of stages to divide the training data
                                 there's 14 training sessions (or 16, if use_extra_trials is set),
                                 which will be split into n_stages training sessions to enhance statistics...
                                 for instance, if n_stages = 4,
                                 then the training sessions 1..14 will be split into
                                 4 probability matrices, one for each learning stage:
                                 [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14]
                                    P1,        P2,        P3,         P4
        L_lattice          -> number of squares in each row and each column of the overlaid square grid
        start_from_zero    -> this only works if prob_calc == 'cumulative_prob'
                              if True, then the initial probability (before any walk) is assumed P0 = 0 for all steps
                              otherwise, we use P0 = 1/4 for all steps (the boundary steps will be normalized accordingly)
        prob_calc          -> ProbabilityType enum object to specify one of the methods below:
                             'cumulative_step':
                                 the probability of stage k is calculated by summing all the step for a given grid crossing up to stage k,
                                 and then dividing by the total amount of grid crossings in all the stages up to k:
                                 Pk_ij = [ N(k-1)_ij + Nk_ij ] / sum(N(k-1) + Nk), with N0 = 0 for all steps
                             'cumultive_prob':
                                 the probability of stage k is calculated by the union of
                                 taking that step in stage k-1 and taking the same step at stage k alone
                                 (and these two events are not independent):
                                 Pk_ij = [ P(k-1)_ij + pk_ij - P(k-1)_ij*pk_ij ], with pk_ij=Nk_ij/sum(Nk) being the step probability of stage k alone
                             'independent':
                                 each learning stage is calculated independently of the previous stages
                                 yielding pk_ij for each stage k
                                 then, the Pk_ij = P0_ij + pk_ij - P0_ij*pk_ij to take into account the probability to walk on the other grid squares
        use_extra_trials   -> if True, we also use trials 16 and 17 as training sessions (skip trial 15 because it is right after the probe session)
        stop_at_food       -> stop counting steps when the mouse reaches the food for the 1st time
        align_method       -> defines how tracks are aligned; must be one of 'none','entrance','target','target_trial_consistent'
        use_latest_target  -> if stop_at_food, stops at the latest target between track.r_target and track.r_target_alt (useful for 2 targets probe trial)
        use_reverse_target -> if stop_at_food, then stops with min distance to othe reverse target location
    """
    align_method = align_method.lower()
    assert align_method in ['none','entrance','target','target_trial_consistent'],"align_method must be one of 'none','entrance','target','target_trial_consistent'"
    return misc.structtype(mouse_part         =  mouse_part        ,
                           n_stages           =  n_stages          ,
                           L_lattice          =  L_lattice         ,
                           prob_calc          =  prob_calc         ,
                           start_from_zero    =  start_from_zero   ,
                           use_extra_trials   =  use_extra_trials  ,
                           stop_at_food       =  stop_at_food      ,
                           align_method       =  align_method      ,
                           use_latest_target  =  use_latest_target ,
                           use_reverse_target =  use_reverse_target)

def calc_step_probability(mouse_dir=None,param_struct=None,get_help=False,tracks=None,return_as_file_struct=False):
    param_help = """
    given a mouse directory that has files preprocessed by this lib, this function
    calculates the step probability matrices
    (i.e., number of times the mouse steps from site j to site i, yielding P_ij)
    based on the user-defined parameters

    mouse_dir        -> directory containing the subdirectories mouse_m, where m is the mouse index
    tracks           -> if track is set, ignores mouse_dir and uses only the tracks in track

    param_struct ( returned by get_step_prob_input_param_config_list );
        is a misc.structtype with the following fields:
        mouse_part         -> 'nose', 'center', 'tail' (the tracked mouse part to use in this calculation)
        n_stages           -> number of stages to divide the training data
                                 there's 14 training sessions (or 16, if use_extra_trials is set),
                                 which will be split into n_stages training sessions to enhance statistics...
                                 for instance, if n_stages = 4,
                                 then the training sessions 1..14 will be split into
                                 4 probability matrices, one for each learning stage:
                                 [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14]
                                    P1,        P2,        P3,         P4
        L_lattice          -> number of squares in each row and each column of the overlaid square grid
        start_from_zero    -> this only works if prob_calc == 'cumulative_prob'
                              if True, then the initial probability (before any walk) is assumed P0 = 0 for all steps
                              otherwise, we use P0 = 1/4 for all steps (the boundary steps will be normalized accordingly)
        prob_calc          -> ProbabilityType enum object to specify one of the methods below:
                             'cumulative_step':
                                 the probability of stage k is calculated by summing all the step for a given grid crossing up to stage k,
                                 and then dividing by the total amount of grid crossings in all the stages up to k:
                                 Pk_ij = [ N(k-1)_ij + Nk_ij ] / sum(N(k-1) + Nk), with N0 = 0 for all steps
                             'cumultive_prob':
                                 the probability of stage k is calculated by the union of
                                 taking that step in stage k-1 and taking the same step at stage k alone
                                 (and these two events are not independent):
                                 Pk_ij = [ P(k-1)_ij + pk_ij - P(k-1)_ij*pk_ij ], with pk_ij=Nk_ij/sum(Nk) being the step probability of stage k alone
                             'independent':
                                 each learning stage is calculated independently of the previous stages
                                 yielding pk_ij for each stage k
                                 then, the Pk_ij = P0_ij + pk_ij - P0_ij*pk_ij to take into account the probability to walk on the other grid squares
        use_extra_trials   -> if True, we also use trials 16 and 17 as training sessions (skip trial 15 because it is right after the probe session)
        stop_at_food       -> stop counting steps when the mouse reaches the food for the 1st time
        align_method       -> defines how tracks are aligned; must be one of 'none','entrance','target','target_trial_consistent'
        use_latest_target  -> if stop_at_food, stops at the latest target between track.r_target and track.r_target_alt (useful for 2 targets probe trial)
        use_reverse_target -> if stop_at_food, then stops with min distance to othe reverse target location

    returns:
    *** all positions in lattice coords are 0-based indices
    *** the trial as an index is the trial used to calculate the probability that correspond to a given original run with id given by trial_id[k][trial]
        P[k][i,j]                      -> probability of stepping from j to i in learning stage k (i and j are the linearized indices of the lattice, i,j = y + x*L, row y col x)
        P_specific[k][m][i,j]          -> probability of stepping from j to i for mouse m in learning stage k (i and j are the linearized indices of the lattice, i,j = y + x*L, row y col x)
        N[k][trial][i,j]               -> number of steps from j to i separated by trial and stage (i and j are the linearized indices of the lattice, i,j = y + x*L, row y col x)
        G[k][trial][y,x]               -> total number of visits to site in row y and column x of sqr lattice (trials are separated on each stage)
        stage_trials[k]                -> trial # of the trials that pertain to stage k
        n_trials[k]                    -> number of files that were processed to yield stage k
        r_target                       -> (x,y) position (in lattice coordinate indices, i.e., x=col,y=row) of the target
        r_mouse[k][trial][t,:]         -> T-by-(x,y) mouse position (sqr lattice coord, 0-based) for each time step
        t_to_food[k][trial]            -> time the mouse takes to reach the food in each trial (in time steps)
        mouse_id[k][trial]             -> number of the mouse for stage k and trial
        mouse_id_specific[k][trial][m] -> number of the mouse m for stage k and trial corresponding to the mouse m in P_specific
        trial_id[k][trial]             -> original trial index for stage k and probability trial
        P0                             -> initial probability matrix (before learning stage 1)
    """
    if get_help:
        return param_help
    mouse_part         =  param_struct.mouse_part.lower()
    n_stages           =  param_struct.n_stages
    L_lattice          =  param_struct.L_lattice
    prob_calc          =  param_struct.prob_calc
    start_from_zero    =  param_struct.start_from_zero
    use_extra_trials   =  param_struct.use_extra_trials
    stop_at_food       =  param_struct.stop_at_food
    align_method       =  param_struct.align_method.lower()
    use_latest_target  =  param_struct.use_latest_target
    use_reverse_target =  param_struct.use_reverse_target
    if L_lattice%2 == 0:
        raise ValueError('L_lattice must be odd')
    #if not (prob_calc.lower() in ['cumulative_step','cumulative_prob','independent']):
    #    raise ValueError('prob_calc must be either cumulative_step or cumulative_prob')
    if not (mouse_part in ['center','nose','tail']):
        raise ValueError('mouse_part must be either center, nose, or tail')
    #if (n_stages < 1) or ((use_extra_trials and (n_stages > 16)) or (not use_extra_trials and (n_stages > 14)) ):
    #    raise ValueError('n_stages must be greater than 1 and less than 14 or 16 depending on use_extra_trials')

    mouse_part = 'r_' + mouse_part

    if mouse_dir:
        if not mouse_dir.endswith('mouse_*'):
            mouse_dir = os.path.join(mouse_dir,'mouse_*')

    # getting all input files
    if type(tracks) is type(None):
        tracks = io.load_trial_file(mouse_dir,load_only_training_sessions_relative_target=True,skip_15_relative_target=True,use_extra_trials_relative_target=use_extra_trials,fix_nan=False)
    trial_n = numpy.asarray([ plib.trial_to_number(tr.trial) for tr in tracks ])
    
    # now we need to align the entrance of all the files with the vector (0,1) (top of the screen)
    if align_method == 'entrance': # 'entrance' or 'target'
        tracks = plib.rotate_trial_file(tracks,numpy.array( (0,1) ),return_only_track=True)
    elif align_method == 'target': # 'entrance' or 'target'
        tracks = plib.align_targets(tracks,numpy.array( (1,0) ))
    elif align_method == 'target_trial_consistent':
        tracks = sum(plib.align_targets_group_by_start_quadrant(io.group_track_list(tracks,group_by='trial')[0], numpy.array((0,1))),[])
    else:
        if align_method != 'none':
            raise ValueError("param_struct.align_method can only be 'none','entrance','target','target_trial_consistent'")

    # setting some default parameters
    n_mice             = len(pandas.unique([ int(tr.mouse_number) for tr in tracks ]))         # couting total number of mice
    T_arena_to_lattice = [ get_arena_to_lattice_transform(L_lattice,r_center=tr.r_arena_center) for tr in tracks ]  # getting the transform used to convert arena coordinates to lattice coordinates

    # setting the stage-0 probabilities
    ij_adjmat = get_nonzero_rowcol_circsqrlatt(L_lattice)
    nk_prev = scipy.sparse.csc_matrix( ( numpy.zeros( len(ij_adjmat[0]) ), ij_adjmat ), shape=(L_lattice**2,L_lattice**2) )  #numpy.zeros((L_lattice**2,L_lattice**2))
    nk_prev_spec = [ nk_prev.copy() for _ in range(n_mice) ]
    P0 = scipy.sparse.csc_matrix( ( numpy.zeros( len(ij_adjmat[0]) ), ij_adjmat ), shape=(L_lattice**2,L_lattice**2) ) #numpy.zeros((L_lattice**2,L_lattice**2))
    if (prob_calc == misc.ProbabilityType.cumulative_prob) and not start_from_zero:
        P0 = get_initial_step_probability(L_lattice)
    if (prob_calc == misc.ProbabilityType.cumulative_prob):
        p0_spec = [ P0.copy() for _ in range(n_mice) ]

    # creating the learning stages from the input files
    stage_trials            = numpy.array_split(numpy.unique(trial_n),n_stages)
    n_trials                = [None for _ in stage_trials]
    r_mouse                 = [None for _ in stage_trials]
    t_to_food               = [None for _ in stage_trials]
    mouse_id                = [None for _ in stage_trials]
    trial_id                = [None for _ in stage_trials]
    G                       = [None for _ in stage_trials]
    N                       = [None for _ in stage_trials]
    P                       = [None for _ in stage_trials] #[P0] + [None for s in stage_trials]
    P_specific              = [None for _ in stage_trials]
    mouse_id_spec           = [None for _ in stage_trials]
    r_target_trial          = [None for _ in stage_trials]
    r_target_previous_trial = [None for _ in stage_trials]
    r_target_alt_trial      = [None for _ in stage_trials]
    r_target_rev_trial      = [None for _ in stage_trials]
    r_target_revalt_trial   = [None for _ in stage_trials]
    arena_geometry          = [None for _ in stage_trials]
    has_more_than_one_tgt_f = lambda r_tgt: numpy.unique(numpy.array(r_tgt),axis=0).shape[0] > 1
    for k,stage in enumerate(stage_trials):
        # first, we sum all the the number of steps and grid matrices for this stage
        # count_number_of_steps_in_lattice -> returns a tuple of N,G
        # N is a scipy sparse matrix, G is a numpy array matrix
        
        # the variable steps_in_lattice has the following structure:
        # steps_in_lattice[trial][0] -> step matrix (number of steps between adjacent sites) for trial #
        # steps_in_lattice[trial][1] -> grid matrix (number of visits to each site) for trial #
        # steps_in_lattice[trial][2] -> mouse position in lattice coord for trial #
        # steps_in_lattice[trial][3] -> number of time steps to reach the food
        # steps_in_lattice[trial][4] -> trial number of this trial
        # steps_in_lattice[trial][5] -> mouse number of this trial
        steps_in_lattice           = []
        r_target_trial[k]          = []
        r_target_alt_trial[k]      = []
        r_target_rev_trial[k]      = []
        r_target_revalt_trial[k]   = []
        r_target_previous_trial[k] = []
        for tr,T in zip(tracks,T_arena_to_lattice):
            if plib.trial_to_number(tr.trial) in stage:
                r_target = tr.r_target
                if use_latest_target:
                    r_target = _pick_latest_target(tr)
                elif use_reverse_target:
                    r_target = tr.r_target_reverse
                #  steps_in_lattice[k][0] -> scipy.sparse.csc_matrix(N) (adjacency step matrix)
                #  steps_in_lattice[k][1] -> G (lattice grid)
                #  steps_in_lattice[k][2] -> r_latt (position in the lattice)
                #  steps_in_lattice[k][3] -> t_to_food (r_latt.shape[0] -> time to find food in steps)
                #  steps_in_lattice[k][4] -> arena_geometry -> misc.structtype with arena parameters to check lattice and arena alignment
                #  steps_in_lattice[k][5] -> trial id
                #  steps_in_lattice[k][6] -> mouse number
                steps_in_lattice.append(count_number_of_steps_in_lattice(tr.time,tr[mouse_part],L_lattice,r_center=tr.r_arena_center,r_target=r_target,stop_at_food=stop_at_food,return_arena_lattice_geometry=True)     +   (  plib.trial_to_number(tr.trial)  ,int(tr.mouse_number)  )      )
                r_target_trial[k].append(       apply_arena_to_lattice_transform( T,     r_target             )  ) #tr.r_target)
                r_target_alt_trial[k].append(   apply_arena_to_lattice_transform( T,  tr.r_target_alt         )  )
                r_target_rev_trial[k].append(   apply_arena_to_lattice_transform( T,  tr.r_target_reverse     )  )
                r_target_revalt_trial[k].append(apply_arena_to_lattice_transform( T,  tr.r_target_alt_reverse )  )

        # storing the target for each trial
        if has_more_than_one_tgt_f(r_target_trial[k]):
            warnings.warn('  ****  MORE THAN 1 TARGET FOUND ****')
        if has_more_than_one_tgt_f(r_target_alt_trial[k]):
            warnings.warn('  ****  MORE THAN 1 TARGET ALT FOUND ****')
        if has_more_than_one_tgt_f(r_target_rev_trial[k]):
            warnings.warn('  ****  MORE THAN 1 TARGET REVERSE FOUND ****')
        if has_more_than_one_tgt_f(r_target_revalt_trial[k]):
            warnings.warn('  ****  MORE THAN 1 TARGET ALT REVERSE FOUND ****')
        r_target_trial[k]          = scipy.stats.mode(numpy.floor( numpy.array(        r_target_trial[k])   ) , axis=0, nan_policy='omit').mode.flatten()     # numpy.floor(numpy.mean( numpy.array(        r_target_trial[k]) ,axis=0)) 
        r_target_alt_trial[k]      = scipy.stats.mode(numpy.floor( numpy.array(    r_target_alt_trial[k])   ) , axis=0, nan_policy='omit').mode.flatten()     # numpy.floor(numpy.mean( numpy.array(    r_target_alt_trial[k]) ,axis=0)) 
        r_target_rev_trial[k]      = scipy.stats.mode(numpy.floor( numpy.array(    r_target_rev_trial[k])   ) , axis=0, nan_policy='omit').mode.flatten()     # numpy.floor(numpy.mean( numpy.array(    r_target_rev_trial[k]) ,axis=0)) 
        r_target_revalt_trial[k]   = scipy.stats.mode(numpy.floor( numpy.array( r_target_revalt_trial[k])   ) , axis=0, nan_policy='omit').mode.flatten()     # numpy.floor(numpy.mean( numpy.array( r_target_revalt_trial[k]) ,axis=0)) 
        r_target_previous_trial[k] = r_target_trial[k-1] if k > 0 else numpy.nan*numpy.ones(2)

        # old code of this function (working)
        # the new code is way slower, but is way clearer too
        #result = functools.reduce(lambda A,B: (A[0]+B[0],[A[1]]+[B[1]],[A[2]]+[B[2]] ) , steps_in_lattice)
        #N[k+1],G[k],r_mouse[k] = result #
        #G[k+1] = G[k] + G[k+1]

        arena_geometry[k] = [ s[4] for s in steps_in_lattice ]

        # number of trials for this stage
        n_trials[k] = len(steps_in_lattice)

        # storing mouse and trial id for this stage
        trial_id[k]      = numpy.asarray([ s[5] for s in steps_in_lattice ])
        mouse_id[k]      = numpy.asarray([ s[6] for s in steps_in_lattice ])
        mouse_id_spec[k] = misc.unique_stable(mouse_id[k]) #numpy.unique(mouse_id[k])

        # total number of time steps the mouse takes to reach the food
        t_to_food[k] = numpy.asarray([ s[3] for s in steps_in_lattice ])

        # number of visits to position (x,y) of the lattice at trial # and stage k
        # G[k][trial][y,x]
        # I'm not going to accumulate the grid matrix, as I did before
        G[k] =  io.list_of_arr_to_arr_of_obj([ s[1] for s in steps_in_lattice ])

        # number of steps between adjacent sites for each trial in stage k
        # N[k][trial][i,j] -> number of steps from square j to square i (i,j are linearized grid indices, i,j = y_i,j + x_i,j * L)
        # I'm not going to accumulate the steps to to save
        N[k] = numpy.asarray([ s[0] for s in steps_in_lattice ],dtype=object)
        nk = functools.reduce(lambda N1,N2: N1+N2,N[k]) # accumulating the step matrices of this learning stage for probability calculation
        # one nk for each mouse, hence "specific"
        nk_specific = [ functools.reduce(lambda N1,N2: N1+N2,[ nn for nn,m_id in zip(N[k],mouse_id[k]) if m_id == mm ]) for mm in mouse_id_spec[k] ]

        # mouse position in lattice coord
        # r_mouse[k][trial][t,:] -> (x,y) coordinates (lattice frame) at time t for trial # and stage k
        r_mouse[k] = numpy.asarray([ s[2] for s in steps_in_lattice ],dtype=object)

        # the stage index (k) of N,G,P is displaced by 1 compared to stage_trials
        # because there is one extra element in each of these 3 arrays
        if (prob_calc == misc.ProbabilityType.cumulative_prob):
            P[k]          = calc_step_probability_matrix_cumuprob(P0,nk)
            P_specific[k] = numpy.asarray([ calc_step_probability_matrix_cumuprob(pp0,nn) for pp0,nn in zip(p0_spec,nk_specific) ],dtype=object)
            P0            = P[k].copy() # the next stage will be accumulated with this stage
            p0_spec       = [ pp.copy() for pp in P_specific[k] ]
        elif (prob_calc == misc.ProbabilityType.cumulative_step):
            nk            = nk + nk_prev # nk_prev contains the accumulated step numbers up to stage k-1
            nk_specific   = [ nn + nn_prev for nn,nn_prev in zip(nk_specific,nk_prev_spec) ]
            P[k]          = calc_step_probability_matrix_stage_k(nk) # this stage is based on the accumulated steps of all previous stage, including this stage too
            P_specific[k] = numpy.asarray([ calc_step_probability_matrix_stage_k(nn) for nn in nk_specific ],dtype=object)
            nk_prev       = nk.copy()
            nk_prev_spec  = [ nn.copy() for nn in nk_specific ]
        else: #if (prob_calc == ProbabilityType.independent)
            P[k]          = calc_step_probability_matrix_cumuprob(P0,nk)
            P_specific[k] = numpy.asarray([ calc_step_probability_matrix_cumuprob(P0,nn) for nn in nk_specific ],dtype=object)
    
    # getting the target coordinates in lattice frame of reference
    # averaged over all trials of this experiment
    # (only meaningful if the target is fixed over trials with all the tracks aligned --
    # this is not the case for random entrance experiments)
    r_target = scipy.stats.mode(numpy.floor( numpy.stack( [   apply_arena_to_lattice_transform(T,tr.r_target ) for tr,T in zip(tracks,T_arena_to_lattice)  ] ) ) , axis=0, nan_policy='omit').mode.flatten()

    if return_as_file_struct:
        return get_step_probability_file_struct(param_struct,P,N,G,stage_trials,n_trials,r_target,r_mouse,t_to_food,mouse_id,trial_id,P_specific,mouse_id_spec,r_target_trial,r_target_previous_trial,r_target_alt_trial,r_target_rev_trial,r_target_revalt_trial,arena_geometry)
    else:
        return P,N,G,stage_trials,n_trials,r_target,r_mouse,t_to_food,mouse_id,trial_id,P_specific,mouse_id_spec,r_target_trial,r_target_previous_trial,r_target_alt_trial,r_target_rev_trial,r_target_revalt_trial,arena_geometry

def calc_step_probability_matrix_cumuprob(pkm1,nk):
    """
    calculates the step probability matrix for learning stage k
    cumulative probability

    pkm1 -> probability of taking a step in stage k-1 (or in the ground state)
    nk   -> number of steps only in stage k
    """
    pk = calc_step_probability_matrix_stage_k(nk)
    return normalize_cols(pkm1 + pk - pkm1.multiply(pk)) # the probability of taking a step in stage k
                                                         # is the probability of taking a step in all stages up to k
                                                         # + probability of taking a step in stage k alone

def calc_step_probability_matrix_stage_k(nk):
    """
    calculates the step probability matrix for learning stage k
    nk -> number of steps up in stage k
    """
    # divide the total number of steps up to this stage between every two adjacent grids and the total number of steps
    return normalize_cols(nk.multiply(1.0 / nk.sum()))

def normalize_cols(P):
    """
    normalize each col of P
    P is a step probability matrix, so each row must be normalized, such that the probability of stepping out of a grid
    is always 100%
    """
    s = P.sum(0) # sum over rows for each column
    s[s==0.0]=1.0 # replacing the sums that equal zero by 1 to avoid division by zero warning
    return P.multiply(1.0/s) # sum(0) sums over rows for each col, then multiplies by P (each column of 1/sum by each col in P)

def get_initial_step_probability(L,asdense=False):
    G,_,_,_ = get_circular_grid_graph(L)
    G = G.to_directed()
    for j in G.nodes:
        n_neigh = sum(1.0 for _ in G.neighbors(j))
        for i in G.neighbors(j):
            G[i][j]['weight'] = 1.0 / n_neigh # step from j to i happens with probability 1/n where n is the number of adjacent squares
    if asdense:
        P0 = numpy.asarray(networkx.adjacency_matrix(G,weight='weight').todense())
    else:
        P0 = networkx.adjacency_matrix(G,weight='weight')
    return P0.tocsc()

def get_nonzero_rowcol_circsqrlatt(L):
    G,_,_,_ = get_circular_grid_graph(L)
    return networkx.adjacency_matrix(G.to_directed()).nonzero()

def _pick_latest_target(track,hole_horizon=None):
    if numpy.any(numpy.isnan(track.r_target_alt)):
        return track.r_target
    all_targets = numpy.array((track.r_target,track.r_target_alt))
    if type(hole_horizon) is type(None):
        tind_inter = [numpy.argmin(numpy.linalg.norm(track.r_nose-r_tgt,axis=1)) for r_tgt in all_targets]
    else:
        tind_inter = tran.find_first_intersection_index(track.r_nose,all_targets,time=track.time,hole_horizon=hole_horizon)
    return all_targets[numpy.argmax(tind_inter)]

def count_number_of_steps_in_lattice(t_points,r,L_lattice,r_center=None,r_target=None,stop_at_food=False,track=None,return_arena_lattice_geometry=False):
    """
    this function overlays a lattice of size LxL over the arena (centered at the center of the arena)
    then it counts the number of times the trajectory in r crossed from site j to site i of the lattice,
    given that sites i and j are adjacent.
    missing data points in r are interpolated using cubic spline
    (any step that does not go to an adjacent square is considered a missing data point)

    all input vectors are given in arena coordinates

    t_points     -> T-by-1 vector with time points
    r            -> T-by-2 matrix of trajectory points (r[t,:] -> coords of the mouse at time t)
    L_lattice    -> number of squares on each side (x,y) of the square lattice (an odd scalar)
    r_center     -> (x,y) coord of the center of the arena
    r_target     -> (x,y) position of the target (in arena coords)
    stop_at_food -> if stop at food is set, then the step couting stops in the first passsage
                    of the mouse through the food site (i.e., r_target)

    returns
        N         -> (L_lattice**2 by L_lattice**2)
                     number of steps matrix for the square lattice, where N[i,j] is the number of steps the mouse took from site j to site i
                     the site indices i and j are computed by m+n*L  [[ where m is the row index and n is the column index of the square lattice grid ]]
        G         -> (L_lattice by L_lattice) 
                     the actual grid of the square lattice with rows m and columns n, such that
                     G[m,n] = number of times the mouse entered that square
        r_latt    -> (x,y) position of the mouse on the lattice coords
        t_to_food -> number of time steps the mouse took to get the closest to the food
    """
    if L_lattice % 2 == 0:
        raise ValueError('L_lattice must be an odd scalar number')
    if (r_target is None) and stop_at_food:
        if misc.exists(track):
            r_target = track.r_target
        else:
            raise ValueError('please input the r_target because stop_at_food is set')

    # we first define the transformation of coordinates from the arena to the lattice
    r_center            = r_center if misc.exists(r_center) else (track.r_arena_center if misc.exists(track) else plib.get_arena_center(track))
    T,arena_dx,arena_dy = get_arena_to_lattice_transform(L_lattice,r_center=r_center,return_arena_limits=True)

    # transforming the target
    r_target = numpy.asarray(r_target) if stop_at_food else numpy.array((0,0))

    # getting the arena radius
    r_arena_rad_latt = get_circular_grid_radius(L_lattice)

    # we need to remove from r_latt the rows that correspond to time steps where the mouse was lost by the camera
    # these time points will be interpolated in the while-loop below
    t_nan,_  = numpy.nonzero(numpy.isnan(r))
    t_nan    = numpy.unique(t_nan)
    t_points = numpy.delete(t_points,t_nan)
    r        = numpy.delete(r,t_nan,axis=0)

    # then we simply calculate the position in the lattice indices
    r_latt        = apply_arena_to_lattice_transform(T,r) # these are the adjacency matrix indices
                                           # all r_latt indices are guaranteed to be within (0,L_lattice-1) because we added the arena offset of 2cm
    r_target_latt = apply_arena_to_lattice_transform(T,r_target)

    # the adjacency matrix of the square lattice
    # each r_latt[t,:] is a pair of row (  r[t,1]  )  and column  (  r[t,0]  )
    # of the matrix N
    #ij_adjmat = get_nonzero_rowcol_circsqrlatt(L_lattice)
    #N = scipy.sparse.csc_matrix( ( numpy.zeros( len(ij_adjmat[0]) ), ij_adjmat ), shape=(L_lattice**2,L_lattice**2) )
    #N_stop = N.copy()
    N = numpy.zeros((L_lattice**2,L_lattice**2),dtype=float)
    G = numpy.zeros((L_lattice,L_lattice),dtype=float) # square lattice grid

    # now, we must check for each transition 
    # r_latt[t,:] -> r_latt[t+1,:]
    # and count 1 in the adjacency matrix if they are adjacent
    # or interpolate between them if they are not adjacent
    # first we define the interpolation and its parameters
    interpolate_r  = scipy.interpolate.interp1d(t_points,r,kind='linear',axis=0,copy=False)
    n_interp_steps = 10 # number of time points to insert between two steps that did not happen between adjacent squares
    #t_insertion = -1 # time step where the interpolation happens
    #counter_insertion = 0 # counts how many times we interpolated in a given time step
    #n_max_interp = 10 # we allow mostly 10 interpolation insertions to be able to resolve the square adjacency 
    # -- avoids infinite loops --
    # since the spline is continuous, and the grid is sequential, the algorithm is guaranteed to converge

    # then we define some helper functions
    is_adjacent    = lambda r1,r2: ( (abs(r2[0] - r1[0]) == 1) and (abs(r2[1] - r1[1]) == 0) ) ^ ((abs(r2[1] - r1[1]) == 1) and (abs(r2[0] - r1[0]) == 0) )# ^ -> bitwise XOR: i.e., either we are displaced horizontally or vertically, but not both
    is_diagonal    = lambda r1,r2: (abs(r2[0] - r1[0]) == 1) and (abs(r2[1] - r1[1]) == 1)
    is_same_square = lambda r1,r2: numpy.all(r1==r2) #(r1[0] == r2[0]) and (r1[1] == r2[1])
    get_linear_ind = lambda yy,xx: yy+xx*L_lattice

    #if stop_at_food:
    #    is_target = lambda mm,nn: (mm == r_target_latt[0]) and (nn == r_target_latt[1])
    #else:
    #    is_target = lambda mm,nn: False
    #is_target_counter = 0
    #d_to_target = numpy.inf
    # we want to minimize the distance to the target
    #hole_dist = get_mean_min_hole_dist()
    t_ind     = numpy.argmin(numpy.linalg.norm(r-r_target,axis=1))
    t_min     = t_points[t_ind]
    t_to_food = 0

    # here, the magic happens
    t      = 1 # we want to start checking from the second time step (0-based)
    tTotal = r_latt.shape[0]
    while t < tTotal:
        if is_same_square( r_latt[t,:], r_latt[t-1,:] ):
            # if we didn't change squares in this time step
            t+=1 # we simply continue to check the next
            continue
        if not is_adjacent( r_latt[t,:], r_latt[t-1,:] ):
            # oops, we moved to a square that is not adjacent (i.e. 1 unit left or right, 0 top-bottom, OR 1 unit top or bottom, 0 left-right)
            if is_diagonal( r_latt[t,:], r_latt[t-1,:] ):
                # if the step was towards a diagonal neighbor (hence, not adjacent as well)
                # we add a single intermediary horizontal or vertical step
                rr    = r_latt[t,:].copy()
                rr[1] = r_latt[t-1,1] # this is a horizontal step, so we keep the y coordinate equal
                if is_outside_grid_circle(rr[0],rr[1],r_arena_rad_latt): # oops, taking a horizontal step takes us out of the arena
                    rr    = r_latt[t,:].copy() # then we need to take a vertical intermediary step 
                    rr[0] = r_latt[t-1,0] # this is a vertical step, so we keep the x coordinate equal
                tt       = (t_points[t-1] + t_points[t])/2.0
                rr_arena = interpolate_r(tt)
            else:
                # if the step wasn't to an adjacent square, we need to interpolate
                tt       = numpy.linspace(t_points[t-1],t_points[t],n_interp_steps)
                rr_arena = interpolate_r(tt)
                rr       = apply_arena_to_lattice_transform(T,rr_arena) # interpolating trajectory between time steps t-1 and t with n_interp_steps
            #if numpy.any(rr<0):
            #    print('fudeu')
            # we insert the new interpolated coordinates into the original lattice positions
            # at time step t --  we cannot increment t because the t-1 value must be compared to the new t value that has just been inserted
            t_points = numpy.insert(t_points,t,tt)
            r_latt   = numpy.insert(r_latt, t, rr, axis=0)
            r        = numpy.insert(r, t, rr_arena, axis=0)
            # then we have to tell the loop that it must go on a little further
            tTotal   = r_latt.shape[0]
            #print('tTotal=%d'%tTotal)
            continue
        
        # getting row (y) and col (x) in the square lattice (y coord is the row index, x is the col index)
        # from this site and the previous one
        x,y   = r_latt[t,:].astype(int) #get_row_col(r_latt[t,:])
        x0,y0 = r_latt[t-1,:].astype(int) #get_row_col(r_latt[t-1,:])

        # for debugging purposes
        #if (x==x0) and (y==y0):
        #    print('didnt move')

        # calculating the linearized index of the sites for the step matrix
        i,j = get_linear_ind(y,x),get_linear_ind(y0,x0)
        
        # counting one for a visit in this square
        G[y,x] += 1.0

        # counting 1 for a step from site j to site i
        N[i,j] += 1.0

        if t_points[t] <= t_min:
            t_to_food += 1 # counts steps only
        elif stop_at_food:
            break
        #print('t=%d'%t)
        t+=1
    if stop_at_food:
        r_latt    = remove_consecutive_duplicate_rows(append_position(r_latt[:t,:],r_target_latt))
        t_to_food = r_latt.shape[0]
    else:
        r_latt = remove_consecutive_duplicate_rows(r_latt)
    if return_arena_lattice_geometry:
        arena_geometry = misc.structtype(r_center=r_center,arena_radius=plib.get_arena_diameter_cm()/2.0,lattice_extent=misc.flatten_list([arena_dx,numpy.flip(arena_dy)],return_list=True))
        return scipy.sparse.csc_matrix(N),G,r_latt,t_to_food,arena_geometry
    else:
        return scipy.sparse.csc_matrix(N),G,r_latt,t_to_food

def append_position(r,r_new):
    if not numpy.all(r[-1]==r_new):
        return numpy.vstack((r,r_new))
    else:
        return r

def remove_consecutive_duplicate_rows(r):
    """
    r is a numpy.ndarray
    """
    is_equal = lambda v1,v2: numpy.all(v1==v2)
    s = numpy.insert([not is_equal(v1,v2) for v1,v2 in zip(r[:-1],r[1:])],0,True) # checking if consecutive elements are not equal
    return r[s] # returning only those elements that are not equal to their consecutive ones

def apply_arena_to_lattice_transform(T,r):
    return numpy.floor(T(r))#.astype(int)

#def _get_int_or_nan(r):
#    if numpy.any(numpy.isnan(r)):
#        return r
#    else:
#        return r.astype(int)

def get_arena_to_lattice_transform(L,r_center=None,return_arena_limits=False):
    X_arena_lim, Y_arena_lim = plib.get_arena_grid_limits(r_center=r_center)
    X_lattice_lim = (0,L-0.00001) # -0.00001 because python is 0-based index
    Y_lattice_lim = (0,L-0.00001) # -0.00001 because python is 0-based index
    result = misc.LinearTransf2D( X_arena_lim, X_lattice_lim, Y_arena_lim, Y_lattice_lim )
    if return_arena_limits:
        result = (result, X_arena_lim, Y_arena_lim)
    return result

def get_circular_grid_graph(L):
    r = get_circular_grid_radius(L)
    G = networkx.grid_2d_graph(L,L,periodic=False)
    out_nodes = [ (x,y) for x,y in G.nodes if is_outside_grid_circle(x,y,r) ]
    #removing the nodes makes the adjacency matrix smaller and breaks the code
    #G.remove_nodes_from(out_nodes)
    out_edges = [ e for e in G.edges if (is_outside_grid_circle(e[0][1],e[0][0],r) or is_outside_grid_circle(e[1][1],e[1][0],r)) ]
    G.remove_edges_from(out_edges)
    grid = numpy.zeros((L,L))
    for y,x in G.nodes:
        if not is_outside_grid_circle(x,y,r):
            grid[y,x] = 1.0
    return G,grid,out_nodes,out_edges

def get_circular_grid_radius(L):
    return 0.5+float(L-1)/2.0 # arena radius in grid coordinates, we add 0.5 just to include the boundary squares

def is_outside_grid_circle(x,y,r):
    # ((x-r+0.5)**2+(y-r+0.5)**2) > r**2 # sqr center is outside the circle
    # so all the four corners of the square centered at (x,y)
    # must be outside of the circle
    # for the square to be outside of the circle
    return (((x-r)**2+(y-r)**2) > r**2) and (((x-r+1.0)**2+(y-r+1.0)**2) > r**2) and (((x-r+1.0)**2+(y-r)**2) > r**2) and (((x-r)**2+(y-r+1.0)**2) > r**2)

