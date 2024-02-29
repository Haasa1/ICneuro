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
import seaborn
import warnings
import matplotlib.cm
import matplotlib.colors
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches
import modules.helper_func_class as misc
import scipy.interpolate
#import warnings

# def plot_group_comparison(m_sd_se_tuple_A,m_sd_se_tuple_B,ttest_AB=None,show_boxplot=False,boxplotArgs=None,**scatterArgs):
#     """
#     m_sd_se_tuple_X -> (mean,sd,se) of group X
#     """

def savefig(fileName,format='png',dpi=300,facecolor=(1,1,1,1),bbox_inches='tight', verbose=True, **savefig_args):
    """
    wrapper for matplotlib.pyplot.savefig function
    """
    if verbose:
        print(' *** saving ... ', fileName)
    plt.savefig(fileName, format=format,dpi=dpi,facecolor=facecolor,bbox_inches=bbox_inches,**savefig_args)

def plot_stairs_with_edge(ax=None,data=None,edges=None,label='',fill=False,show_edges=True,edgecolor=None,facecolor=None,color=None,facealpha=1.0,edgealpha=1.0,**stairs_kwargs):
    ax = ax if misc.exists(ax) else plt.gca()
    assert misc.exists(data), "you must input some data"
    if misc.exists(facecolor):
        facecolor = numpy.array(matplotlib.colors.to_rgba(facecolor))
    if misc.exists(edgecolor):
        edgecolor = numpy.array(matplotlib.colors.to_rgba(edgecolor))
    if misc.exists(color):
        color     = numpy.array(matplotlib.colors.to_rgba(color))
    if misc.exists(color): # overrides facecolor and edgecolor
        edgecolor = color.copy()
        facecolor = color.copy()
    if misc.exists(facealpha):
        facecolor[-1] = facealpha
    if misc.exists(edgealpha):
        edgecolor[-1] = edgealpha
    if show_edges and fill:
        ax.stairs(data,edges=edges,label='_fill_' + label,fill=True ,facecolor=facecolor,**stairs_kwargs)
        ax.stairs(data,edges=edges,label=label           ,edgecolor=edgecolor,fill=False,**stairs_kwargs)
    else:
        ax.stairs(data,edges=edges,label=label,edgecolor=edgecolor,fill=fill,facecolor=facecolor,**stairs_kwargs)

def displace_bbox(bbox,dx=0.0,dy=0.0,dw=0.0,dh=0.0):
    return bbox.from_bounds(bbox.xmin+dx,bbox.ymin+dy,bbox.width+dw,bbox.height+dh)

def set_axes_property(axh,**axh_args):
    for ax in axh.flatten():
        ax.set(**axh_args)
    return axh

def widen_axes_distance(axh,dx=0.0,dy=0.0):
    try:
        is_1d = axh.ndim == 1
    except:
        return
    try:
        gs    = axh[0].get_gridspec() if is_1d else axh[0,0].get_gridspec()
        nrows = gs.nrows
        ncols = gs.ncols
    except:
        warnings.warn('widen_axes_distance :: axh :: axh has no get_gridspec() function... cannot determine rows or cols')
        return
    n_panels      = axh.size
    is_horizontal = ncols == n_panels
    if is_1d:
        for i,ax in enumerate(axh):
            d  = (dx*i,dy) if is_horizontal else (dx,dy*i)
            p0 = displace_bbox(ax.get_position(),d[0],d[1])
            ax.set_position(p0,which='both')
    else:
        for i in range(nrows):
            for j in range(ncols):
                p0 = displace_bbox(axh[i,j].get_position(),dx*j,dy*i)
                axh[i,j].set_position(p0,which='both')
    return axh



def _annotate_boxplot(ax, x1_ind, x2_ind, data, data_max=None, dy=None, TXT=None, color='k', is_log_scale=False, x1_plot_coord=None, x2_plot_coord=None, use_global_min_max=False, data_min=None, use_min=False, line_args=None, txt_args=None, symbol_args=None, bracket_size=None, dy_bracket=None, dy_txt=None):
    get_func_value = lambda func,d: func(numpy.array(data,dtype=float).flatten()) if use_global_min_max else func((func(d[x1_ind]),func(d[x2_ind]))) # func=max, and max(max(data[x1_ind-1]),max(data[x2_ind-1]))
    dy_default     = 1.15 if is_log_scale else 2.0
    x1_plot_coord  = x1_plot_coord if misc.exists(x1_plot_coord) else x1_ind
    x2_plot_coord  = x2_plot_coord if misc.exists(x2_plot_coord) else x2_ind
    data_max       = data_max      if misc.exists(data_max)      else get_func_value(numpy.max,data)
    data_min       = data_min      if misc.exists(data_min)      else get_func_value(numpy.min,data)
    dy             = dy            if misc.exists(dy)            else dy_default
    dy_txt         = dy_txt        if misc.exists(dy_txt)        else 0.0
    dy_bracket     = dy_bracket    if misc.exists(dy_bracket)    else 0.0
    bracket_size   = bracket_size  if misc.exists(bracket_size)  else 1.0
    operator       = (float.__truediv__ if is_log_scale else float.__sub__ ) if use_min else (float.__mul__  if is_log_scale else float.__add__)
    operator_np    = (numpy.divide      if is_log_scale else numpy.subtract) if use_min else (numpy.multiply if is_log_scale else numpy.add    )
    displace_y     = lambda y0,delta_y: operator(float(y0),float(delta_y))
    displace_y_np  = lambda y0,delta_y: operator_np(y0,delta_y)
    data_lim       = data_min if use_min else data_max
    new_data_lim   = displace_y(data_lim,dy)
    y, h, col      = new_data_lim, (bracket_size*dy), color
    y_val          = numpy.array([y, displace_y(y,h), displace_y(y,h), y])
    #if is_log_scale:
    #    if not misc.exists(dy):
    #        dy = 1.15
    #    new_data_max = data_max * dy
    #else:
    #    if not misc.exists(dy):
    #        dy = 2
    #    new_data_max = data_max + dy
    #y, h, col = new_data_max, dy, color
    #y_val = [y, y*h, y*h, y] if is_log_scale else [y, y+h, y+h, y]
    ax.plot(numpy.array([x1_plot_coord, x1_plot_coord, x2_plot_coord, x2_plot_coord]), displace_y_np(y_val,dy_bracket), **_get_kwargs(line_args, linewidth=1.5, color=col))
    y0 = displace_y(y,h*dy*dy) if is_log_scale else ( displace_y(y,h) if misc.exists(TXT) else displace_y(y,3*h) )
    y0 = displace_y(displace_y(y0,dy_bracket),dy_txt)
    if misc.exists(TXT):
        #y0 = y*h*dy*dy if is_log_scale else y+h
        txt_va = 'top' if use_min else 'bottom' 
        ax.text((x1_plot_coord+x2_plot_coord)*.5, y0, TXT, **_get_kwargs(txt_args,ha='center', va=txt_va, color=col))
    else:
        #y0 = y*h*dy*dy if is_log_scale else y+3*h
        ax.plot((x1_plot_coord+x2_plot_coord)*.5, y0, **_get_kwargs(symbol_args,markersize=6, marker=(5,2), linestyle='none', color=col))
    return displace_y(y0,dy) #y0*dy if is_log_scale else y0+dy

def plot_boxplot(ax,data,labels,colors=None,significance=None,data_std=None,is_log_scale=False,boxplotargs=None,errorbarargs=None,positions=None):
    """
    data         -> a list of variables (each element contains all the data points used to calculate the average of that element, etc )
    colors       -> list of colors for each entry in data
    labels       -> list of labels for each entry in data
    significance -> a matrix of significance: each entry is a list of 0 and 1 (for significant) for the corresponding data variable versus the other data variables
    data_std     -> a list of std for each data variable (if not given, this will be automatically calculated)
    data_control -> a list of control variables for each variable in data (not necessary, it can contain empty (or None) entries meaning that data variable has no control)
    is_log_scale -> if True, significance asterisks are adjusted for log-scale y-axis plots
    boxplotargs  -> arguments passed down to the seaborn.boxplot function (and further down to matplotlib.pyplot.boxplot)
    errorbarargs -> arguments passed down to the maxplotlib.pyplot.errorbar function that plots stddev

    https://www.python-graph-gallery.com/30-basic-boxplot-with-seaborn
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    https://seaborn.pydata.org/generated/seaborn.boxplot.html
    """
    #N = len(data)
    colors                      = get_tab10_colors(len(data))             if type(colors)       is type(None) else colors
    positions                   = numpy.arange(len(data))                 if type(positions)    is type(None) else positions
    data_std                    = misc.get_empty_list(len(data))          if type(data_std)     is type(None) else data_std
    #data_control               = misc.get_empty_list(len(data))          if type(data_control) is type(None) else data_control
    boxplotargs                 = dict()                                  if type(boxplotargs)  is type(None) else boxplotargs
    errorbarargs                = dict()                                  if type(errorbarargs) is type(None) else errorbarargs
    boxplotargs                 = misc.set_default_kwargs(boxplotargs, patch_artist=True, vert=True, widths=0.2, meanline=True, showmeans=False, notch=False, boxprops=dict(), medianprops=dict(), whiskerprops=dict(), showcaps=False)
    boxplotargs['boxprops']     = misc.set_default_kwargs(boxplotargs['boxprops']    ,linewidth=0.5,alpha=0.2)
    boxplotargs['medianprops']  = misc.set_default_kwargs(boxplotargs['medianprops'] ,linewidth=3,color='k')
    boxplotargs['whiskerprops'] = misc.set_default_kwargs(boxplotargs['whiskerprops'],color=(0,0,0,0))
    errorbarargs                = misc.set_default_kwargs(errorbarargs,fmt='o',linewidth=3)
    #data_labels_fake = [ k for k in positions ]
    #color_palette    = { lab:cc for lab,cc in zip(data_labels_fake,colors) }
    #dfs = [ _get_dataframe_for_boxplot(lab,x,None) for x,lab in zip(data,data_labels_fake) ] #for x,lab,x_control in zip(data,labels,data_control)
    #dfm = dfs[0].append(dfs[1:])
    #ax = seaborn.boxplot(x='label', y='value', data=dfm, palette=color_palette, **boxplotargs) 
    bplot = ax.boxplot(data, labels=labels, positions=positions, **boxplotargs) 
    for k,(box_patch,c) in enumerate(zip(bplot['boxes'],colors)):
        box_patch.set_facecolor(c)
    ax.set_xticks(ticks=ax.get_xticks(),labels=labels)
    #ax = seaborn.boxplot(x='label', y='value', hue='is_control', data=dfm, **boxplotargs)# works for data_control, but I dont know how to change colors
    #ax = seaborn.boxplot(x='is_control', y='value', hue='label', data=dfm, palette=color_palette, **boxplotargs)
    get_data_std = lambda k,dd: data_std[k] if not(type(data_std[k]) is type(None)) else misc.nanstd(dd)
    for k,(p,dd) in enumerate(zip(positions,data)):
        ax.errorbar(p, misc.nanmean(dd), yerr=get_data_std(k,dd), color=colors[k], **errorbarargs)
    if misc.exists(significance):
        if len(significance) > 0:
            mm = misc.nanmax(misc.asarray_nanfill(data).flatten())
            for i,ind in enumerate(significance):
                for j in numpy.nonzero(ind)[0]:
                    mm = _annotate_boxplot(ax,i,j,data,data_max=mm,is_log_scale=is_log_scale,color=colors[i],x1_plot_coord=positions[i],x2_plot_coord=positions[j])
    return ax

def _get_dataframe_for_boxplot(label,x,x_control):
    values     = numpy.asarray(x)
    is_control = numpy.zeros(values.size,dtype=int)
    if misc.exists(x_control) and (len(x_control) > 0):
        values     = numpy.append(values,x_control)
        is_control = numpy.append(is_control,numpy.ones(numpy.asarray(x_control).size,dtype=int))
    return pandas.DataFrame(dict(label=numpy.repeat(label,values.size), value=values))#, is_control=is_control))

def plot_scatter(ax,X_rt,Y_rt,X_ft,Y_ft,linreg_rt=None,linreg_ft=None,ind=None,use_log_for_linreg=False,red_colors=None,green_colors=None,X_err_rt=None,Y_err_rt=None,X_err_ft=None,Y_err_ft=None,linreg_func_internal=None,pSymbols_rt=None,pSymbols_ft=None,markersize_rt=None,markersize_ft=None,errorbarArgs_rt=None,errorbarArgs_ft=None,show_connecting_lines=False,connLines_args=None,markerfacecolor_alpha=1.0,color_regression_rt=None,color_regression_ft=None):
    n = X_rt.shape[0]
    get_if_not_none = lambda x,k: None if type(x) is type(None) else x[k,:]
    linreg_func = lambda x,linreg: linreg.intercept + linreg.slope * x
    color_rt = numpy.array((65, 102, 216, 255))/255
    color_ft = numpy.array((224, 53, 53, 255))/255
    linreg_func_internal = linreg_func                  if type(linreg_func_internal) is type(None) else linreg_func_internal
    red_colors           = get_gradient(n,'red')        if type(red_colors)           is type(None) else red_colors
    green_colors         = get_gradient(n,'blue2')      if type(green_colors)         is type(None) else green_colors
    pSymbols_rt          = ['o']                        if type(pSymbols_rt)          is type(None) else pSymbols_rt
    pSymbols_ft          = ['s']                        if type(pSymbols_ft)          is type(None) else pSymbols_ft
    markersize_rt        = [5]                          if type(markersize_rt)        is type(None) else markersize_rt
    markersize_ft        = [4]                          if type(markersize_ft)        is type(None) else markersize_ft
    errorbarArgs_rt      = dict()                       if type(errorbarArgs_rt)      is type(None) else errorbarArgs_rt
    errorbarArgs_ft      = dict()                       if type(errorbarArgs_ft)      is type(None) else errorbarArgs_ft
    connLines_args       = dict()                       if type(connLines_args)       is type(None) else connLines_args
    color_regression_rt  = color_rt                     if type(color_regression_rt)  is type(None) else color_regression_rt
    color_regression_ft  = color_ft                     if type(color_regression_ft)  is type(None) else color_regression_ft
    get_k = misc.get_item_recurrently
    if type(ind) is type(None):
        ind = list(range(n))
    elif numpy.isscalar(ind):
        ind = list(range(int(ind),int(ind)+1))
    if show_connecting_lines:
        n_mouse = X_rt.shape[1]
        for j in range(n_mouse):
            cl_args = misc.set_default_kwargs(connLines_args,linestyle=':',alpha=0.4,label='_connLine %d'%j)
            ax.plot(X_rt[ind,j].T,Y_rt[ind,j].T,color=green_colors[0],**cl_args)
            ax.plot(X_ft[ind,j].T,Y_ft[ind,j].T,color=  red_colors[0],**cl_args)
    get_label_flag = lambda k: ('' if k == ind[-1] else '_') if type(linreg_rt) is type(None) else '_'
    for k in ind:
        label_flag = get_label_flag(k) #('' if k == ind[-1] else '_') if type(linreg_rt) is type(None) else '_'
        args_rt = misc.set_default_kwargs(errorbarArgs_rt,color=green_colors[k],marker=get_k(k,pSymbols_rt),
                                          markersize=get_k(k,markersize_rt),linestyle='none',
                                          markerfacecolor=list(green_colors[k])+[markerfacecolor_alpha],
                                          label=label_flag+'Static entrance')
        ax.errorbar(X_rt[k,:],Y_rt[k,:],xerr=get_if_not_none(X_err_rt,k),yerr=get_if_not_none(Y_err_rt,k),**args_rt)
    for k in ind:
        label_flag = get_label_flag(k) #('' if k == ind[-1] else '_') if type(linreg_rt) is type(None) else '_'
        args_ft = misc.set_default_kwargs(errorbarArgs_ft,color=  red_colors[k],marker=get_k(k,pSymbols_ft),
                                          markersize=get_k(k,markersize_ft),linestyle='none',
                                          markerfacecolor=list(red_colors[k])+[markerfacecolor_alpha],
                                          label=label_flag+'Random entrance')
        ax.errorbar(X_ft[k,:],Y_ft[k,:],xerr=get_if_not_none(X_err_ft,k),yerr=get_if_not_none(Y_err_ft,k),**args_ft)
    if (type(linreg_rt) is type(None)) and (type(linreg_ft) is type(None)):
        return ax,green_colors,red_colors
    if use_log_for_linreg:
        get_pos = lambda x: x[x>0]
        x_linreg_rt = numpy.logspace(numpy.log10(misc.nanmin(get_pos(X_rt.flatten()))),numpy.log10(misc.nanmax(X_rt.flatten())),100)
        x_linreg_ft = numpy.logspace(numpy.log10(misc.nanmin(get_pos(X_ft.flatten()))),numpy.log10(misc.nanmax(X_ft.flatten())),100)
    else:
        x_linreg_rt = numpy.linspace(misc.nanmin(X_rt.flatten()),misc.nanmax(X_rt.flatten()),100)
        x_linreg_ft = numpy.linspace(misc.nanmin(X_ft.flatten()),misc.nanmax(X_ft.flatten()),100)
    if not(type(linreg_rt) is type(None)):
        ax.plot(x_linreg_rt,linreg_func_internal(x_linreg_rt,linreg_rt),'-',color=color_regression_rt,linewidth=3,label='Static entrance; R={0:.2g};p={1:.3g}'.format(linreg_rt.rvalue,linreg_rt.pvalue),zorder=100)
    if not(type(linreg_ft) is type(None)):
        ax.plot(x_linreg_ft,linreg_func_internal(x_linreg_ft,linreg_ft),'-',color=color_regression_ft,  linewidth=3,label='Random entrance; R={0:.2g};p={1:.3g}'.format(linreg_ft.rvalue,linreg_ft.pvalue),zorder=100)
    return ax,green_colors,red_colors


def get_gradient_between(c1,c2,N=None,use_matplotlib=True):
    c = (matplotlib.colors.to_rgb(c1),matplotlib.colors.to_rgb(c2))
    if use_matplotlib:
        f=matplotlib.colors.LinearSegmentedColormap.from_list('Custom',c)
    else:
        c = numpy.array(c,dtype=float)
        f = scipy.interpolate.interp1d(numpy.arange(2).astype(float),c,kind='linear',axis=0,copy=False)
    if misc.exists(N):
        return f(numpy.linspace(0.0,1.0,N))
    else:
        return f
    

def get_cmap_plasma_inv_lum(N=None,reverse=False):
    c = numpy.array([(232,221,255,255),
                     (255,167,253,255),
                     (210,150,146,255),
                     (146,146,150,255)],dtype=float)/255.0
    if reverse:
        c = numpy.flipud(c)
    f = matplotlib.colors.LinearSegmentedColormap.from_list('Custom',c)
    if misc.exists(N):
        return f(numpy.linspace(0.0,1.0,N))
    else:
        return f

def get_gradient(N=None,color='red',cmap_name=None):
    if misc.exists(cmap_name):
        f = plt.get_cmap(cmap_name)
    else:
        if type(color) is str:
            color = color.lower()
            color_str_options = ['red','green','blue','blue2','yellow','purple','orange']
            if (color in color_str_options):
                if color == 'red':
                    c = numpy.array([[244,138,140],[133,27,30]],dtype=float)/255.0
                elif color == 'blue':
                    c = numpy.array([[138,140,244],[27,30,133]],dtype=float)/255.0
                elif color == 'blue2':
                    c = numpy.array([[153,217,247],[9,91,130]],dtype=float)/255.0
                elif color == 'green':
                    c = numpy.array([[0.698,0.83922,0.4863],[0.33,0.47,0.13]],dtype=float)
                elif color == 'yellow':
                    c = numpy.array([[234,221,121],[140,128,36]],dtype=float)/255.0
                elif color == 'purple':
                    c = numpy.array([[227,168,247],[92,4,124]],dtype=float)/255.0
                elif color == 'orange':
                    c = numpy.array([[247, 183, 106],[117, 78, 31]],dtype=float)/255.0
                f = scipy.interpolate.interp1d(numpy.arange(2).astype(float),c,kind='linear',axis=0,copy=False)
            else:
                warnings.warn('Assuming color is a matplotlib colormap')
                try:
                    f = get_gradient(cmap_name=color)
                except:
                    raise ValueError("color must be either a valid matplotlib colormap or one of %s"%(str(color_str_options)[1:-1]))
        else:
            # assuming color is a function
            warnings.warn('Assuming color is a function')
            f = color
    if misc.exists(N):
        return f(numpy.linspace(0.0,1.0,N))
    else:
        return f

def get_tab10_colors(N):
    c = plt.get_cmap('tab10')(numpy.linspace(0,1,10))
    if N > 10:
        c = numpy.tile(c,(int(numpy.ceil(float(N)/10.0)),1))[:N]
    else:
        c = c[:N]
    return c

def fill_between_trajectories(ax,r1,r2,**polygonArgs):
    """
    r1,r2 -> two x(col0),y(col1) vectors of time
    the list of valid polygonArgs: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html#matplotlib.patches.Polygon
    includes: facecolor,edgecolor,linewidth,alpha,antialiased,label
    """
    if type(ax) is type(None):
        ax = plt.gca()
    polygonArgs = misc.set_default_kwargs(polygonArgs,facecolor='tab:gray',edgecolor=None,linewidth=0,alpha=0.2,antialiased=True,label='')
    rr = numpy.vstack((r1,numpy.flipud(r2)))
    return ax.fill(rr[:,0],rr[:,1],**polygonArgs)

def plot_trajectory(ax,r,time=None,velocity=None,label=None,r_start=None,r_target=None,r_target_alt=None,r_target_reverse=None,r_target_alt_reverse=None,color='tab:blue',line_gradient_variable='none',linewidth=1.5,alpha=1.0,startArgs=None,targetArgs=None,targetAltArgs=None,targetRevArgs=None,targetAltRevArgs=None,show_colorbar=True):
    """
    plots mouse trajectory

    if color is callable, for example,
    color = plt.get_cmap('jet')
    then, the trajectory is colored according to this gradient

    line_gradient_variable -> 'time': then color the gradient accordint to time (t=0 is one color, and t=max time is the last color)
                              'velocity': color the trajectory according to the velocity of the mouse
                              'none': use solid color
                              numpy.ndarray: color for each time point (samme shape and size as time)
                              list of numpy.ndarray: each entry is the color for each trajectory in the r list
    """
    is_valid_str_line_grad_value = (type(line_gradient_variable) is str) and (line_gradient_variable.lower() in ['time','velocity','none'])
    is_line_gradient_str_eq = lambda lg,val: is_valid_str_line_grad_value and (lg == val)
    if is_valid_str_line_grad_value:
        line_gradient_variable = line_gradient_variable.lower()
    if type(r) is list:
        r_start                = r_start                if misc.is_list_of_1d_collection_or_none(r_start,                min_n_elem=2)                                     else misc.repeat_to_complete([r_start],                len(r))
        r_target               = r_target               if misc.is_list_of_1d_collection_or_none(r_target,               min_n_elem=2)                                     else misc.repeat_to_complete([r_target],               len(r))
        r_target_alt           = r_target_alt           if misc.is_list_of_1d_collection_or_none(r_target_alt,           min_n_elem=2)                                     else misc.repeat_to_complete([r_target_alt],           len(r))
        r_target_reverse       = r_target_reverse       if misc.is_list_of_1d_collection_or_none(r_target_reverse,       min_n_elem=2)                                     else misc.repeat_to_complete([r_target_reverse],       len(r))
        r_target_alt_reverse   = r_target_alt_reverse   if misc.is_list_of_1d_collection_or_none(r_target_alt_reverse,   min_n_elem=2)                                     else misc.repeat_to_complete([r_target_alt_reverse],   len(r))
        time                   = time                   if misc.is_list_of_1d_collection_or_none(time,                   min_n_elem=2)                                     else misc.repeat_to_complete([time],                   len(r))
        velocity               = velocity               if misc.is_list_of_1d_collection_or_none(velocity,               min_n_elem=2)                                     else misc.repeat_to_complete([velocity],               len(r))
        line_gradient_variable = line_gradient_variable if misc.is_list_of_1d_collection_or_none(line_gradient_variable, min_n_elem=r[0].shape[0], collection_is_str=True) else misc.repeat_to_complete([line_gradient_variable], len(r))
        alpha                  = alpha                  if (not numpy.isscalar(alpha))                                                                                     else misc.repeat_to_complete([alpha],                  len(r))
        linewidth              = linewidth              if (not numpy.isscalar(linewidth))                                                                                 else misc.repeat_to_complete([linewidth],              len(r))
        #raise ValueError('line_gradient_variable must be either time, velocity, or none; or ndarray of the same shae as time, or a list of ndarray, or a list of str; list assign each item to each item in the r list')
        if callable(color):
            color = [ color ]
        else:
            if type(color) is list:
                if len(color) < len(r):
                    color = misc.repeat_to_complete(color,len(r))
                else:
                    color = color[:len(r)]
            else:
                color = get_tab10_colors(len(r))
        l = misc.get_empty_list(len(r))
        for k,rr in enumerate(r):
            #s="""
            #    misc.get_element_or_none(time,k)                   = %s
            #    misc.get_element_or_none(velocity,k)               = %s
            #    misc.get_element_or_none(label,k)                  = %s
            #    misc.get_element_or_none(r_start,k)                = %s
            #    misc.get_element_or_none(r_target,k)               = %s
            #    misc.get_element_or_none(r_target_reverse,k)       = %s
            #    misc.get_element_or_none(color,k)                  = %s
            #    misc.get_element_or_none(line_gradient_variable,k) = %s
            #    misc.get_element_or_none(linewidth,k)              = %s
            #    misc.get_element_or_none(alpha,k)                  = %s
            #"""
            #print(s%(misc.get_element_or_none(time,k),misc.get_element_or_none(velocity,k),misc.get_element_or_none(label,k),misc.get_element_or_none(r_start,k),misc.get_element_or_none(r_target,k),misc.get_element_or_none(r_target_reverse,k),misc.get_element_or_none(color,k),misc.get_element_or_none(line_gradient_variable,k),misc.get_element_or_none(linewidth,k),misc.get_element_or_none(alpha,k)))
            l[k] = plot_trajectory(ax,rr,time                   = misc.get_element_or_none(time,k),
                                         velocity               = misc.get_element_or_none(velocity,k),
                                         label                  = misc.get_element_or_none(label,k),
                                         r_start                = misc.get_element_or_none(r_start,k),
                                         r_target               = misc.get_element_or_none(r_target,k),
                                         r_target_alt           = misc.get_element_or_none(r_target_alt,k),
                                         r_target_reverse       = misc.get_element_or_none(r_target_reverse,k),
                                         r_target_alt_reverse   = misc.get_element_or_none(r_target_alt_reverse,k),
                                         color                  = misc.get_element_or_none(color,k),
                                         line_gradient_variable = misc.get_element_or_none(line_gradient_variable,k),
                                         linewidth              = misc.get_element_or_none(linewidth,k),
                                         alpha                  = misc.get_element_or_none(alpha,k),
                                         startArgs=startArgs,targetArgs=targetArgs,targetAltArgs=targetAltArgs,targetRevArgs=targetRevArgs,targetAltRevArgs=targetAltRevArgs)
        l = l[0] if len(l) == 1 else l
        return l
    else:
        startArgs        = _get_kwargs(startArgs        ,marker='s',color='r',markersize=15,fillstyle='none',linewidth=2)
        targetArgs       = _get_kwargs(targetArgs       ,marker='d',color='r',markersize=15,fillstyle='none',linewidth=2)
        targetAltArgs    = _get_kwargs(targetAltArgs    ,marker='d',color='b',markersize=15,fillstyle='none',linewidth=2)
        targetRevArgs    = _get_kwargs(targetRevArgs    ,marker='D',color='m',markersize=15,fillstyle='none',linewidth=2)
        targetAltRevArgs = _get_kwargs(targetAltRevArgs ,marker='D',color='g',markersize=15,fillstyle='none',linewidth=2)
        assert (type(r) is numpy.ndarray) and (r.ndim == 2),"r must be a 2d numpy.ndarray with rows as time and col1 == x, col2 == y"
        show_start              = misc.is_valid_1d_collection(r_start,2)
        show_target             = misc.is_valid_1d_collection(r_target,2)
        show_target_alt         = misc.is_valid_1d_collection(r_target_alt,2)
        show_reverse_target     = misc.is_valid_1d_collection(r_target_reverse,2)
        show_reverse_alt_target = misc.is_valid_1d_collection(r_target_alt_reverse,2)
        l = []
        if show_start:
            txt,txtArgs = _get_label(startArgs) # must come before so that the 'pop' removes labelArgs before the plot
            l.append(ax.plot(r_start[0],r_start[1],                   zorder=1000, **startArgs    ))
            if misc.exists(txt):
                padding = txtArgs.pop('pad',numpy.zeros(2))
                l.append(ax.annotate(txt,numpy.asarray(r_start)+padding,**txtArgs))
        if show_target:
            txt,txtArgs = _get_label(targetArgs)  # must come before so that the 'pop' removes labelArgs before the plot
            l.append(ax.plot(r_target[0],r_target[1],                **_get_kwargs(targetArgs,zorder=1001)   ))
            if misc.exists(txt):
                padding = txtArgs.pop('pad',numpy.zeros(2))
                l.append(ax.annotate(txt,numpy.asarray(r_target)+padding,**txtArgs))
        if show_target_alt:
            txt,txtArgs = _get_label(targetAltArgs) # must come before so that the 'pop' removes labelArgs before the plot
            l.append(ax.plot(r_target_alt[0],r_target_alt[1],        **_get_kwargs(targetAltArgs,zorder=1002)))
            if misc.exists(txt):
                padding = txtArgs.pop('pad',numpy.zeros(2))
                l.append(ax.annotate(txt,numpy.asarray(r_target_alt)+padding,**txtArgs))
        if show_reverse_target:
            txt,txtArgs = _get_label(targetRevArgs) # must come before so that the 'pop' removes labelArgs before the plot
            l.append(ax.plot(r_target_reverse[0],r_target_reverse[1], **_get_kwargs(targetRevArgs,zorder=1003)))
            if misc.exists(txt):
                padding = txtArgs.pop('pad',numpy.zeros(2))
                l.append(ax.annotate(txt,numpy.asarray(r_target_reverse)+padding,**txtArgs))
        if show_reverse_alt_target:
            txt,txtArgs = _get_label(targetAltRevArgs) # must come before so that the 'pop' removes labelArgs before the plot
            l.append(ax.plot(r_target_alt_reverse[0],r_target_alt_reverse[1], **_get_kwargs(targetAltRevArgs,zorder=1004)))
            if misc.exists(txt):
                padding = txtArgs.pop('pad',numpy.zeros(2))
                l.append(ax.annotate(txt,numpy.asarray(r_target_alt_reverse)+padding,**txtArgs))
        if callable(color) or (is_valid_str_line_grad_value and (line_gradient_variable != 'none')) or (not numpy.isscalar(line_gradient_variable)):
            if is_line_gradient_str_eq(line_gradient_variable,'time') and (type(time) is type(None)):
                raise ValueError('if you want use time as gradient, set the time parameter')
            if is_line_gradient_str_eq(line_gradient_variable,'velocity') and (type(velocity) is type(None)):
                raise ValueError('if you want use velocity as gradient, set the velocity parameter')
            if not callable(color):
                color = plt.get_cmap('plasma') if is_line_gradient_str_eq(line_gradient_variable,'time') else plt.get_cmap('jet')
            if is_line_gradient_str_eq(line_gradient_variable,'time'):
                colors = normalize_variable_to_01(time)
            elif is_line_gradient_str_eq(line_gradient_variable,'velocity'):
                colors = normalize_variable_to_01(velocity)
            else:
                if not is_valid_str_line_grad_value:
                    colors = normalize_variable_to_01(line_gradient_variable)
            l.append(plot_gradient_line(r, colors=colors, cmap=color, linewidth=linewidth, alpha=alpha, ax=ax, label=label))
            if show_colorbar:
                cbar_label = 'normalized ' + (line_gradient_variable if type(line_gradient_variable) is str else 'custom var')
                _add_colorbar(ax,color,label=cbar_label,w_fraction_of_ax=0.02,h_fraction_of_ax=0.15,bbox_ref=ax.get_position())
        else:
            l.append(ax.plot(r[:,0],r[:,1],'-',color=color,linewidth=linewidth, alpha=alpha, label=label))
        l = [ ll for ll in l if not(type(ll) is type(None)) ]
        return l

def _get_kwargs(args,**defaults):
    args = args if misc.exists(args) else dict()
    return misc.set_default_kwargs(args,**defaults)

def _add_colorbar(ax,cmap,label='',w_fraction_of_ax=0.02,h_fraction_of_ax=0.15,p0=(1,1),bbox_ref=None,minmax_tick_labels=None,title='',**cbarArgs):
    """
    #cbar.ax.yaxis.set_ticks_position('left')
    #cbar.ax.xaxis.set_ticks_position('top')
    """
    if type(cmap) is numpy.ndarray:
        assert cmap.ndim == 2,"the colors must be a 2d array of size Nx3 or Nx4"
        cmap = matplotlib.colors.ListedColormap(cmap)
    if type(bbox_ref) is type(None):
        bbox_ref = ax.get_position()
    pos = [bbox_ref.xmin + bbox_ref.width *(p0[0]-w_fraction_of_ax-0.025), #x
           bbox_ref.ymin + bbox_ref.height*(p0[1]-h_fraction_of_ax-0.015), #y
           bbox_ref.width *w_fraction_of_ax, #w
           bbox_ref.height*h_fraction_of_ax] #h
    cax = ax.get_figure().add_axes(pos,label='color_traj')
    #cax = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax,width="5%",height="15%",loc='upper right',bbox_to_anchor=(1,1,1,1),bbox_transform=ax.transAxes,borderpad=0)
    #cax.set_title(cbar_label)
    title_txt,title_args = _get_label(cbarArgs,'title')
    _,ticklabels_args = _get_label(cbarArgs,'ticklabels')
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap),cax=cax,ax=ax,label=label,**cbarArgs)
    title = title if len(title)>0 else title_txt
    if title:
        cax.set_title(title,**title_args)
    if misc.exists(minmax_tick_labels): #not(type(minmax_tick_labels) is type(None)):
        cbar.set_ticks((0,1))
        cbar.set_ticklabels(minmax_tick_labels,**ticklabels_args)
    return cbar,cax

def plot_mouse_trajectory(ax,track,mouse_part='nose',show_start=True,show_target=True,show_reverse_target=False,show_alt_target=False,show_reverse_alt_target=False,color='tab:blue',line_gradient_variable='none',linewidth=1.5,alpha=1.0,startArgs=None,targetArgs=None,targetAltArgs=None,targetRevArgs=None,targetAltRevArgs=None,show_colorbar=True):
    """
    plots mouse trajectory

    if color is callable, for example,
    color = plt.get_cmap('jet')
    then, the trajectory is colored according to this gradient

    line_gradient_variable -> 'time': then color the gradient accordint to time (t=0 is one color, and t=max time is the last color)
                              'velocity': color the trajectory according to the velocity of the mouse
                              'none': use solid color
                              numpy.ndarray: color for each time point (samme shape and size as time)
                              list of numpy.ndarray: each entry is the color for each trajectory in the r list
    """
    is_valid_str_line_grad_value = (type(line_gradient_variable) is str) and (line_gradient_variable.lower() in ['time','velocity','none'])
    is_line_gradient_str_eq = lambda lg,val: is_valid_str_line_grad_value and (lg == val)
    if is_valid_str_line_grad_value:
        line_gradient_variable = line_gradient_variable.lower()
    mouse_part = 'nose' if mouse_part is None else mouse_part
    if not (mouse_part in ['center','nose','tail']):
        raise ValueError('mouse_part must be center, nose or tail')
    lab = 'r_'+mouse_part
    if type(track) is list:
        line_gradient_variable = line_gradient_variable if misc.is_list_of_1d_collection_or_none(line_gradient_variable, min_n_elem=track[0].time.size, collection_is_str=True) else misc.repeat_to_complete([line_gradient_variable],len(track))
        alpha                  = alpha                  if (not numpy.isscalar(alpha))                                                                                          else misc.repeat_to_complete([alpha],len(track))
        linewidth              = linewidth              if (not numpy.isscalar(linewidth))                                                                                      else misc.repeat_to_complete([linewidth],len(track))
        if type(color) is list:
            if len(color) < len(track):
                color = misc.repeat_to_complete(color,len(track))
            else:
                color = color[:len(track)]
        else:
            color = get_tab10_colors(len(track))
        l = misc.get_empty_list(len(track))
        for k,tr in enumerate(track):
            r_start              = tr.r_start              if show_start              else None
            r_target             = tr.r_target             if show_target             else None
            r_target_alt         = tr.r_target_alt         if show_alt_target         else None
            r_target_reverse     = tr.r_target_reverse     if show_reverse_target     else None
            r_target_alt_reverse = tr.r_target_alt_reverse if show_reverse_alt_target else None
            l[k] = plot_trajectory(ax,tr[lab],time=tr.time,velocity=tr.velocity,label='Mouse %s, trial %s'%(tr.mouse_number,tr.trial),
                                   r_start=r_start,
                                   r_target=r_target,
                                   r_target_alt=r_target_alt,
                                   r_target_reverse=r_target_reverse,
                                   r_target_alt_reverse=r_target_alt_reverse,
                                   color=misc.get_element_or_none(color,k),
                                   line_gradient_variable=misc.get_element_or_none(line_gradient_variable,k),
                                   linewidth=misc.get_element_or_none(linewidth,k),
                                   alpha=misc.get_element_or_none(alpha,k),
                                   startArgs=startArgs,targetArgs=targetArgs,targetAltArgs=targetAltArgs,targetRevArgs=targetRevArgs,targetAltRevArgs=targetAltRevArgs,show_colorbar=show_colorbar)
        l = l[0] if len(l) == 1 else l
    else:
        r_start              = track.r_start              if show_start              else None
        r_target             = track.r_target             if show_target             else None
        r_target_alt         = track.r_target_alt         if show_alt_target         else None
        r_target_reverse     = track.r_target_reverse     if show_reverse_target     else None
        r_target_alt_reverse = track.r_target_alt_reverse if show_reverse_alt_target else None
        l = plot_trajectory(ax,track[lab],time=track.time,velocity=track.velocity,label='Mouse %s, trial %s'%(track.mouse_number,track.trial),
                            r_start=r_start,
                            r_target=r_target,
                            r_target_alt=r_target_alt,
                            r_target_reverse=r_target_reverse,
                            r_target_alt_reverse=r_target_alt_reverse,
                            color=color,line_gradient_variable=line_gradient_variable,linewidth=linewidth,alpha=alpha,
                            startArgs=startArgs,targetArgs=targetArgs,targetAltArgs=targetAltArgs,targetRevArgs=targetRevArgs,targetAltRevArgs=targetAltRevArgs,show_colorbar=show_colorbar)
    return l

def _get_text_anchor_alignment(r):
    valign = 'baseline' if r[1] > 0 else 'top'
    halign = 'left'     if r[0] > 0 else 'right'
    return valign,halign

def _get_label(args,label_arg_name='label'):
    if misc.exists(args):
        return args.pop(label_arg_name,None), _get_kwargs(args.pop(label_arg_name+'Args',None))
    return None,dict()

def _get_first_ax(ax):
    try:
        return ax.flatten()[0]
    except AttributeError:
        return ax

def _fix_title(a):
    txt = a.get_title()
    if len(txt) > 0:
        fontdict=dict(fontsize          = a.title._fontproperties._size,
                    fontweight          = a.title._fontproperties._weight,
                    color               = a.title._color                 ,
                    verticalalignment   = a.title._verticalalignment     )
        a.set_title(txt,y=a.title._y,loc=a.title._horizontalalignment,fontdict=fontdict,pad=-8)

def tight_arena_panels(ax,set_axis_off=False,adjust_title_position=True,dx_amid_panels=0.0,dy_amid_panels=0.0,dy0=0.0):
    if not misc._is_numpy_array(ax):
        ax.autoscale()
        ax.set_aspect('equal','box')
        if set_axis_off:
            ax.axis('off')
        if adjust_title_position:
            _fix_title(ax)
        return ax
    shape = ax.shape
    is_1d = ax.ndim == 1
    ax = ax.flatten()
    [ a.autoscale() for a in ax.flatten() ]
    [ a.set_aspect('equal','box') for a in ax.flatten() ]
    p0 = [ numpy.min([ a.get_position().x0 for a in ax.flatten() ]), numpy.max([ a.get_position().y1 for a in ax.flatten() ]), ax[0].get_position().width,  ax[0].get_position().height]
    for k,a in enumerate(ax.flatten()):
        if set_axis_off:
            a.axis('off')
        if is_1d:
            i,j = (k,0) if a.get_gridspec().nrows > 1 else (0,k)
        else:
            i,j = numpy.unravel_index(k,shape)
        p = [  p0[0]+j*(p0[2]+dx_amid_panels), dy0+p0[1]-i*(p0[3]+dy_amid_panels)-0.1, p0[2], p0[3]  ]
        a.set_position( p, which='both' )
        if adjust_title_position:
            _fix_title(a)
    return ax.reshape(shape)

def plot_dispersion(r_mean, r_eigdir, r_disp, ax=None, show_center=True, color=None, marker=None, center_args=None, **ellipse_args):
    """
    plots dispersion returned by misc.calc_dispersion
    """
    color        = color  if misc.exists( color) else 'k'
    marker       = marker if misc.exists(marker) else 'x'
    center_args  = misc._get_kwargs(center_args ,markeredgewidth=3, color=color,marker=marker)
    ellipse_args = misc._get_kwargs(ellipse_args,facecolor='none', edgecolor=color, linestyle='--', linewidth=2)
    return draw_ellipse(r_mean, r_eigdir, r_disp, ax=ax, show_center=show_center, center_args=center_args,**ellipse_args)

def draw_circle(r_center,radius,ax=None,**circle_args):
    circle = plt.Circle(r_center, radius, **_get_kwargs(circle_args,edgecolor='k',fill=False))
    if misc.exists(ax):
        ax.add_patch(circle)

def _get_rect(img_extent):
    """
    given a list of coords:
    img_extent = [left,right,bottom,top]

    returns x_left,y_bottom,width,heigh
    """
    return img_extent[0],img_extent[2],img_extent[1]-img_extent[0],img_extent[3]-img_extent[2]

def draw_rectangle(r_bottom_left=None,width=None,height=None,extent=None,ax=None,**rect_args):
    if misc.exists(extent):
        x,y,width,height = _get_rect(extent)
        r_bottom_left    = numpy.array((x,y))
    rect = plt.Rectangle(r_bottom_left,width,height,**_get_kwargs(rect_args,edgecolor='k',fill=False))
    if misc.exists(ax):
        ax.add_patch(rect)

def draw_ellipse(r_center,axes_direction,semi_axis_length,ax=None,show_center=False,center_args=None,**ellipse_args):
    """
    draws ellipse defined by its axes directions and semi-lengths

    r_center         -> (x,y) coordinates of the ellipse center
    axes_direction   -> single vector or a list of 2 vectors, each one defining each axis of the ellipse (vector 0 correspond to semi_axis_length[0], and same for vector 1)
                        if a single vector is given, it must correspond to the first semi_axis_length
    semi_axis_length -> the length in each of the axis defined in the axes_direction
    ax               -> plot axis if any
    other args ...

    returns:
        ellipse patch from matplotlib
    """
    if not misc.exists(ax):
        ax = plt.gca()
    v            = numpy.array(axes_direction[0] if (type(axes_direction) is list) else axes_direction).flatten()
    alpha        = misc.angle_uv(v,numpy.array((1,0)))*180.0/numpy.pi # angle between the first axis of the ellipse and the x-axis
    width,height = 2.0*numpy.array(semi_axis_length).flatten()
    ellipse      = pltpatches.Ellipse(r_center, width, height, angle=alpha, **ellipse_args)
    if misc.exists(ax):
        ax.add_patch(ellipse)
    if show_center:
        plot_point(r_center,ax=ax,pointArgs=center_args)
    return ellipse


def plot_point(r,label='',fmt='o', color='k', markersize=6, ax=None, pointArgs=None, verbose=True, **textArgs):
    if not misc.exists(ax):
        ax = plt.gca()
    if misc._is_numpy_array(ax):
        return [ plot_point(r,label=copy.deepcopy(label),fmt=copy.deepcopy(fmt),color=copy.deepcopy(color),markersize=copy.deepcopy(markersize),ax=a,pointArgs=copy.deepcopy(pointArgs),**textArgs) for a in ax.flatten() ]
    else:
        txt,labelArgs = _get_label(pointArgs)
        pointArgs     = _get_kwargs(pointArgs,color=color,markersize=markersize,fillstyle='none')
        fmt           = pointArgs.pop('fmt',fmt)
        l = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            l.append(ax.plot(r[0],r[1],fmt,**pointArgs))
        txt   = txt if misc.exists(txt) else ''
        label = label if misc.exists(label) else ''
        label = txt if len(label)==0 else label
        if (len(label)>0):
            valign,halign  = _get_text_anchor_alignment(r)
            padding        = textArgs.pop('pad',None)
            padding        = padding if misc.exists(padding) else labelArgs.pop('pad',numpy.zeros(2))
            labelArgs      = _get_kwargs(labelArgs,**_get_kwargs(textArgs,va=valign,ha=halign,labelcolor='k'))
            labelColor     = labelArgs.pop('color',None)
            if misc.exists(labelColor):
                labelArgs.pop('labelcolor',None)
            else:
                labelColor = labelArgs.pop('labelcolor',None)
            l.append(ax.annotate(label,numpy.asarray(r)+padding,color=labelColor,**labelArgs))
        return l

def plot_arrow(r0,r1,label='',labelpos=None,color='k',ax=None,arrowArgs=None,**textArgs):
    if ax is None:
        ax = plt.gca()
    if arrowArgs is None:
        arrowArgs = {}
    #arrowArgs_new = dict(length_includes_head=True,head_width=2,color=color)
    #arrowArgs_new.update(arrowArgs)
    arrowArgs_new = misc.set_default_kwargs(arrowArgs,length_includes_head=True,head_width=2,color=color)
    ax.arrow(r0[0],r0[1],r1[0]-r0[0],r1[1]-r0[1],**arrowArgs_new)
    if len(label)>0:
        labelpos = labelpos if labelpos else 'center'
        if labelpos == 'end':
            lp = r1
        elif labelpos == 'start':
            lp = r0
        else:
            lp = r0+(r1 - r0)/2.0 # center
        ax.text(lp[0],lp[1],label,**textArgs)

def plot_mouse(mou,t_ind,colors,fontsize=9,zero_vec_alpha=0.2,show_labels=False,vec_label='',labelpos='center',vec_fontsize=10,vec_text_args={}):
    plot_point(mou.r_nose[t_ind],'nose' if show_labels else '',fmt='^',color=colors[0],markersize=3,va='bottom',ha='left',fontsize=fontsize)
    plot_arrow(numpy.zeros(2),mou.r_nose[t_ind],'',color=numpy.insert(colors[0][:3],3,zero_vec_alpha),fontsize=fontsize)
    plot_point(mou.r_center[t_ind],'CM' if show_labels else '',fmt='v',color=colors[1],markersize=3,va='top',ha='left',fontsize=fontsize)
    plot_arrow(numpy.zeros(2),mou.r_center[t_ind],'',color=numpy.insert(colors[1][:3],3,zero_vec_alpha),fontsize=fontsize)
    plot_arrow(mou.r_center[t_ind],mou.r_nose[t_ind],vec_label,labelpos=labelpos,color=colors[2],arrowArgs=dict(head_width=1),fontsize=vec_fontsize,**vec_text_args)

def plot_errorfill(x, y, yerr, fmt='o', color=None, fill_line_color=None, color_fill=None, alpha_fill=0.3, ax=None, label=None, xlabel=None, ylabel=None, absolute_err=False, **plotArgs):
    if ax is None:
        f = plt.figure()
        ax = f.gca()
    if type(x) is list:
        x = numpy.asarray(x)
    if type(y) is list:
        y = numpy.asarray(y)
    if type(yerr) is list:
        yerr = numpy.asarray(yerr)
    yErrIsNone = type(yerr) is type(None)
    if yErrIsNone:
        ymin,ymax=numpy.nan,numpy.nan
    else:
        if numpy.isscalar(yerr) or len(yerr) == len(y):
            if absolute_err:
                ymin = yerr
                ymax = yerr
            else:
                ymin = y - numpy.abs(yerr)
                ymax = y + numpy.abs(yerr)
        elif len(yerr) == 2:
            if absolute_err:
                ymin, ymax = yerr[0],yerr[1]
            else:
                ymin, ymax = y-numpy.abs(yerr[0]),y+numpy.abs(yerr[1])
    n = y.shape[1] if y.ndim > 1 else 1
    if callable(color):
        color = color(n)
    lw = 0 if fill_line_color is None else 1
    default_colors=get_default_colors(n)
    hLine = []
    hFill = []
    if n > 1:
        if (type(label) is list) and (len(label) < n):
            label = misc.repeat_to_complete(label,n)
    for i in range(n):
        xx = x[:,i] if x.ndim > 1 else x
        yy = y[:,i] if y.ndim > 1 else y
        if not yErrIsNone:
            y1 = ymax[:,i] if ymax.ndim > 1 else ymax
            y2 = ymin[:,i] if ymin.ndim > 1 else ymin
        cc = default_colors[i] if color is None else get_color(color,i)
        if n > 1:
            ll = ('%s %d'%(label,i)) if type(label) is str else (label if label is None else label[i])
        else:
            ll = label
        hLine.append(ax.plot(xx, yy, fmt, color=cc, label=ll, **plotArgs))
        if not yErrIsNone:
            cf = hLine[i][0].get_color() if color_fill is None else color_fill
            y1[numpy.isnan(y1)]=0.0
            y2[numpy.isnan(y1)]=0.0
            hFill.append(ax.fill_between(xx, y1, y2, color=cf, alpha=alpha_fill, edgecolor=fill_line_color, linewidth=lw))
    if xlabel or ylabel:
        if (xlabel and ('$' in xlabel)) or (ylabel and ('$' in ylabel)):
            plt.matplotlib.rc('text',usetex=True)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if (xlabel and ('$' in xlabel)) or (ylabel and ('$' in ylabel)):
            plt.matplotlib.rc('text',usetex=False)
    if not yErrIsNone:
        if n == 1:
            hFill = hFill[0]
    return hLine,hFill

def plot_gradient_errorbar(ax,x,y,yErr,cmap=plt.get_cmap('copper'), label=None, **errorbarArgs):
    """
    """
    x    = numpy.asarray(x)
    y    = numpy.asarray(y)
    yErr = numpy.asarray(yErr)
    N = x.size
    colors = cmap(numpy.linspace(0,1,N))
    lh = []
    for i in range(N):
        if i == 0:
            label = label
        else:
            label = None
        lh.append(ax.errorbar(x[i], y[i], yErr[i], color=colors[i], label=label, **errorbarArgs))
    return lh

def plot_gradient_line(r, colors=None, cmap=plt.get_cmap('copper'), norm=None, linewidth=3, alpha=1.0, ax=None, label=None, linestyle='-', **linecollectparam):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates in r: r[:,0] == x and r[:,1] == y
    Optionally specify colors in the array colors
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if type(colors) is type(None):
        colors = numpy.linspace(0.0, 1.0, r.shape[0])
    # Special case if a single number:
    if not hasattr(colors, "__iter__"):  # to check for numerical input -- this is a hack
        colors = numpy.array([colors])
    colors = numpy.asarray(colors)
    if type(norm) is type(None):
        norm = plt.Normalize(0.0, 1.0)
    segments = make_line_segments_for_plot(r)
    lc = matplotlib.collections.LineCollection(segments, array=colors, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, **linecollectparam)
    lc.set_linestyle(linestyle)
    if not (label is None):
        lc.set_label(label)
    ax = plt.gca() if ax is None else ax
    ax.add_collection(lc)
    return lc

def make_line_segments_for_plot(r):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = r.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def get_color(clist,k):
    if type(clist) is list:
        return clist[k%len(clist)]
    elif type(clist) is numpy.ndarray:
        if clist.ndim == 1:
            return clist
        return clist[k%clist.shape[0],:]
    return clist

def get_default_colors(N=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    clist = []
    for c in colors:
        clist.append(c)
    if N is None:
        return clist
    else:
        if N > len(clist):
            return misc.repeat_to_complete(clist,N)
        else:
            return clist[:N]

def plot_vertical_lines(x,ax=None,yMin=None,yMax=None,**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    aymin,aymax = ax.get_ylim()
    yMin = aymin if yMin is None else yMin
    yMax = aymax if yMax is None else yMax
    if (type(x) is list) or (type(x) is numpy.ndarray):
        h = []
        for xx in x:
            h.append(plot_vertical_lines(xx, ax=ax, yMin=yMin, yMax=yMax, **plotArgs))
    else:
        h = ax.vlines(x,yMin,yMax,**plotArgs)
    ax.set_ylim((aymin,aymax))
    return h

def plot_horizontal_lines(y,ax=None,xMin=None,xMax=None,**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    axmin,axmax = ax.get_xlim()
    xMin = axmin if xMin is None else xMin
    xMax = axmax if xMax is None else xMax
    if (type(y) is list) or (type(y) is numpy.ndarray):
        h = []
        for yy in y:
            h.append(plot_horizontal_lines(yy, ax=ax, xMin=xMin, xMax=xMax, **plotArgs))
    else:
        h = ax.hlines(y,xMin,xMax,**plotArgs)
    ax.set_xlim((axmin,axmax))
    return h

def plot_trajectory_points(r,ax=None,use_scatter=False,**plotArgs):
    ax = ax if misc.exists(ax) else plt.gca()
    if r.size == 0:
        warnings.warn('plot_trajectory_points :: r is empty',RuntimeWarning)
        return None
    if use_scatter:
        return ax.scatter(r[:,0],r[:,1],**plotArgs)
    else:
        return ax.plot(r[:,0],r[:,1],**plotArgs)

def normalize_variable_to_01(x):
    min_x = numpy.nanmin(x)
    max_x = numpy.nanmax(x)
    return (x - min_x)/(max_x - min_x)

def has_arena_pic_to_plot(arena_pic):
    return ((type(arena_pic) is bool) and arena_pic) or ( (type(arena_pic) is numpy.ndarray) and (arena_pic.size > 0) )

def set_box_axis(ax,state=True):
    if misc._is_numpy_array(ax):
        for a in ax.flatten():
            set_box_axis(a,state)
    else:
        # Hide the right and top spines
        ax.spines.right.set_visible(state)
        ax.spines.top.set_visible(state)

        if not state:
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

def _add_legend_custom_order_labels(ax,order_ind,**legendArgs):
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) == 0:
        return handles
    return ax.legend([handles[i] for i in order_ind],[labels[i] for i in order_ind],**legendArgs)

def set_xlabel(ax=None,label='',**args):
    if not misc.exists(ax):
        ax = plt.gca()
    if misc.is_iterable(ax):
        return [ set_xlabel(ax=a,label=label,**args) for a in ax ]
    else:
        return ax.set_xlabel(label,**args)

def set_ylabel(ax=None,label='',**args):
    if not misc.exists(ax):
        ax = plt.gca()
    if misc.is_iterable(ax):
        return [ set_ylabel(ax=a,label=label,**args) for a in ax ]
    else:
        return ax.set_ylabel(label,**args)
