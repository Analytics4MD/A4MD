import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def barplot(df,variable,stride,ylabel,show_n_largest=10,multiplier=1,logscale=True):
    df_iter = df[df.stride==stride]
    fig1, ax1 = plt.subplots()

    df_sorted = df_iter.sort_values(variable,ascending=False).head(show_n_largest)
    num_functions = df_sorted['Function'].nunique()
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0., 1., num_functions))
    print(len(df_sorted['Function_short']))
    ax1.bar(np.arange(len(df_sorted['Function_short'])),
            df_sorted[variable]*multiplier)
    if logscale:
        ax1.set_yscale('log')
    ax1.set_xticks(np.arange(len(df_sorted['Function_short'])))
    ax1.set_xticklabels(df_sorted['Function_short'], rotation = 40, ha='right', zorder=100,fontsize=10)
    print(df_sorted['Function_short'].values)
    ax1.set_xlabel('Function Name')
    ax1.set_ylabel(ylabel)
    plt.title('{} (Stride: {})'.format(variable,stride), fontsize=10)
    
def lineplot_functions_wrt_stride(df, column_name, xlabel, ylabel, title, sortby=None):
    if sortby != None:
        df_sorted = df.sort_values(sortby)
    else:
        df_sorted = df
    num_functions = df_sorted['Function'].nunique()
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0., 1., num_functions))
            
    for i, (key, df_grp) in enumerate(df_sorted.groupby('Function')):
        func = key.split(' ')[0]
        if func != '':#'.TAU':
            if sortby != None:
                plt.plot(df_grp['stride'],
                         df_grp[column_name]/1e6,
                         label=func,
                         color=colors[i])
            else:
                plt.plot(df_grp[column_name]/1e6,
                         label=func,
                         color=colors[i])

    plt.legend(ncol=2,fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def pieplot(df,stride,variable,show_n_largest):

    df_iter = df[df.stride==stride]
    fig1, ax1 = plt.subplots()

    df_sorted = df_iter.sort_values(variable,ascending=False).head(show_n_largest)
#df_sorted = df_iter.sort_values(variable,ascending=False)
    num_functions = df_sorted['Function'].nunique()
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0., 1., num_functions))
    ax1.pie(df_sorted[variable], 
            #explode=explode, 
            #labels=labels, 
            autopct='%1.1f%%',
            #shadow=True,
            colors=colors,
            startangle=0)
    ax1.legend(labels=df_sorted['Function_short'], 
               fontsize=10, 
               ncol=1, 
               bbox_to_anchor=(1,0.5), 
               loc="center right", 
               bbox_transform=plt.gcf().transFigure)
    ax1.axis('equal')
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)
    plt.title('Stride = {}'.format(stride))
