import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import numpy as np

def plot_avg_opt(df,metric='approxUB'):
    df0=df[df['r']==1]
    sns.set_style('whitegrid')
    sns.lineplot(x='N', y=metric, data=df0, marker='o', linestyle='', err_style='bars')
    if metric == 'approxUB':
        plt.ylabel('avg/max')
    elif metric == 'gapLB':
        plt.ylabel('avg/min-1')
    
def plot_N_log1r_avgmetric(df,metric='approxUB'):
    df_mean = df.groupby(['log(1/r)', 'N'])[metric].mean().reset_index()
    sns.set_style('whitegrid')
    sns.lineplot(data=df_mean, x='log(1/r)', y=metric, hue='N')

def plot_3Dmesh(df,metric='approxUB'):
    # Since wireframe plots require structured grid data, create a meshgrid
    N_unique = np.sort(df['N'].unique())
    log1r_unique = np.sort(df['log(1/r)'].unique())
    N, log1r = np.meshgrid(N_unique, log1r_unique)

    # Initialize an empty matrix for target-metric result values
    results = np.empty(N.shape)

    for i in range(len(log1r_unique)):
        for j in range(len(N_unique)):
            result_value = np.mean(df[(df['N'] == N_unique[j]) & (df['log(1/r)'] == log1r_unique[i])][metric].values)
            results[i, j] = result_value
            
    # Plotting
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Plotting the wireframe
    
    ax.plot_wireframe(N, log1r, results, color='pink', alpha=0.2)    
    
    df_mean = df.groupby(['log(1/r)', 'N'])[metric].mean().reset_index()
    value_targets = np.linspace(min(df_mean[metric])+0.03, max(df_mean[metric])-0.03, 50)
    value_targets = np.round(value_targets,3)
    
    epsilon = 2e-4
    
    # Plotting lines for each value target
    # target_df_dict = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(value_targets)))
    for i, target in enumerate(value_targets):
        target_df = df[(df[metric] >= target - epsilon) & (df[metric] <= target + epsilon)]
        # if target_df['N'].nunique() >= 2:
        #     target_df_dict[target] = target_df
        target_df_mean = target_df.groupby(['N'])['log(1/r)'].mean().reset_index().sort_values(by=['N'])
        if not target_df_mean.empty:
            ax.plot(target_df_mean['N'], target_df_mean['log(1/r)'], np.full(target_df_mean.shape[0], target), color=colors[i], linewidth=1)

    # Adding a color bar to indicate the values
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=value_targets[0], vmax=value_targets[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, aspect=10, shrink=0.5, label=metric)

    ax.set_xlabel('N')
    ax.set_ylabel('log(1/r)')
    ax.set_zlabel(metric)
    plt.show()
    

def plot_fitting(mesh,df,model,args,metric='approxUB'):
    # Since wireframe plots require structured grid data, create a meshgrid
    N_unique = np.sort(df['N'].unique())
    log1r_unique = np.sort(df['log(1/r)'].unique())
    N, log1r = np.meshgrid(N_unique, log1r_unique)

    # Initialize an empty matrix for target-metric result values
    results = np.empty(N.shape)

    for i in range(len(log1r_unique)):
        for j in range(len(N_unique)):
            result_value = np.mean(df[(df['N'] == N_unique[j]) & (df['log(1/r)'] == log1r_unique[i])][metric].values)
            results[i, j] = result_value
            
    # Plotting
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('N')
    ax.set_zlabel(metric)
    
    if mesh:
        # Plotting the wireframe
        ax.plot_wireframe(N, log1r, results, color='pink', alpha=0.2)
        ax.set_ylabel('log(1/r)')
    else:
        ax.set_ylabel('depth')
    
    # 设置颜色映射
    cmap = get_cmap('viridis')
    if mesh:
        p_list = [int(p) for p in range(0,16)]
    else:
        p_list = [int(p) for p in range(0,101)]
    norm = Normalize(vmin=min(p_list), vmax=max(p_list))
    
    for p in p_list:
        x = N_unique.tolist()
        z = [model(N,p,*args) for N in N_unique.tolist()]
        if mesh:
            y = [np.log((2*p+1)**2)]*len(N_unique.tolist())  # 根据p值计算y，这里简化为直接使用log表达式
        else:
            y = [p]*len(N_unique.tolist())
        color = cmap(norm(p))
        ax.plot3D(x, [y]*len(x), z, color=color)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, aspect=10, shrink=0.5, label='p value', alpha=0.8)


def plot_fitting_N(N_list,model,params):
    p_list = [p for p in range(51)]
    for N in N_list:
        plt.plot([np.log((2*p+1)**2) for p in p_list],[model(N,p,*params) for p in p_list],label = str(N))
    plt.legend()
    
    
def plot_predict_sample(sample_data,model,params,metric='approx',plot_type="line"):
    p_list = np.sort(sample_data['depth'].unique()).tolist()
    N_list = np.sort(sample_data['N'].unique()).tolist()
    palette = sns.color_palette("deep", n_colors=len(p_list))
    for i in range(len(p_list)):
        data = sample_data[sample_data['depth'] == p_list[i]].copy()
        if plot_type == "line":
            sns.lineplot(x='N', y=metric, data=data, marker='o', linestyle='', err_style='bars',color=palette[i], label=f'Depth: {p_list[i]}, Sampled',alpha=0.7)
        elif plot_type == "scatter":
            sns.scatterplot(x='N', y=metric, data=data, color=palette[i], label=f'Depth: {p_list[i]}, Sampled',alpha=0.5)
        sns.lineplot(x=N_list,y=[model(N,p_list[i],*params) for N in N_list],color=palette[i], label=f'Depth: {p_list[i]}, Fitted Bound')

    plt.legend(bbox_to_anchor=(1., .5), loc='center left')
    
    
def plot_comparison(data,metric1,metric2):
    
    sns.scatterplot(data=data, x=metric1, y=metric2, hue="depth", style="N")
    x_min = data[metric1].min() - 0.01
    x_max = data[metric1].max() + 0.01
    y_min = data[metric2].min() - 0.01
    y_max = data[metric2].max() + 0.01

    lim = (min(x_min, y_min), max(x_max, y_max))

    plt.xlim(lim)
    plt.ylim(lim)

    plt.plot(lim, lim, linestyle='--', color='gray')


def plot_opt_density(df):
    df['log(opt_density)'] = np.log(df['opt_density'])
    sns.set_style('whitegrid')
    sns.lineplot(x='N', y='log(opt_density)', data=df, marker='o', linestyle='', err_style='bars')
    
