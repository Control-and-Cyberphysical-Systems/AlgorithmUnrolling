import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

def save_scatter_tikz(
    values: np.ndarray,
    title: str,
    xlabel: str = r'$x_0$',
    ylabel: str = r'$x_1$',
    filename: str = 'scatter.txt',
    point_size: float = 2.0,
    max_points: int = 3000,
    triangles:set=set()):
    '''
    turns data into pgfplots-Scatterplot and saves it as a standalone .tex-file.

    values : (N, 4) - Column: [x0, x1, x2, meta]
             Column 0 → x-Achse, Column 1 → y-Achse, Column 3 → Farbe
    '''
    x, y, meta = values[0,:], values[1,:], values[2,:]
    # if triangles is NOT given:
    if not triangles:
        triangles = triangle_finder(values=values)
                    
    data = ''
    for t in triangles:
        data += f'\n({x[t[0]]:.2f},{y[t[0]]:.2f})[{meta[t[0]]:.2f}] ({x[t[1]]:.2f},{y[t[1]]:.2f})[{meta[t[1]]:.2f}] ({x[t[2]]:.2f},{y[t[2]]:.2f})[{meta[t[2]]:.2f}]'
    # data = '\n'.join(f'({xi:.4f}, {yi:.4f},{mi:.6f})' for xi, yi, mi in zip(df['x0'], df['x1'], df['z']))

    tex = f'''
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width = 0.2*\\textwidth,
    height = 0.2*\\textwidth,
    xlabel={{{xlabel}}},
    ylabel={{{ylabel}}},
    title={{{title}}},
    point meta min={meta.min():.4f},
    point meta max={meta.max():.4f},
    colorbar,
    colormap/jet,
    colorbar horizontal,
    colorbar style={{
        at={(0.5,-0.5)}, % Position relativ zur Achse (x, y)
        anchor=north,     % Ankerpunkt der Colorbar
        xticklabel pos=lower % Ticks unten an der Colorbar
        }},
]
\\addplot[
    patch,                     % Delaunay Triangulation
    patch type=triangle,       % Dreiecke (default)
    shader=flat,            % farbige Interpolation
    draw=none,
    line width=0pt,
    patch refines = 1,
    point meta=explicit       % z Wert wird als Metadaten Zahl benutzt
] coordinates
{{
{data}
}};
\\end{{axis}}
\\end{{tikzpicture}}
'''
    
    own_dic = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    with open(os.path.join(own_dic,filename), 'w', encoding='utf-8') as f:
        f.write(tex)

    print(f'-> {os.path.join(filename)} with ({len(x)} points)')
    return

def save_plot_tikz(model_results: dict,
                   algo_steps: list,
                   unlearned_keys:list,
                   title: str,
                   xlabel: str = r'$x_0$',
                   ylabel: str = r'$x_1$',
                   filename: str = 'scatter.txt',)->None:
    '''
    Takes data from a plot and saves them as a tikz-readable file
    '''
    comp_list = {'admm dual':0,
                 'admm base':1,
                 'gd dpc dual':2,
                 'gd dpc prim':3,
                 'dadmm':0,
                 'admm':1,
                 'dpg':2,
                 'pg':3,
                 }
    trained_on = {'admm dual':'mega dual last',
                 'admm base':'mega last',
                 'gd dpc dual':'mega dual last',
                 'gd dpc prim':'mega last',
                 'dadmm':'mega dual last',
                 'admm':'mega last',
                 'dpg':'mega dual last',
                 'pg':'mega last',
                 }
    thin = 0.5
    thick = 1.0
    plots = ''
    colors_rgb = [
    'firebrick',
    'navy',
    'darkgoldenrod',
    'forestgreen',
    'fuchsia',
    'teal',
    'darkorange',
    'black',
    'blue',
    'red',
    'green'
    ]
    markers = [
        '*',
        'diamond*',
        'triangle*',
        'square*',
        '+',
        'x',
        'star',
        'Mercedes star',
    ]
    unknown_index=0
    for model_name, model_result in model_results.items():
        data = ''
        for i, a_step in enumerate(algo_steps):
            data += f'({a_step},{model_result[i]:.2f})'
        if model_name in comp_list.keys():
            color_index = comp_list[model_name]
        else:
            color_index = len(list(comp_list.keys()))+unknown_index
            unknown_index += 1
        if trained_on[model_name] in title:
            linewidth = thick
        else:
            linewidth = thin
        if model_name in unlearned_keys:

            current_plot = f'''
            \\addplot[color={colors_rgb[color_index]},
                dashed,
                mark options={{solid}},
                mark = {markers[color_index]},
                line width ={linewidth},
                ] coordinates
                {{
                {data}
                }};
                \\addlegendentry{{{model_name}}}
            '''
        else:
            current_plot = f'''
            \\addplot[color={colors_rgb[color_index]},
                mark = {markers[color_index]},
                line width ={linewidth},
                ] coordinates
                {{
                {data}
                }};
                \\addlegendentry{{{model_name}}}
            '''
        plots+=current_plot

    full_plot = f'''
    \\begin{{tikzpicture}}
    \\begin{{axis}}[
        xlabel={{{xlabel}}},
        ylabel={{{ylabel}}},
        title={{{title}}},
        legend style={{at={{(1.02, 0.5)}},anchor = west}},
    ]
    {plots}
    \\end{{axis}}
    \\end{{tikzpicture}}
    '''
    own_dic = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    with open(os.path.join(own_dic,'tikz',filename), 'w', encoding='utf-8') as f:
        f.write(full_plot)
    return

def save_trajectory_tikz(xk2,x_s,u_preds):
    '''
    Produces a trajectoryplot in tikz for the closed loop simulations for one K
    '''
    colrs = {'dpc':'dpc',
             '0':'pgdc',
             '1':'dpgdc',
             '2':'admmc',
             '3':'dadmmc'}
    forbidden_key = 'perfect'
    name_keys = {'dpc':'',
                '0':'',
                 '1':'',
                 '2':'',
                 '3':''}
    trajectory_plots = ''
    x0_trajectories = ''
    x1_trajectories = ''
    u0_trajectories = ''
    for k in name_keys.keys():
        if k == 'dpc':
            x0 = xk2[0,:]
            x1 = xk2[1,:]
            u = u_preds[k]
        else:
            x0 = x_s[k][0,:]
            x1 = x_s[k][1,:]
            u = u_preds[k]
        traj_coords = ''
        x0_coords = ''
        x1_coords = ''
        u0_coords = ''

        for t in range(0,u.shape[1]):
            traj_coords += f'({x0[t]:.2f},{x1[t]:.2f})'
            x0_coords += f'({t:.2f},{x0[t]:.2f})'
            x1_coords += f'({t:.2f},{x1[t]:.2f})'
            u0_coords += f'({t:.2f},{u[0,t]:.2f})'
        trajectory_plots+= f'\\addplot[coror = {colrs[k]}] coordinates {{{traj_coords}}}\n'
        x0_trajectories+= f'\\addplot coordinates {{{x0_coords}}}\n'
        x1_trajectories+= f'\\addplot coordinates {{{x1_coords}}}\n'
        u0_trajectories+= f'\\addplot coordinates {{{u0_coords}}}\n'
    full_plot =f'''
    \\begin{{tikzpicture}}
    \\definecolor{{perfect}}{{RGB}}{{215,25,28}}   % rot
    \\definecolor{{dpc}}{{RGB}}{{43,131,186}}  % blau
    \\begin{{groupplot}}[
    group style = {{
        group name = trajs,
        group size = 4 by 2,
        % Abstand zwischen den Teilplots
        horizontal sep = 1.2cm,
        vertical sep   = 1.2cm,

    }},
    width = 5.5cm,
    height = 4cm,
    legend style={{font=\\small, cells={{anchor=west}}, legend columns=-1}},
    xlabel near ticks,
    ylabel near ticks,
    % tick label style={{font=\\footnotesize}},
    % label style={{font=\\footnotesize}},
    ]
    
    % trajectory plots
    \\nextgroupplot[
    title={{Trajektorie}},
    xlabel={{$x_0$}},
    ylabel={{$x_1$}},
    axis equal,
    % keine Legende hier (wird zentral im Bild erzeugt)
    hide axis,
    ]
    {trajectory_plots}

    % state x0 over time
    \\nextgroupplot[
    title={{$x_0$ over $t$}},
    xlabel={{$t$}},
    ylabel={{$x_0$}},
    % Legende erst im ersten Plot mit allen Solver‑Farben
    legend entries={{REF, PG, DPG, ADMM, D-ADMM}},
    legend to name=solverslegend,
    legend style={{at={(0.5,-0.25)},anchor=north}},
    ]
    {x0_trajectories}
    
    % state x1 over time
    \\nextgroupplot[
    title={{$x_1$ over $t$}},
    xlabel={{$t$}},
    ylabel={{$x_1$}},
    % Legende wird nicht wiederholt
    hide legend,
    ]
    {x1_trajectories}
    
    \nextgroupplot[
    title={{$u$}},
    xlabel={{$t$}},
    ylabel={{$u$}},
    % Legende wird hier nicht noch einmal gezeigt
    hide legend,
    ]
    {u0_trajectories}

    \\end{{groupplot}}

    % ---- Legende separat unterhalb des gesamten Bildes ----
    \\ref{{solverslegend}}
    
    \\end{{tikzpicture}}
    '''
    own_dic = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    with open(os.path.join(own_dic,'tikz',f'trajectory_tikz{np.random.randint(100,10000)}.txt'), 'w', encoding='utf-8') as f:
        f.write(full_plot)
    return

def triangle_finder(values: np.ndarray)->set:
    '''
    To make the scatterplot tikz-readable, it needs to be transformed into a triangulated plot over all parameter space.
    This function finds the triangles that are spanned between the points and hands it to the plot function
    '''
    x, y= values[0,:], values[1,:]
    df = pd.DataFrame(values.T,columns=['x0','x1'])
    df = df.sort_values(['x0','x1']).reset_index(drop=True)
    pts = list(range(0,x.shape[0]))
    # calculate edge length of the triangles
    min_x = 1000#
    min_y = 1000
    for i in range(1,len(pts)):
        poss_x = (x[0]-x[i])**2
        poss_y = (y[0]-y[i])**2
        if poss_x > 0:
            min_x = min(poss_x,min_x)
        if poss_y > 0:
            min_y = min(poss_y,min_y)
    # find the points in avg reach*sqrt2
    edges = []
    for i in range(0,len(pts)):
        found_pts = 0
        j = 0
        while found_pts < 8 and j < len(pts):
            if (x[i]-x[j])**2 < 1.01*min_x and (y[i]-y[j])**2 < 1.01*min_y and j != i:
                f_entry = max(i,j)
                s_entry = min(i,j)
                if (f_entry,s_entry) not in edges:
                    edges.append((f_entry,s_entry))
                found_pts +=1
            j+=1
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    triangles = set()
    tria_geoms = []
    # edge_scores = {}
    for u, v in edges:
        # Gemeinsame Nachbarn von u und v
        common = adj[u] & adj[v]
        for w in common:
            tri_pot = Polygon([(x[u],y[u]),
                                (x[v],y[v]),
                                (x[w],y[w])])
            no_intersects = True
            for t in tria_geoms:
                if tri_pot.intersection(t).area> 0.1:
                    no_intersects = False
                    break
            if no_intersects:
                tri = tuple(sorted((u,v, w)))
                triangles.add(tri)
                tria_geoms.append(tri_pot)
    return triangles