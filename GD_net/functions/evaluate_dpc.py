import torch
import numpy as np
from functions.helperfunctions import *
from functions.save_as_tikz import save_scatter_tikz,triangle_finder,save_plot_tikz,save_trajectory_tikz
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
def evaluate_dpc(self,others:list,tikz:bool=False,printer:bool=False):
    def to_cvxopt_matrix(a):
        a = np.asarray(a, dtype=np.float64)
        a = np.asfortranarray(a)
        return matrix(a, tc='d')

    # Set the learning to 'off'
    torch.autograd.set_detect_anomaly(True)
    self.model = self.load_model(self.model)
    self.model.eval()
    for other in others:
        other.model = other.load_model(other.model)
        other.model.eval()
        if 'dpc' == other.loss_fcn.name:
            other.get_loss_function(run_dict={'loss function':'unsupervised dual last'})
    if 'dpc' == self.loss_fcn:
        self.get_loss_function(run_dict={'loss function':'unsupervised dual last'})
    # Set up an objectlist with the models and self included
    models = others
    models.insert(0,self)
    # set the evaluation name
    eval_name = '|'
    for m in models:
        eval_name = eval_name+ f'|{m.model.name}|'
    eval_name = eval_name +'|'
    # Stack learning and test variables
    X = torch.vstack((self.x_train,self.x_test))
    Y = torch.vstack((self.y_train,self.y_test))

    # load the dpc parameters
    own_dic = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{self.N}','QP_parameters_SS.mat'))
    H_DPC = dpc_parameters['H']
    H_dpc = to_cvxopt_matrix(0.5*(H_DPC+H_DPC.T))
    F_dpc = dpc_parameters['F']
    d_dpc = dpc_parameters['d']
    G_dpc = to_cvxopt_matrix(dpc_parameters['G'])
    E_dpc = dpc_parameters['E']
    GHG_dpc = to_cvxopt_matrix(dpc_parameters['G']@np.linalg.inv(H_DPC)@dpc_parameters['G'].T)
    EGHF_dpc = dpc_parameters['E']+dpc_parameters['G']@np.linalg.inv(H_DPC)@dpc_parameters['F']
    G_dual_dpc = to_cvxopt_matrix(-np.eye(dpc_parameters['G'].shape[0]))
    h_dual_dpc = to_cvxopt_matrix(np.zeros(dpc_parameters['G'].shape[0]))

    # Set up the standard  PGD and DPGD
    local_dpg = DPG(H=np.array(H_dpc),F =np.array(F_dpc),G = np.array(G_dpc),M=np.array(GHG_dpc),
                    N=np.array(EGHF_dpc),C=d_dpc,n_steps=self.opt_steps)
    local_pg = PG(H=np.array(H_dpc),F =np.array(F_dpc),G = np.array(G_dpc),
                  d = np.array(d_dpc),E = np.array(E_dpc),n_steps=self.opt_steps)

    # Set up the standard ADMM and D-ADMM
    local_admm = BADMM(H=np.array(H_dpc),F =np.array(F_dpc),G = np.array(G_dpc),
                       d = np.array(d_dpc),E = np.array(E_dpc),n_steps=self.opt_steps)
    local_dadmm =BDADMM(H=np.array(H_dpc),F =np.array(F_dpc),G = np.array(G_dpc),M=np.array(GHG_dpc),
                        N=np.array(EGHF_dpc),C=d_dpc,n_steps=self.opt_steps)

    # Plot dependent variables
    # produces one big plot to compare the predictions of every model in every other models' cost function
    # - As well as the unlearned functions
    batch_size = 1
    colors = list(clrs.CSS4_COLORS.keys())[19:]
    fig, axes = plt.subplots(nrows = len(models)+5,ncols= len(models)+2, figsize=(6*(len(models)+5), 9*(len(models)+2)),constrained_layout=True)
    values = {'x':np.zeros((X.shape[0],self.n)),
              'perfect':np.zeros((X.shape[0],len(models))),
              'pg':{},
              'dpg':{},
              'admm':{},
              'dadmm':{}}
    values['pg']['dist prim']= np.zeros((X.shape[0]))
    values['dpg']['dist prim']= np.zeros((X.shape[0]))  
    values['admm']['dist prim']= np.zeros((X.shape[0]))
    values['dadmm']['dist prim']= np.zeros((X.shape[0]))
    values['pg']['dist dual']= np.zeros((X.shape[0]))
    values['dpg']['dist dual']= np.zeros((X.shape[0]))
    values['admm']['dist dual']= np.zeros((X.shape[0]))
    values['dadmm']['dist dual']= np.zeros((X.shape[0]))

    # save all the devices and dtypes for consistency
    devices = {}
    dtypes = {}
    for m,model in enumerate(models):
        devices[f'{m}'] = model.device
        dtypes[f'{m}'] = model.dtypus
        values[f'{m}'] = {}
        values[f'{m}']['dist prim'] = np.zeros(X.shape[0])
        values[f'{m}']['dist dual'] = np.zeros(X.shape[0])
        values['pg'][f'{m},{model.loss_fcn.name}'] = np.zeros(X.shape[0])
        values['dpg'][f'{m},{model.loss_fcn.name}'] = np.zeros(X.shape[0])
        values['admm'][f'{m},{model.loss_fcn.name}'] = np.zeros(X.shape[0])
        values['dadmm'][f'{m},{model.loss_fcn.name}'] = np.zeros(X.shape[0])
        for n,nodel in enumerate(models):
            values[f'{m}'][f'{n},{nodel.loss_fcn.name}'] = np.zeros(X.shape[0])
                                    
    # Start the statespace loop -> One sample at a time!
    to_print = int(np.ceil(X.shape[0]/1000.0))
    for i in range(0,X.shape[0]):
        batch_idx = i
        X_batch = X[batch_idx].to(self.device)
        Y_batch = Y[batch_idx].to(self.device)#.cpu().detach().numpy()
        if batch_size == 1:
            X_batch = X_batch.reshape(1, X.shape[1])
            Y_batch = Y_batch.reshape(1, Y.shape[1])

        # Make a prediction by our model
        predictions = {}
        for m,model in enumerate(models):
            predictions[f'{m}'] = model.model(X_batch)

        # Use CVXOPT for a 'better' solution
        xdpc = np.array(X_batch.detach().cpu())
        q_dpc = to_cvxopt_matrix(EGHF_dpc@xdpc.T + d_dpc)
        sol_dpc = solvers.qp(P = GHG_dpc,q = q_dpc,G=G_dual_dpc,h=h_dual_dpc)
        opt_y = np.array(sol_dpc['x'])
        opt_y_torch = torch.tensor(opt_y,device=X_batch.device,dtype = X_batch.dtype)
        f_cvx = to_cvxopt_matrix(F_dpc@xdpc.T)
        h_cvx = to_cvxopt_matrix(E_dpc@xdpc.T + d_dpc)
        # Get the perfect primal solution
        sol_prim = solvers.qp(P=H_dpc,q = f_cvx,G=G_dpc,h=h_cvx)
        opt_y_prim = np.array(sol_prim['x'])
        opt_y_prim_torch = torch.tensor(opt_y_prim,device=X_batch.device,dtype = X_batch.dtype)
        # Save the values
        values['x'][i,:]= np.array(X_batch[0,:].detach().cpu()).reshape(self.n,)
        opt_dual = opt_y_torch.unsqueeze(0).expand(batch_size,predictions['0'][2].shape[1],predictions['0'][2].shape[2]).to(devices['0'])
        opt_prim = opt_y_prim_torch.unsqueeze(0).expand(batch_size,predictions['0'][0].shape[1],predictions['0'][0].shape[2]).to(devices['0'])
        norm_dual = np.linalg.norm(opt_dual.detach().cpu())
        dual_tol = 1e-1
        norm_prim = np.linalg.norm(opt_prim.detach().cpu())
        prim_tol = 1e-1

        # unlearned predictions
        normal_gd = local_dpg(X_batch.detach().cpu().numpy())
        primal_gd = local_pg(X_batch.detach().cpu().numpy())
        prim_admm = local_admm(X_batch.detach().cpu().numpy())
        dual_admm = local_dadmm(X_batch.detach().cpu().numpy())
        # learned models
        for m,model in enumerate(models):
            values['perfect'][i,m] = float(model.loss_fcn(z=opt_prim,
                                            z_hat=opt_prim,
                                            mu=opt_dual,
                                            z_opt=opt_y_torch.unsqueeze(-1),
                                            x=X_batch).detach().cpu())
            values['dpg'][f'{m},{model.loss_fcn.name}'][i] = float(model.loss_fcn(z=torch.tensor(normal_gd[0]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_hat=torch.tensor(normal_gd[1]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    mu=torch.tensor(normal_gd[2][:,0:predictions['0'][2].shape[1]]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_opt=opt_y_torch.unsqueeze(-1),
                                                                    x=X_batch).detach().cpu())
            values['pg'][f'{m},{model.loss_fcn.name}'][i] = float(model.loss_fcn(z=torch.tensor(primal_gd[0]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_hat=torch.tensor(primal_gd[1]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    mu=torch.tensor(primal_gd[2][:,0:predictions['0'][2].shape[1]]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_opt=opt_y_torch.unsqueeze(-1),
                                                                    x=X_batch).detach().cpu())
            values['admm'][f'{m},{model.loss_fcn.name}'][i] = float(model.loss_fcn(z=torch.tensor(prim_admm[0]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_hat=torch.tensor(prim_admm[1]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    mu=torch.tensor(prim_admm[2][:,0:predictions['0'][2].shape[1]]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_opt=opt_y_torch.unsqueeze(-1),
                                                                    x=X_batch).detach().cpu())
            values['dadmm'][f'{m},{model.loss_fcn.name}'][i] = float(model.loss_fcn(z=torch.tensor(dual_admm[0]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_hat=torch.tensor(dual_admm[1]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    mu=torch.tensor(dual_admm[2][:,0:predictions['0'][2].shape[1]]).to(device = devices[f'{m}'],dtype = dtypes[f'{m}']),
                                                                    z_opt=opt_y_torch.unsqueeze(-1),
                                                                    x=X_batch).detach().cpu())
            if  norm_dual > dual_tol:
                values[f'{m}']['dist dual'][i] = np.min([(np.linalg.norm(np.array(predictions[f'{m}'][2][:,:,-1].detach().cpu())-np.array(opt_y).T)/np.linalg.norm(opt_y)),1000])
                values['dpg']['dist dual'][i] = np.linalg.norm(normal_gd[2][0,:,-1]-np.array(opt_y).T)/np.linalg.norm(opt_y)
                values['pg']['dist dual'][i] = np.linalg.norm(primal_gd[2][0,:,-1]-np.array(opt_y).T)/np.linalg.norm(opt_y)
                values['admm']['dist dual'][i] = np.linalg.norm(prim_admm[2][0,:,-1]-np.array(opt_y).T)/np.linalg.norm(opt_y) 
                values['dadmm']['dist dual'][i] = np.linalg.norm(dual_admm[2][0,:,-1]-np.array(opt_y).T)/np.linalg.norm(opt_y) 
            if  norm_prim > prim_tol:
                values[f'{m}']['dist prim'][i] = np.min([(np.linalg.norm(np.array(predictions[f'{m}'][0][:,:,-1].detach().cpu())-np.array(opt_y_prim).T)/np.linalg.norm(opt_y_prim)),1000])
                values['dpg']['dist prim'][i] = np.linalg.norm(normal_gd[0][0,:,-1]-np.array(opt_y_prim).T)/np.linalg.norm(opt_y_prim)
                values['pg']['dist prim'][i] = np.linalg.norm(primal_gd[0][0,:,-1]-np.array(opt_y_prim).T)/np.linalg.norm(opt_y_prim)
                values['admm']['dist prim'][i] = np.linalg.norm(prim_admm[0][0,:,-1]-np.array(opt_y_prim).T)/np.linalg.norm(opt_y_prim)
                values['dadmm']['dist prim'][i] = np.linalg.norm(dual_admm[0][0,:,-1]-np.array(opt_y_prim).T)/np.linalg.norm(opt_y_prim)
            for n,nodel in enumerate(models):
                values[f'{m}'][f'{n},{nodel.loss_fcn.name}'][i] =float(nodel.loss_fcn(z=predictions[f'{m}'][0],
                                                                                z_hat=predictions[f'{m}'][1],
                                                                                mu=predictions[f'{m}'][2],
                                                                                z_opt=opt_y_torch.unsqueeze(-1),
                                                                                x=X_batch).detach().cpu())
        if (i)%to_print == 0 and printer:
            update_progress_bar(i,X.shape[0]*3,f'Eval: {eval_name} -> Parameter space evaluation')
    update_progress_bar(X.shape[0]-1,X.shape[0]*3,f'Eval: {eval_name}-> Parameter space evaluation\n')
    # Simulation of a closed loop for 20 seconds
    n_time = int(np.ceil(20/self.T_s))
    # Setup the starting position of the simulation
    x0 = np.array([20,-4])
    xk1 = np.zeros([2,n_time+1])
    xk1[:,0] = x0
    xk2 = np.zeros([2,n_time+1])
    xk2[:,0] = x0

    u_preds = {}
    x_s = {}
    for i in range(0,len(models)):
        u_preds[f'{i}'] = np.zeros((self.m*self.N,n_time))
        x_s[f'{i}'] = np.zeros([2,n_time+1])
        x_s[f'{i}'][:,0] = x0
    u_opt = np.zeros((self.m*self.N,n_time))
    u_dpc = np.zeros((self.m*self.N,n_time))
    ghg = self.G@np.linalg.inv(self.H)@self.G.T
    a_std = 1/np.max(np.linalg.eigh(ghg).eigenvalues)
    efgh = self.G@np.linalg.inv(self.H)@self.F+self.E
    P = matrix(ghg)
    G = matrix(-np.eye(self.G.shape[0]))
    h = matrix(np.zeros(self.G.shape[0]))
    preds = {}
    u_preds = {}
    for m in range(0,len(models)):
        preds[f'{m}'] = {}
        u_preds[f'{m}'] = np.zeros((self.m*self.N,n_time))
    for i in range(0,n_time):
        # Ask the nn for predictions and find the inputs inside of them
        for m,model in enumerate(models):
            preds[f'{m}'][i] = model.model(torch.tensor(x_s[f'{m}'][:,i].reshape((1,self.n)),device=X_batch.device,dtype=X_batch.dtype))
            u_preds[f'{m}'][:,i] = np.array(preds[f'{m}'][i][0][0,0:self.N*self.m,-1].detach().cpu()).reshape(self.m*self.N)

        # normal optimization
        q = matrix((efgh)@xk1[:,i]+self.d.reshape(self.d.shape[0]))
        sol = solvers.qp(P=P,q=q,G=G,h=h)
        Fx = xk1[:,i] @ self.F.T
        Glam = self.G.T@np.array(sol['x'])
        rhs = Fx.reshape((Fx.shape[0],1))+ Glam
        u = (-np.linalg.inv(self.H)@rhs)[0:self.m*self.N]

        # Optimize over dpc
        Fx_dpc = to_cvxopt_matrix(xk2[:,i]@F_dpc.T)
        h_dpc = to_cvxopt_matrix((E_dpc@xk2[:,i] + d_dpc.T).reshape(E_dpc.shape[0]))
        sol_dpc = solvers.qp(P = H_dpc,q = Fx_dpc,G=G_dpc,h=h_dpc)
        u_dpc[:,i] = np.array(sol_dpc['x'][0:self.m*self.N]).reshape(self.N*self.m)
        u_opt[:,i] = u.reshape(self.m*self.N)

        xk1[:,i+1] = (self.A@xk1[:,i]+(self.B@u[0:self.m]).reshape(self.n,)).reshape(self.n)
        xk2[:,i+1] = (self.A@xk2[:,i]+self.B@u_dpc[0:self.m,i]).reshape(self.n)
        for mo in range(0,len(models)):
            x_s[f'{mo}'][:,i+1] = (model.A@x_s[f'{mo}'][:,i] + model.B@u_preds[f'{mo}'][0:self.m,i] ).reshape(self.n)
        update_progress_bar(n_time+i,3*n_time,f'Eval: {eval_name}-> Closed Loop simulation')
    update_progress_bar(2*n_time-1,3*n_time,f'Eval: {eval_name}-> Closed Loop simulation\n')
        
    u_preds['perfekt']  = u_opt
    u_preds['dpc'] = u_dpc

    cmaps = ['viridis',
             'spring',
             'hot',
             'plasma',
             'berlin',
             'rainbow',
             'vanimo',
             'summer',
             'winter',
             'copper',
             'inferno', 
             'magma', 
             'cividis',
             'turbo',
             'ocean'
             ]
    unlearned = ['pg','dpg','admm','dadmm']
    c_bar_coords = {}
    size = 500
    
    update_progress_bar(12,18,f'Eval: {eval_name}-> Plotting')
    # First row: Perfect values and trajectories
    for m, model in enumerate(models):
        perfect_values = values['perfect'][:,m]
        axes[0,m].set_title(f'Perfect |{model.loss_fcn.name}|{np.mean(perfect_values):.2f}')
        c_bar_coords[(0,m)] = axes[0,m].scatter(values['x'][:,0],values['x'][:,1],c=perfect_values,cmap=cmaps[m],s=size)
        axes[0,m].set_xlabel('$x_{{0}}$')
        axes[0,m].set_ylabel('$x_{{1}}$')
        fig.colorbar(c_bar_coords[(0,m)],ax=axes[0,m])

    update_progress_bar(12,18,f'Eval: {eval_name}-> Plotting')
    # compare the results of different cost functions
    for m,model in enumerate(models):
        for n,nodel in enumerate(models):
            m_vals = values[f'{m}'][f'{n},{nodel.loss_fcn.name}']
            axes[m+1,n].set_title(f'{model.model.name} |{nodel.loss_fcn.name}|: {np.mean(m_vals):.2f}')
            c_bar_coords[(m+1,n)] = axes[m+1,n].scatter(values['x'][:,0],values['x'][:,1],c=m_vals,cmap=cmaps[n],s=size)
            axes[m+1,n].set_xlabel('$x_{{0}}$')
            axes[m+1,n].set_ylabel('$x_{{1}}$')
            fig.colorbar(c_bar_coords[(m+1,n)],ax=axes[m+1,n])
    len_mods = len(models)
    for u,ulm in enumerate(unlearned):
        for m,model in enumerate(models):
            unlearned_vals = values[ulm][f'{m},{model.loss_fcn.name}']
            axes[1+len_mods+u,m].set_title(f'{ulm} |{model.loss_fcn.name}|: {np.mean(unlearned_vals):.2f}')
            c_bar_coords[(1+len_mods+u,m)] = axes[1+len_mods+u,m].scatter(values['x'][:,0],values['x'][:,1],c=unlearned_vals,cmap=cmaps[m],s=size)
            axes[1+len_mods+u,m].set_xlabel('$x_{{0}}$')
            axes[1+len_mods+u,m].set_ylabel('$x_{{1}}$')
            fig.colorbar(c_bar_coords[(1+len_mods+u,m)],ax=axes[1+len_mods+u,m])
    
    update_progress_bar(13,18,f'Eval: {eval_name}-> Plotting')
    # compare the results of different cost functions
    for m,model in enumerate(models):
        dual_dist_vals = values[f'{m}']['dist dual']
        prim_dist_vals = values[f'{m}']['dist prim']
        axes[m+1,len(models)].set_title(f'Difference to $\\mu^{{\\ast}}$ | {model.model.name} |: {np.mean(dual_dist_vals):.2f}')
        c_bar_coords[(m+1,len(models))] = axes[m+1,len(models)].scatter(values['x'][:,0],values['x'][:,1],c=dual_dist_vals,cmap=cmaps[len(models)],s=size)
        axes[m+1,len(models)].set_xlabel('$x_{{0}}$')
        axes[m+1,len(models)].set_ylabel('$x_{{1}}$')
        fig.colorbar(c_bar_coords[(m+1,len(models))],ax=axes[m+1,len(models)])
        axes[m+1,len(models)+1].set_title(f'Difference to $z^{{\\ast}}$ | {model.model.name} |: {np.mean(prim_dist_vals):.2f}')
        c_bar_coords[(m+1,len(models)+1)] = axes[m+1,len(models)+1].scatter(values['x'][:,0],values['x'][:,1],c=prim_dist_vals,cmap=cmaps[len(models)+1],s=size)
        axes[m+1,len(models)+1].set_xlabel('$x_{{0}}$')
        axes[m+1,len(models)+1].set_ylabel('$x_{{1}}$')
        fig.colorbar(c_bar_coords[(m+1,len(models)+1)],ax=axes[m+1,len(models)+1])


    update_progress_bar(14,18,f'Eval: {eval_name}-> Plotting')
    len_mods = len(models)
    for u,ulm in enumerate(unlearned):
        dual_dist_vals = values[ulm]['dist dual']
        prim_dist_vals = values[ulm]['dist prim']
        axes[1+len_mods+u,len(models)].set_title(f'Difference to $\\mu^{{\\ast}}$ | {ulm} |: {np.mean(dual_dist_vals):.2f}')
        c_bar_coords[(1+len_mods+u,len(models))] = axes[1+len_mods+u,len(models)].scatter(values['x'][:,0],values['x'][:,1],c=dual_dist_vals,cmap=cmaps[len(models)],s=size)
        axes[1+len_mods+u,len(models)].set_xlabel('$x_{{0}}$')
        axes[1+len_mods+u,len(models)].set_ylabel('$x_{{1}}$')
        fig.colorbar(c_bar_coords[(1+len_mods+u,len(models))],ax=axes[1+len_mods+u,len(models)])

        axes[1+len_mods+u,len(models)+1].set_title(f'Difference to $z^{{\\ast}}$ | {ulm} |: {np.mean(prim_dist_vals):.2f}')
        c_bar_coords[(1+len_mods+u,len(models)+1)] = axes[1+len_mods+u,len(models)+1].scatter(values['x'][:,0],values['x'][:,1],c=prim_dist_vals,cmap=cmaps[len(models)+1],s=size)
        axes[1+len_mods+u,len(models)+1].set_xlabel('$x_{{0}}$')
        axes[1+len_mods+u,len(models)+1].set_ylabel('$x_{{1}}$')
        fig.colorbar(c_bar_coords[(1+len_mods+u,len(models)+1)],ax=axes[1+len_mods+u,len(models)+1])

    update_progress_bar(15,18,f'Eval: {eval_name}-> Plotting')
    track_colors = ['peru',
                    'forestgreen',
                    'cyan',
                    'crimson',
                    'darkgoldenrod',
                    'lime',
                    'teal',
                    'darkturquoise']
    # plot the trajectories from the closed loop sim
    axes[0,len_mods].set_title(f'Trajectories')
    axes[0,len_mods].plot(xk1[0,:],xk1[1,:],c = 'navy',label='opt',linewidth= 1,linestyle = '--')
    axes[0,len_mods].plot(xk2[0,:],xk2[1,:],c = 'firebrick',label='dpc opt',linewidth= 1,linestyle = '--')
    for m,model in enumerate(models):
        axes[0,len_mods].plot(x_s[f'{m}'][0,:],x_s[f'{m}'][1,:],c = track_colors[m],label = model.model.name, linewidth = 2)
    axes[0,len_mods].plot([-25,-25],[ 5,-5],c = 'black',linestyle = '--')
    axes[0,len_mods].plot([ 25, 25],[ 5,-5],c = 'black',linestyle = '--')
    axes[0,len_mods].plot([-25, 25],[-5,-5],c = 'black',linestyle = '--')
    axes[0,len_mods].plot([-25, 25],[ 5, 5],c = 'black',linestyle = '--')
    axes[0,len_mods].set_xlim([-26,26])
    axes[0,len_mods].set_ylim([-5.2,5.2])
    axes[0,len_mods].scatter(x0[0],x0[1],c='black',marker='X',label='start')
    axes[0,len_mods].scatter(0,0,c='red',marker='X',label='goal')
    axes[0,len_mods].legend()
    axes[0,len_mods].set_xlabel('$x_{{0}}$')
    axes[0,len_mods].set_ylabel('$x_{{1}}$')

    axes[0,len_mods+1].set_title('Input Trajectories')
    # Plot the input trajectories over time

    update_progress_bar(16,18,f'Eval: {eval_name}-> Plotting')
    for i in range(0,self.m):
        c = float(i+1)/float(self.m+1)
        if i == 0:
            for m,model in enumerate(models):
                axes[0,len_mods+1].plot(u_preds[f'{m}'][i,:],c=track_colors[m],label=f'$u_{{{model.model.name}}}$')
            axes[0,len_mods+1].plot(u_preds[f'perfekt'][i,:],c='navy',label='$u_{{\\ast}}$')
            axes[0,len_mods+1].plot(u_preds[f'dpc'][i,:],c='firebrick',label='$u_{{dpc}}$')
        else:
            for m,model in enumerate(models):
                axes[0,len_mods+1].plot(u_preds[f'{m}'][i,:],c=track_colors[m],label=f'$u_{{{model.model.name}}}$')
            axes[0,len_mods+1].plot(u_preds[f'perfekt'][i,:],c='navy')
            axes[0,len_mods+1].plot(u_preds[f'dpc'][i,:],c='firebrick')
    axes[0,len_mods+1].plot([0,n_time*self.T_s],[-1,-1],c = 'black',linestyle='--')
    axes[0,len_mods+1].plot([0,n_time*self.T_s],[ 1, 1],c = 'black',linestyle='--')
    axes[0,len_mods+1].set_ylim([-1.5,1.5])
    axes[0,len_mods+1].legend()
    own_dic = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    fig.savefig(os.path.join(own_dic,'evaluation_results',f'test_{self.model.num_pars}.png'))
    plt.close(fig)

    update_progress_bar(17,18,f'Eval: {eval_name}-> Plotting\n')
    # Save the figures in tikz to be used in latex
    if tikz:
        key_title = {
               'dist prim': {'title':'\\zb^{{\\ast}}-\\zb^{{(N)}}\\|_{{2}}$',
                             'save_name':'prim_dist'},
               'dist dual': {'title':'\\mub^{{\\ast}}-\\mub^{{(N)}}\\|_{{2}}$',
                             'save_name':'dual_dist'}
            }
        for m,model in enumerate(models):
            key_title[f'{m},{model.loss_fcn.name}'] = {'title':model.loss_fcn.name,
                                                       'save_name':f'{m}_{model.loss_fcn.name}'}
        plot_triangles = triangle_finder(np.vstack((values['x'][:,0],values['x'][:,1])))
        for m,model in enumerate(models):
            for key,item in values[f'{m}'].items():
                key_title_loc = key_title[key]['title']
                key_savename = key_title[key]['save_name'].replace(' ','_')
                secure_name = model.model.name.replace(' ','_')
                save_scatter_tikz(values=np.vstack((values['x'][:,0],values['x'][:,1],item)),
                                title=f' ${model.model.name}\\ | {key_title_loc}',
                                filename=os.path.join('tikz',f'{secure_name}_{key_savename}.txt'),
                                triangles = plot_triangles)
        for ulm in unlearned:
            for key,item in values[ulm].items():
                key_title_loc = key_title[key]['title']
                key_savename = key_title[key]['save_name'].replace(' ','_')
                secure_name = model.model.name.replace(' ','_')
                save_scatter_tikz(values=np.vstack((values['x'][:,0],values['x'][:,1],item)),
                                title=f' ${ulm}\\ | {key_title_loc}',
                                filename=os.path.join('tikz',f'{secure_name}_{key_savename}.txt'),
                                triangles = plot_triangles)
        save_trajectory_tikz(xk2,x_s,u_preds)
    # Save the average values to be used in the plot
    avg_values = {'perfect':{},
                  'key translator':{},
                  'unlearned':unlearned}
    # Average values of the trained solvers
    for m,model in enumerate(models):
        avg_values['key translator'][f'{m}'] = {'model':model.model.name,
                                                'loss fcn':model.loss_fcn.name}
        avg_values['perfect'][f'{m}'] = np.mean(values['perfect'][:,m])
        avg_values[f'{m}'] = {}
        for key,item in values[f'{m}'].items():
            avg_values[f'{m}'][key] = np.mean(values[f'{m}'][key])
    # Average values of the untrained solvers
    for ulm in unlearned:
        avg_values[ulm] = {}
        for key,item in values[ulm].items():
            avg_values[ulm][key] = np.mean(values[ulm][key])
    return avg_values
