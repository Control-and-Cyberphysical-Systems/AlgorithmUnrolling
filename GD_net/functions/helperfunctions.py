import numpy as np
import control as ctrl
from cvxopt import matrix,solvers
import os
import sys
import json
def find_H(A:np.ndarray,B:np.ndarray,
           Mx:np.ndarray,wx:np.ndarray,
           Mu:np.ndarray,wu:np.ndarray,
           N:int,Q:np.ndarray,R:np.ndarray,
           lx:np.ndarray,ux:np.ndarray,
           lu:np.ndarray,uu:np.ndarray,
           Mt=None,wt=None)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Hilfsfunktion, die aus einem gegebenen 
    System mit Kosten und Constraints die nötigen Matrizen für ein MQP produziert
    '''
    def compute_condensed_matrices(A:np.ndarray,B:np.ndarray,Q:np.ndarray,
                                   R:np.ndarray,P:np.ndarray,N:int,n:int,m:int):
        '''
        Berechne die kondensierten Matrizen aus einem gegebenen System
        '''
        AA = np.zeros(((N+1)*n,n))
        BB = np.zeros(((N+1)*n,N*m))
        QQ = np.zeros(((N+1)*n,(N+1)*n))
        RR = np.zeros((N*m,N*m))
        for k in range(0,N+1):
            AA[k*n:(k+1)*n,:] = np.linalg.matrix_power(A, k)

            for j in range(0,k):
                BB[k*n:(k+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, k-j-1) @ B
        
            if k < N:
                QQ[k*n:(k+1)*n, k*n:(k+1)*n] = Q
                RR[k*m:(k+1)*m, k*m:(k+1)*m] = R
            else:
                QQ[k*n:(k+1)*n, k*n:(k+1)*n] = P#2*Q #100*Q # Terminal costs. Kann auch einfach die normalen Kosten (nur sehr hoch) sein
        return AA,BB,QQ,RR

    def compute_condensed_constraints(Mx:np.ndarray,wx:np.ndarray,
                                      Mu:np.ndarray,wu:np.ndarray,N:int,*args):
        '''
        Berechne die Constraints für das kondensierte System
        '''
        # switch nargin equivalent
        if len(args) == 0:
            Mt = Mx
            wt = wx
        elif len(args) == 2:
            Mt, wt = args
        else:
            raise ValueError("Unsupported number of inputs")

        lx, n = Mx.shape
        lt = Mt.shape[0]
        lu, m = Mu.shape

        MMx = np.zeros((N*lx + lt, (N+1)*n))
        wwx = np.zeros((N*lx + lt, 1))
        MMu = np.zeros((N*lu, N*m))
        wwu = np.zeros((N*lu, 1))

        for k in range(N):
            MMx[k*lx:(k+1)*lx, k*n:(k+1)*n] = Mx
            wwx[k*lx:(k+1)*lx] = wx
            MMu[k*lu:(k+1)*lu, k*m:(k+1)*m] = Mu
            wwu[k*lu:(k+1)*lu] = wu

        MMx[N*lx:N*lx+lt, N*n:(N+1)*n] = Mt
        wwx[N*lx:N*lx+lt] = wt
        return MMx,wwx,MMu,wwu
    
    def compute_terminal_set(A:np.ndarray,B:np.ndarray,K:np.ndarray,Mx:np.ndarray,wx:np.ndarray,Mu:np.ndarray,wu:np.ndarray):
        """
        Subfunction to determine the terminal constraints of the control law
        """
        ABK = A - B@K
        Ml = np.vstack((Mx,-Mu@K))
        wl = np.vstack((wx,wu))
        nl = np.shape(wl)[0]#
        k = 0
        Mt = Ml
        wt = wl
        found_new_constraint = True
        while found_new_constraint:
            k+=1
            found_new_constraint = False
            ABKk = np.linalg.matrix_power(ABK,k)
            for i in range(0,nl):
                fT = Ml[i,:]@ABKk
                c = matrix(-fT)
                G = matrix(Mt)
                h = matrix(wt)
                solvers.options["show_progress"] = False
                res = solvers.lp(c,G,h)
                xAst  = np.array(res['x']).flatten()
                wMax  = fT @ xAst
                if wMax>wl[i]:
                    found_new_constraint = True
                    Mt = np.vstack((Mt,fT))
                    wt = np.vstack((wt,wl[i]))
        return Mt,wt,Ml,wl,k
    
    def compute_uncondensed_matrices(A:np.ndarray,B:np.ndarray,
                                     Q:np.ndarray,R:np.ndarray,
                                     P:np.ndarray,N:int,
                                     lx:np.ndarray,ux:np.ndarray,
                                     lu:np.ndarray,uu:np.ndarray):
        n,m = B.shape
        p = N*(n+m)
        big_H = np.zeros((p,p))
        big_Q = np.zeros((n,n))
        big_G = np.zeros((N*n,p))
        big_F = np.zeros((N*n,n))
        Lz = np.zeros((p,1))
        Uz = np.zeros((p,1))
        big_F[0:n,:] = A
        for k in range(0,N):
            big_H[k*(n+m):k*(n+m)+m,k*(n+m):k*(n+m)+m] = R
            if k<N-1:
                big_H[k*(n+m)+m:(k+1)*(n+m),k*(n+m)+m:(k+1)*(n+m)] = Q
            else:
                big_H[k*(n+m)+m:(k+1)*(n+m),k*(n+m)+m:(k+1)*(n+m)] = P
            for j in range(0,k+1):
                big_G[k*n:(k+1)*n,j*(n+m):j*(n+m)+m] = -np.linalg.matrix_power(A,k-j)@B
                if j == k:
                    big_G[k*n:(k+1)*n,j*(n+m)+m:(j+1)*(n+m)] = np.eye(n)
                if j == k-1:
                    big_G[k*n:(k+1)*n,j*(n+m)+m:(j+1)*(n+m)] = -A
            Lz[k*(n+m):k*(n+m)+m] = lu
            Lz[k*(n+m)+m:(k+1)*(n+m)] = lx
            Uz[k*(n+m):k*(n+m)+m] = uu
            Uz[k*(n+m)+m:(k+1)*(n+m)] = ux
        big_Q = Q

        return big_H,big_Q,big_G,big_F,Lz,Uz

    n,m = B.shape
    P,_,K = ctrl.dare(A,B,Q,R)
    #K Nur wichtig für das Terminal set ~> Erstmal ohne
    # Apply the control law to find the terminal constraints as well as the terminal costs
    # if Mt is None or wt is None:
    #     Mt,wt,_,_,_ = compute_terminal_set(A,B,K,Mx,wx,Mu,wu)

    AA,BB,QQ,RR = compute_condensed_matrices(A,B,Q,R,P,N,n,m)
    
    H = 2*(RR+BB.T@QQ@BB)
    F = 2*BB.T@QQ@AA
    c = AA.T@QQ@AA

    if Mt is None and wt is None:
        [MMxT,wwxT,MMu,wwu] = compute_condensed_constraints(Mx,wx,Mu,wu,N)#,Mt,wt)
    else:
        [MMxT,wwxT,MMu,wwu] = compute_condensed_constraints(Mx,wx,Mu,wu,N,Mt,wt)
    
    G =np.vstack((MMxT@BB,MMu)) #zusammengefasste NB x und u
    E =np.vstack((-MMxT@AA,np.zeros((wwu.shape[0],n))))
    d =np.vstack((wwxT,wwu))
    big_H,big_Q,big_G,big_F,Lz,Uz = compute_uncondensed_matrices(A,B,Q,R,P,N,lx,ux,lu,uu)
    return H,F,c,G,E,d,AA,BB,big_H,big_Q,big_G,big_F,Lz,Uz,QQ,RR

def set_problem(T_s:float,N_pred:int,A:np.ndarray=None,B:np.ndarray=None)->dict[str,np.ndarray]:
    '''
    Set the double integrator problem dependent on the parameters 
    '''
    n = 2
    m = 1 
    Ts = T_s
    N_pred = N_pred
    if A is None:
        A = np.eye(n)
        A[0,1] = Ts

        B = np.ones((n,m))
        B[0] = (Ts**2 )/ 2
        B[1] = Ts
    
    # Define input and state constraints
    lx = np.ones((n,1))
    lx[0,0] = -25
    lx[1,0] = -5
    ux = np.ones((n,1))
    ux[0,0] = 25
    ux[1,0] = 5
    lu = np.ones((m,1))
    lu[0,0] = -1
    uu = np.ones((m,1))
    uu[0,0] = 1

    lbx = np.eye(n)
    ubx = np.eye(n)
    lbu = np.eye(m)
    ubu = np.eye(m)

    Mx = np.vstack((ubx,-lbx))
    wx = np.vstack((ux,-lx))
    Mu = np.vstack((ubu,-lbu))
    wu = np.vstack((uu,-lu))

    # Define the cost matrices of the normal system
    Q = np.eye(n)

    R = 0.1*np.eye(m)

    Mt = None
    wt = None
    # Get the matrices from the system
    H,F,c,G,E,d,AA,BB,HH,QQ,GG,FF,Lz,Uz,Qb,Rb = find_H(A=A,
                                                   B=B,
                                                 Mx=Mx,
                                                 wx=wx,
                                                 Mu=Mu,
                                                 wu=wu,
                                                 N=N_pred,
                                                   Q=Q,
                                                   R=R,
                                                 Mt=Mt,
                                                 wt=wt,
                                                 lx=lx,
                                                 ux=ux,
                                                 lu=lu,
                                                 uu=uu)
    
    prob_dict = {"A":A,
                 "B":B,
                 "lx":lx,
                 "ux":ux,
                 "lu":lu,
                 "uu":uu,
                 "Q":Q,
                 "R":R,
                 "H":H,
                 "F":F,
                 "c":c,
                 "G":G,
                 "E":E,
                 "d":d,
                 "AA":AA,
                 "BB":BB,
                 "GG":GG,
                 "HH":HH,
                 "QQ":QQ,
                 "FF":FF,
                 "Lz":Lz,
                 "Uz":Uz,
                 "Qx":Qb,
                 "Ru":Rb,
                 }
    return prob_dict

def json_array(arr:np.ndarray)->dict:
    """
    Json cannot work with np.arrays, it needs to be transformed 
    into a list, but needs to be recreatable\n
    INPUT: np.array \n
    OUTPUT: dict[data,dtype,shape]"""
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    data = arr.tolist()
    shape = arr.shape
    dtype = str(arr.dtype)
    return {"data":data,"shape":shape,"dtype":dtype}

def array_json(arr_dict:dict)->np.ndarray:
    """
    Inverse transformation from json readable format to numpy array
    INPUT: encoded array
    OUTPUT: decoded array
    """
    arr_loaded = np.array(arr_dict["data"], dtype=arr_dict["dtype"])
    arr_loaded = arr_loaded.reshape(arr_dict["shape"])
    return arr_loaded 

def save_data(save_dict: dict | None = None):
    """
    Save the prepared Data to JSON. Converts numpy arrays in arbitrarily nested
    dicts/lists/tuples using json_array().
    """
    if save_dict is None:
        save_dict = {}

    def convert(item):
        if isinstance(item, np.ndarray):
            return json_array(item)
        elif isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        else:
            return item

    to_save_dict = convert(save_dict)

    # Dateiname bestimmen
    name = save_dict.get("Name", "data")
    if not name.endswith(".json"):
        name = f"{name}.json"

    # Zielpfad
    cd = os.path.abspath(__file__)
    cd2 = os.path.dirname(cd)
    cd3 = os.path.dirname(cd2)
    pd = os.path.dirname(cd3)
    filepath = os.path.join(pd, "data", name)

    with open(filepath, "w") as f:
        json.dump(to_save_dict, f)

    return True, name

def read_data(name: str):
    """
    Read saved JSON data and convert stored numpy arrays back.
    """

    if not name.endswith(".json"):
        name = f"{name}.json"

    cd = os.path.abspath(__file__)
    pd = os.path.dirname(os.path.dirname(os.path.dirname(cd)))
    filepath = os.path.join(pd, "data", name)

    with open(filepath) as f:
        savefile = json.load(f)

    def convert(item):
        # numpy-array representation
        if isinstance(item, dict) and "dtype" in item:
            return array_json(item)

        # nested dict
        elif isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}

        # lists / tuples
        elif isinstance(item, list):
            return [convert(v) for v in item]

        else:
            return item

    return convert(savefile)

def diff(x,dt):
    '''
    Calculate the numerical tangent dependent on the shape of x:
        - two entries: forward tangent
        - three entries: symmetrix tangent
        - otherwise: 0
    '''
    if x.shape[0] == 2:
        df = (x[1]-x[0])/dt
    elif x.shape[0] == 3:
        df = (x[2]-x[0])/(2*dt)
    else:
        df = 0
    return df

def update_progress_bar(i:int=0,maxiter:int=100,process:str="..."):
    '''
    Produces and updates a progress bar
    
    :param i: current index of the progress bar
    :type i: int
    :param maxiter: maximal iteration (to calculate the 100%)
    :type maxiter: int
    :param process: Name of the current process
    :type process: str
    '''
    progress = (i+1)/maxiter
    bar = '█' * int(progress*40)
    empty = (2*len(process))*' '
    # sys.stdout.write(f'\rSimulation: |{bar:<40}| {progress*100:05.1f}% {empty}')
    sys.stdout.write(f'\rSimulation: |{bar:<40}| {progress*100:05.1f}% {process}')
    sys.stdout.flush()
    return

class DPG():
    '''
    Basic untrained dual projected gradient descent
    '''
    def __init__(self,M,N,C,H,F,G,n_steps) -> None:
        self.M = M
        self.N = N
        self.C = C
        self.H = H
        self.F = F
        self.G = G
        self.alpha = 1/np.max(np.linalg.eigh(M).eigenvalues)
        self.n_steps = n_steps
        return

    def __call__(self,x:np.ndarray)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
        lam = np.zeros((1,self.M.shape[0],self.n_steps+1))
        z = np.zeros((1,self.G.shape[1],self.n_steps+1))
        for i in range(0,self.n_steps):
            lam_kp1 = lam[0,:,i] - self.alpha*(self.M@lam[0,:,i]+self.N@x[0,:]+self.C[:,0])
            lam_kp1[lam_kp1<0] = 0
            lam[0,:,i+1] = lam_kp1
            if i == self.n_steps-1:
                z[0,:,i+1] = -np.linalg.solve(self.H, (x @ self.F.T + lam[0,:,i+1] @ self.G).T).T
        return z,z,lam

class PG():
    '''
    Basic untrained projected gradient descent
    '''
    def __init__(self,H,F,G,E,d,n_steps) -> None:
        self.H = H
        self.F = F
        self.G = G
        self.E = E
        self.G_implicit = -np.linalg.inv(H)@F
        self.d = np.array(d,dtype=float)
        self.Uz = self.d[0:H.shape[0],0]
        self.Lz = self.d[H.shape[0]:,0]
        self.alpha = 1/np.max(np.linalg.eigh(H).eigenvalues)
        self.n_steps = n_steps
        self.rho = 0.1
        return

    def __call__(self,x:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        z = np.zeros((1,self.H.shape[0],self.n_steps+1))
        lam = np.zeros((1,self.G.shape[0],self.n_steps+1))
        z[0,:,0] = np.clip(x@self.G_implicit.T, -self.Lz, self.Uz)
        for i in range(0,self.n_steps):

            # Take primal step
            z_kp1_poss = z[0,:,i] - self.alpha*(self.H@z[0,:,i]+self.F@x[0,:])

            # Project into feasible region
            z_kp1 = np.clip(z_kp1_poss, -self.Lz, self.Uz)

            # Calculate the dual variables after each step (for comparisons)
            broken_cons = z_kp1 @ self.G.T - x[0,:] @ self.E.T - self.d.reshape(-1)
            active_cons =  broken_cons > -1e-8
            lam_k = np.zeros((self.G.shape[0],))#lam[0, :, i].copy()
            n_active = np.sum(active_cons)
            if n_active >=1 and i == self.n_steps-1:#==self.G.shape[1]:
                P = matrix(self.G[active_cons,:]@self.G[active_cons,:].T)
                q = matrix(self.G[active_cons,:]@(self.H@z_kp1 + self.F@x[0,:]))
                solu = solvers.qp(P,q,G=matrix(-np.eye(n_active)),h=matrix(np.zeros((n_active, 1))))
                lam_act= np.array(solu["x"]).reshape((n_active,))
                lam_k[active_cons] = np.maximum(lam_act,0.0)
            # else:
            #     print(f"stprü{i}")
            lam[0, :, i+1] = lam_k

            # update variable
            z[0,:,i+1] = z_kp1

        return z,z,lam

class BADMM():
    '''
    Basic untrained Primal ADMM
    '''
    def __init__(self,H,F,G,E,d,n_steps) -> None:
        self.H = H
        self.F = F
        self.G = G
        self.E = E
        self.G_implicit = -np.linalg.inv(H)@F
        self.d = np.array(d,dtype=float)
        self.Uz = self.d[0:H.shape[0],0]
        self.Lz = self.d[H.shape[0]:,0]
        eigs = np.linalg.eigh(H).eigenvalues
        tol = 1e-3
        self.alpha = np.sqrt(np.max(eigs[eigs>tol])*np.min(eigs[eigs>tol]))
        self.n_steps = n_steps
        M = np.linalg.pinv(H+self.alpha*np.eye(H.shape[0]))
        self.M1 = M
        self.M2 = -M@F
        return

    def __call__(self,x:np.ndarray)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
        z = np.zeros((1,self.H.shape[0],self.n_steps+1))
        y = np.zeros((1,self.H.shape[0],self.n_steps+1))
        mu = np.zeros((1,self.H.shape[0],self.n_steps+1))
        lam = np.zeros((1,self.G.shape[0],self.n_steps+1))
        y[0,:,0] = x@self.G_implicit.T
        z[0,:,0] = np.clip(x@self.G_implicit.T, -self.Lz, self.Uz)
        for i in range(0,self.n_steps):

            # Take primal step
            y[0,:,i+1] = ((self.alpha*z[0,:,i] - mu[0,:,i])@self.M1 + x@self.M2.T).reshape(self.H.shape[0],)

            # Project into feasible region
            z[0,:,i+1] = np.clip(y[0,:,i+1]+(1/self.alpha)*mu[0,:,i], -self.Lz, self.Uz)
            
            # Update dual variable
            mu[0,:,i+1] = mu[0,:,i] + self.alpha*(y[0,:,i+1]-z[0,:,i+1])
            
            # Calculate the dual variables after each step (for comparisons)
            broken_cons = y[0,:,i+1] @ self.G.T - x[0,:] @ self.E.T - self.d.reshape(-1)
            active_cons =  broken_cons > -1e-8
            lam_k = np.zeros((self.G.shape[0],))#lam[0, :, i].copy()
            n_active = np.sum(active_cons)
            if n_active >=1 and i == self.n_steps-1:#==self.G.shape[1]:
                P = matrix(self.G[active_cons,:]@self.G[active_cons,:].T)
                q = matrix(self.G[active_cons,:]@(self.H@z[0,:,i+1] + self.F@x[0,:]))
                solu = solvers.qp(P,q,G=matrix(-np.eye(n_active)),h=matrix(np.zeros((n_active, 1))))
                lam_act= np.array(solu["x"]).reshape((n_active,))
                lam_k[active_cons] = np.maximum(lam_act,0.0)
            lam[0, :, i+1] = lam_k
        return z,y,lam

class BDADMM():
    '''
    Basic untrained dual ADMM
    '''
    def __init__(self,M,N,C,H,F,G,n_steps) -> None:
        self.M = M
        self.N = N
        self.C = C
        self.H = H
        self.F = F
        self.G = G
        eigs = np.linalg.eigh(M).eigenvalues
        tol = 1e-3
        self.alpha = np.sqrt(np.max(eigs[eigs>tol])*np.min(eigs[eigs>tol]))
        self.n_steps = n_steps
        M1 = np.linalg.inv(M+self.alpha*np.eye(M.shape[0]))
        self.M1 = M1
        self.M2 = -M1@N
        self.M3 = -M1@C
        return

    def __call__(self,x:np.ndarray)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
        z = np.zeros((1,self.M1.shape[0],self.n_steps+1))
        y = np.zeros((1,self.M1.shape[0],self.n_steps+1))
        mu = np.zeros((1,self.M1.shape[0],self.n_steps+1))
        z_true = np.zeros((1,self.G.shape[1],self.n_steps+1))
        for i in range(0,self.n_steps):
            y[0,:,i+1] =  (self.alpha*z[0,:,i] - mu[0,:,i])@self.M1 + (x@self.M2.T + self.M3.reshape(self.M1.shape[0],)).reshape(self.M1.shape[0],)
            zkp1 =  y[0,:,i+1] + (1/self.alpha)*mu[0,:,i]
            zkp1[zkp1<0] = 0
            z[0,:,i+1] =zkp1
            mu[0,:,i+1] = mu[0,:,i] + self.alpha*(y[0,:,i+1]-z[0,:,i+1])
            if i == self.n_steps-1:
                z_true[0,:,i+1] = -np.linalg.solve(self.H, (x @ self.F.T + z[0,:,i+1] @ self.G).T).T
        return z_true,z_true,z
