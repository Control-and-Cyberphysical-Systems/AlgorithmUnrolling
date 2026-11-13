from functions.helperfunctions import *
from loss_functions import *
from functions import models
import numpy as np
import torch
import cvxopt
import random
from functions import train,test,evaluate_dpc
class GD_net():
    def __init__(self,run_dict:dict):
        # Set the values for the problem
        self.N = run_dict["N"]
        self.T_s = 1
        self.pre_training_or_not = run_dict["pre training"]
        self.learning_rate = run_dict["learning rate"]
        self.rho = 0.1
        solvers.options['show_progress'] = False
        # If we do DPC as a cost function read the appropriate parameters
        if "dpc" in run_dict["loss function"].lower():
            own_dic = os.path.dirname(os.path.realpath(__file__))
            dpc_dict = loadmat(os.path.join(own_dic,f"mat_files_{self.N}","QP_parameters_SS.mat"))
            self.dpc_dict = dpc_dict
            npm = int(np.round(self.dpc_dict["G"].shape[0]/(2*self.N)))
            self.n = self.dpc_dict["E"].shape[1]
            self.m = npm-self.n
            # self.n = self.dpc_dict["B_LS"].shape[0]
            # self.m = self.dpc_dict["B_LS"].shape[1]
            ols_read = dpc_dict["Kx"]
            ols_enhanced = np.vstack((np.eye(self.n),ols_read))
            tls_read = dpc_dict["Ku"]
            tls_enhanced = np.vstack((np.zeros((self.n,self.N*self.m)),tls_read))
            prob_dict = set_problem(T_s=self.T_s,N_pred=self.N)
            prob_dict["AA"] = ols_enhanced
            prob_dict["BB"] = tls_enhanced
            prob_dict["F"] = np.asarray(self.dpc_dict["F"],dtype=np.float64)
            prob_dict["G"] = np.asarray(self.dpc_dict["G"],dtype=np.float64)
            prob_dict["H"] = np.asarray(self.dpc_dict["H"],dtype=np.float64)
            prob_dict["E"] = np.asarray(self.dpc_dict["E"],dtype=np.float64)
            prob_dict["d"] = np.asarray(self.dpc_dict["d"],dtype=np.float64)
        else:
            prob_dict = set_problem(T_s=self.T_s,N_pred=self.N)
        # Set all parameters
        self.A = prob_dict["A"]
        self.B = prob_dict["B"]
        self.n = self.B.shape[0]
        self.m = self.B.shape[1]
        self.p = self.N*(self.n+self.m)
        self.Q = prob_dict["Q"]
        self.R = prob_dict["R"]
        self.lx =prob_dict["lx"]
        self.ux =prob_dict["ux"]
        self.lu =prob_dict["lu"]
        self.uu =prob_dict["uu"]
        self.H = prob_dict["H"]
        self.F = prob_dict["F"]
        self.c = prob_dict["c"]
        self.G = prob_dict["G"]
        self.E = prob_dict["E"]
        self.d = prob_dict["d"]
        self.GG =prob_dict["GG"]
        self.HH =prob_dict["HH"]
        self.QQ =prob_dict["QQ"]
        self.FF =prob_dict["FF"]
        self.Lz =prob_dict["Lz"]
        self.Uz =prob_dict["Uz"]
        self.AA =prob_dict["AA"]
        self.BB =prob_dict["BB"]
        self.GHG= self.G@np.linalg.inv(self.H)@self.G.T
        self.EGHF=self.E+self.G@np.linalg.inv(self.H)@self.F
        self.a_opt = 1/np.max(np.linalg.eigh(self.GHG).eigenvalues)
        prob_dict["N"] = self.N
        prob_dict["m"] = self.m
        prob_dict["n"] = self.n
        self.opt_steps = run_dict["number of model parameters"]
        z_to_u = np.zeros((self.m*self.N,self.N*(self.n+self.m)))
        for i in range(0,self.N):
            for j in range(0,self.m):
                z_to_u[i*self.m+j,i*(self.m+self.n)+j] = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtypus = torch.float64
        self.z_to_u = z_to_u
        self.data_dict = prob_dict
        prob_dict["dtype"] = self.dtypus
        self.get_loss_function(run_dict=run_dict)
        self.create_data_set(run_dict=run_dict)
        # Set the trainable model
        self.model = models.NET_CLASS(name=run_dict["model"],
                                      input_dim=self.n,
                                      output_dim=self.p,
                                      problem_parameters=prob_dict,
                                      num_pars=run_dict["number of model parameters"])
        if run_dict["batch size"] == 0:
            self.batch_size = self.x_train.shape[0]
        else:
            self.batch_size = run_dict["batch size"]
        self.repititions = run_dict["repetitions"]
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.train_loss = []
        self.improvement = []
        # Get some external functions
        self.train = train.train
        self.test = test.test
        self.evaluate_dpc = evaluate_dpc.evaluate_dpc
        return
    
    def save_name(self,model:models.NET_CLASS)->str:
        '''
        Defines the name under which the trained model parameters are saved
        '''
        return f"{model.name}_{self.N}_{model.num_pars}"
    
    def save_model(self,model:models.NET_CLASS)->bool:
        '''
        Saves the model under a predefined naming condition.
        '''
        savename = self.save_name(model)
        
        own_dic = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        torch.save(model.state_dict(),os.path.join(own_dic,"model_parameters",f"{savename}.pt"))
        return True

    def load_model(self,model:models.NET_CLASS)->models.NET_CLASS:
        """
        If there is a saved model, this method will load the model parameter
        """
        savename = self.save_name(model)
        own_dic = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if os.path.isfile(os.path.join(own_dic,"model_parameters",f"{savename}.pt")):
            model.load_state_dict(torch.load(os.path.join(own_dic,"model_parameters",f"{savename}.pt")))
            print(f"Loaded Model: {savename}")
        else:
            print(f"Nothing to load under {savename}.pt ...")
        return model

    def compare(self, xy: np.ndarray):
        """
        Parameters
        ----------
        xy : np.ndarray
            Anfangszustände mit Form (B, n).

        Returns
        -------
        z_full : np.ndarray
            Uncondensed MPC-Variablen mit Form
            (B, N * (m + n), opt_steps + 1).

        X : np.ndarray
            Zustandstrajektorien mit Form
            (B, N + 1, n, opt_steps + 1).

        U : np.ndarray
            Eingangstrajektorien mit Form
            (B, N, m, opt_steps + 1).

        lam : np.ndarray
            Dualvariablen mit Form
            (B, n_constraints, opt_steps + 1).
        """

        def condensed_to_uncondensed(self,x0: np.ndarray,U: np.ndarray):
            """
            Parameters
            ----------
            x0 : np.ndarray
                Form (B, n)

            U : np.ndarray
                Condensed Eingänge mit Form (B, N*m, T)

            Returns
            -------
            z_full : np.ndarray
                Form (B, N*(m+n), T)

            X_traj : np.ndarray
                Form (B, N+1, n, T)

            U_traj : np.ndarray
                Form (B, N, m, T)
            """

            AA = np.asarray(self.AA, dtype=x0.dtype)
            BB = np.asarray(self.BB, dtype=x0.dtype)

            B_batch = x0.shape[0]
            T = U.shape[2]
            # x0: (B, n)
            # AA.T: (n, (N+1)*n)
            # -> (B, (N+1)*n)
            X0_part = x0 @ AA.T

            # BB: ( (N+1)*n, N*m )
            # U:  ( B, N*m, T )
            # ->  ( B, (N+1)*n, T )
            X = X0_part[..., np.newaxis] + np.einsum(
                "ij,bjt->bit",
                BB,
                U
            )
            # (B, N, m, T)
            U_traj = U.reshape(B_batch, self.N, self.m, T)
            # (B, N+1, n, T)
            X_traj = X.reshape(B_batch, self.N + 1, self.n, T)
            # Zustände x_1, ..., x_N mit den Eingängen kombinieren
            # -> (B, N, m+n, T)
            z_full = np.concatenate(
                [U_traj, X_traj[:, 1:, :, :]],
                axis=2
            )

            # -> (B, N*(m+n), T)
            z_full = z_full.reshape(
                B_batch,
                self.N * (self.m + self.n),
                T
            )

            return z_full, X_traj, U_traj

        # Sicherstellen, dass alles NumPy-Arrays mit passendem dtype sind
        xy = np.asarray(xy)

        if not np.issubdtype(xy.dtype, np.floating):
            xy = xy.astype(np.float64)

        F = np.asarray(self.F, dtype=xy.dtype)
        G = np.asarray(self.G, dtype=xy.dtype)
        H = np.asarray(self.H, dtype=xy.dtype)
        GHG = np.asarray(self.GHG, dtype=xy.dtype)
        EGHF = np.asarray(self.EGHF, dtype=xy.dtype)

        B = xy.shape[0]
        n_constraints = G.shape[0]
        T = self.opt_steps + 1

        # EGHF: (n_constraints, n)
        # xy:   (B, n)
        # ->    (B, n_constraints)
        EFGHx = xy @ EGHF.T

        # lam: (B, n_constraints, T)
        lam = np.zeros(
            (B, n_constraints, T),
            dtype=xy.dtype
        )

        # Projected Gradient Descent für die Dualvariablen
        for i in range(self.opt_steps):
            # lam[:, :, i]: (B, n_constraints)
            # GHG:          (n_constraints, n_constraints)
            # ->            (B, n_constraints)
            gradient = np.einsum(
                "ij,bj->bi",
                GHG,
                lam[:, :, i]
            ) + EFGHx+self.d.T

            # Projektion auf lambda >= 0
            lam[:, :, i + 1] = np.maximum(
                lam[:, :, i] - self.a_opt * gradient,
                0.0
            )
        # F:  (N*m, n)
        # xy: (B, n)
        # ->  (B, N*m)
        Fx = xy @ F.T
        # G.T: (N*m, n_constraints)
        # lam: (B, n_constraints, T)
        # ->   (B, N*m, T)
        Glam = np.einsum(
            "ip,bpt->bit",
            G.T,
            lam
        )
        # rhs: (B, N*m, T)
        rhs = Fx[..., np.newaxis] + Glam
        # Gesucht:
        #
        #     H z = -rhs
        #
        # Statt np.linalg.inv(H) @ rhs verwenden wir solve.
        # Dazu werden Batch- und Zeitdimension zusammengefasst.
        n_z = H.shape[0]
        rhs_flat = rhs.transpose(1, 0, 2).reshape(
            n_z,
            B * T
        )
        z_flat = -np.linalg.solve(H, rhs_flat)
        # Zurück zu (B, N*m, T)
        z = z_flat.reshape(n_z, B, T).transpose(1, 0, 2)
        z_full, X, U = condensed_to_uncondensed(self, xy, z)
        return z_full, z_full, lam

    def get_loss_function(self,run_dict:dict)->None:
        '''
        Decides on which loss function to use while training, testing and evaluating
        '''
        if run_dict["loss function"] == "dpc mega dual":
            self.loss_fcn = DPCmegaDUAL(n=self.n,m=self.m,N=self.N)
        elif run_dict["loss function"] == "dpc mega dual last":
            self.loss_fcn = DPCmegaDUALLast(n=self.n,m=self.m,N=self.N)
        elif run_dict["loss function"] == "dpc mega":
            self.loss_fcn = DPCmega(n=self.n,m=self.m,N=self.N)
        elif run_dict["loss function"] == "dpc mega last":
            self.loss_fcn = DPCmegaLast(n=self.n,m=self.m,N=self.N)
        elif run_dict["loss function"] == "dpc mega first last":
            self.loss_fcn = DPCmegaLast(n=self.n,m=self.m,N=self.N)
        elif run_dict["loss function"] == "dpc mega hat":
            self.loss_fcn = DPCmegaHat(n=self.n,m=self.m,N=self.N)
                
        return

    def create_data_set(self,run_dict:dict)->None:
        '''
        Create the data set that is used for training, testing and evaluation
        '''
        self.n_p= run_dict["n_p"]
        data_set_name = f"gd_set_{self.n_p}_{self.N}.json"
        cd = os.path.abspath(__file__)
        cd2 = os.path.dirname(cd)
        pd = os.path.dirname(cd2)
        filepath = os.path.join(pd, "data", data_set_name)
        # Produce data
        if not os.path.isfile(filepath):
            print(f"no file under:{data_set_name}")
            P = matrix(self.G@np.linalg.inv(self.H)@self.G.T)
            G = matrix(-np.eye(self.G.shape[0]))
            h = matrix(np.zeros(self.G.shape[0]))
            X1 = np.linspace(-25,25,self.n_p)
            X2 = np.linspace(-5,5,self.n_p)
            # X: Points in parameter space x0
            X = []
            # Y: Solution to the provided cost function
            Y = []
            # Z: Just the optimal input (for closed loop)
            Z = []
            print(f"Creating Data Set: {self.n_p}x{self.n_p} N: {self.N}")
            for i in range(0,self.n_p):
                for j in range(0,self.n_p):
                    x = np.array([X1[i],X2[j]])
                    q = matrix((self.G@np.linalg.inv(self.H)@self.F+self.E)@x+self.d.reshape(self.d.shape[0]))
                    sol = solvers.qp(P=P,q=q,G=G,h=h)
                    if sol["primal infeasibility"]**2 +sol["dual infeasibility"]**2 < 10**(-14):
                        z_opt = np.array(sol["x"])
                        X.append(x)
                        Y.append(z_opt)
                        Z.append(z_opt[:,0])
                        update_progress_bar(i*self.n_p+j+1,self.n_p**2,"")
            data_set = {"Name":data_set_name,
                        "X":X,
                        "Y":Y,
                        "Z":Z}
            save_data(save_dict=data_set)
        else:
            print(f"Loading data set: {data_set_name}")
            data_set = read_data(name=filepath)

        # Create test and train split

        num_points = len(data_set["X"])
        test_train_split = 0.8
        train_points = int(np.round(num_points*test_train_split))
        order = list(range(0,num_points))
        random.shuffle(order)
        train_indxs = order[0:train_points]
        test_indxs = order[train_points:]
        x_full = []
        y_full = []
        z_full = []
        for i in order:
            x_full.append(data_set["X"][i])
            y_full.append(data_set["Y"][i])
            z_full.append(data_set["Z"][i])
        # save the data appropriately
        self.x_train = torch.tensor(np.array(x_full[0:train_points]),dtype=self.dtypus)
        self.y_train = torch.tensor(np.array(y_full[0:train_points]),dtype=self.dtypus)
        self.z_train = torch.tensor(np.array(z_full[0:train_points]),dtype=self.dtypus)
        self.x_test  = torch.tensor(np.array(x_full[train_points:]) ,dtype=self.dtypus)
        self.y_test  = torch.tensor(np.array(y_full[train_points:]) ,dtype=self.dtypus)
        self.z_test  = torch.tensor(np.array(z_full[train_points:]) ,dtype=self.dtypus)
        return
