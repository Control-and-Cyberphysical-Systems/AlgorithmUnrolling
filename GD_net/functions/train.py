import torch
import random
import numpy as np
from functions.helperfunctions import *
def train(self,printer:bool=False):
    self.model = self.load_model(self.model)
    # Helps debugging by finding the exact line, where the error occurs
    torch.autograd.set_detect_anomaly(True)
    self.model.train()
    X = self.x_train
    Y = self.y_train
    update_progress_bar(1,1000000,f'Start of training | {self.save_name(self.model)} | reps:{self.repititions:.2e} | lr:{self.learning_rate:.2e}')
    
    batch_size = self.batch_size
    loss_glob = []
    print_loss =[]
    to_print= int(np.ceil(self.repititions*(int(np.ceil(X.shape[0]/batch_size)))/1000))#200
    print_iter = 0
    tol_eps = 1e-10
    old_print_loss = 1
    improvement = []
    optimizer = torch.optim.AdamW(self.model.parameters(),lr =self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=1/(10**(1.0/4.0)),
                                                        threshold=0.001,
                                                        # threshold_mode='abs',
                                                        patience=10,
                                                        eps= 1e-16)
    optimizer.zero_grad()
    j = 0
    while j< self.repititions and optimizer.param_groups[0]['lr'] > 1.1*tol_eps:
        order = list(range(0,X.shape[0]))
        random.shuffle(order)
        max_train_iters = batch_size*(len(order)//batch_size)
        for i_ind in range(0,max_train_iters, batch_size):
            batch_idx = order[i_ind:i_ind+batch_size]
            X_batch = X[batch_idx].to(self.device)
            Y_batch = Y[batch_idx].to(self.device)
            if batch_size == 1:
                X_batch = X_batch.reshape(1, X.shape[1])
                Y_batch = Y_batch.reshape(1, Y.shape[1])
            optimizer.zero_grad() #  -> Clear the graidients
            prediction = self.model(X_batch)
            loss_loc = self.loss_fcn(z=prediction[0],
                                    z_hat=prediction[1],
                                    mu=prediction[2],
                                    z_opt=Y_batch,
                                    x=X_batch)
            loss_glob.append(float(loss_loc.detach()))
            print_loss.append(loss_loc.detach().cpu())
            loss_loc.backward()
            optimizer.step()
            if self.batch_size == self.x_train.shape[0]:
                scheduler.step(loss_loc.detach())
            # Update current lr
            current_lr = optimizer.param_groups[0]['lr']
            print_iter +=1
            if print_iter%to_print == 0 and printer:
                update_progress_bar(j*max_train_iters+i_ind+1,self.repititions*max_train_iters,f'Training | loss:{np.mean(print_loss): .8e} | lr:{current_lr:.2e} | rel. improvement per step: {(old_print_loss-np.mean(print_loss))/(old_print_loss+1e8): 2.4e}')
                improvement.append((old_print_loss-np.mean(print_loss))/(old_print_loss+1e8))
                old_print_loss = np.mean(print_loss).copy()
                print_loss = []
                print_iter = 0
        j+=1
    update_progress_bar(99,100,f'End of training | loss:{old_print_loss: .4e} | lr:{current_lr:.2e}\n')

    self.save_model(self.model)
    self.train_loss = loss_glob
    self.improvement = np.array(improvement)
    return