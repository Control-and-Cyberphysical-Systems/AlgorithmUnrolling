import torch
import numpy as np
from functions.helperfunctions import *
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

def test(self,printer:bool=False):
    torch.autograd.set_detect_anomaly(True)
    self.model = self.load_model(self.model)
    self.model.eval()
    trainable = True
    for p in self.model.parameters():
        p.requires_grad = False
    if 'dpc' == self.loss_fcn.name:
        self.get_loss_function(run_dict={'loss function':'unsupervised dual last'})
        print('hello')
    optimizer = torch.optim.AdamW(self.model.parameters(),lr =0.001)
    optimizer.zero_grad()
    X = self.x_test
    Y = self.y_test
    update_progress_bar(1,1000000,f'Start of testing | {self.save_name(self.model)}')
    to_print= int(np.ceil(X.shape[0]/1000))
    order = list(range(0,X.shape[0]))
    batch_size = 1
    max_train_iters = batch_size*(len(order)//batch_size)
    test_loss = []
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
                                    z_opt=Y_batch.unsqueeze(-1),
                                    x=X_batch)

        # Calculate the backward graph and then make the optimizer step
        test_loss.append(float(loss_loc.detach().mean()))
        if (i_ind)%to_print == 0 and printer: 
            update_progress_bar(i_ind,max_train_iters,'Testing')
    fig,axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].plot(np.abs(self.train_loss))
    axs[0].set_yscale('log')
    pos = self.improvement[1:] > 0
    neg = self.improvement[1:] < 0
    if len(self.improvement)>1 and np.any(self.improvement)>0:
        axs[1].plot(np.where(pos, self.improvement[1:], np.nan), 'r-', label='positive')
        axs[1].plot(np.abs(np.where(neg, self.improvement[1:], np.nan)), 'g-', label='negative')
        axs[1].set_yscale('log')  # linthresh: unterhalb linear
        axs[1].axhline(0, color='k', linewidth=0.5)
        axs[1].legend()
    axs[2].plot(np.abs(test_loss))
    axs[2].set_yscale('log')
    own_dic = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    fig.savefig(os.path.join(own_dic,'testing_figures',f'{self.save_name(self.model)}_{self.n_p}.png'))
    plt.close(fig)
    update_progress_bar(99,100,f'End of testing | {self.save_name(self.model)} |\n')
    return