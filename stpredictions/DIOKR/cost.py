import torch
import numpy as np
from stpredictions.DIOKR.utils import load_candidates

def sloss(Omega, K_x_tr_ba, K_y_tr_ba, K_y_ba_ba, K_y):
    
    n_ba = K_x_tr_ba.shape[1]
        
    mse = (1.0/n_ba) * torch.trace(K_x_tr_ba.T @ Omega @ K_y @ Omega.T @ K_x_tr_ba
                                       - 2 * K_x_tr_ba.T @ Omega @ K_y_tr_ba + K_y_ba_ba)

    return mse

def sloss_batch(Omega_block_diag, K_x_tr_te, K_y_tr_te, K_y_te_te, K_y, n_b):
    
    n_te = K_x_tr_te.shape[1]
        
    se = torch.diag((1.0 / n_b ** 2) * K_x_tr_te.T @ Omega_block_diag @ K_y @ Omega_block_diag.T @ K_x_tr_te
                                       - (2.0 / n_b) * K_x_tr_te.T @ Omega_block_diag @ K_y_tr_te + K_y_te_te)
    
    mse = torch.mean(se)
    std = torch.std(se) / n_te**(1/2)

    return mse, std

def F1_score(Y_tr, K_y, K_x_tr_te, Y_test, Y_candidates, output_kernel, Omega_block_diag, n_b):

        """
           Predict structured object by decoding, then compute structured losses
        """

        n_te = K_x_tr_te.shape[1]
        structured_losses = []
        kernel_losses = []
        Y_pred = []

        for i in range(n_te):

            k_x = K_x_tr_te[:, i]
            y_te = Y_test[i]

            # Compute distances:
            # Compute |Ph(x)|^2 ,  <Ph(x) | psi(y_c)>, and |psi(y_c)|^2 
            A = k_x.T @ Omega_block_diag

            K_h_c = A @ output_kernel.compute_gram(Y_tr, Y_candidates)
            prod_h_c = K_h_c
            K_h_h = A @ K_y @ A.T
            norm_h = K_h_h
            norm_c = torch.diag(output_kernel.compute_gram(Y_candidates))

            se = (1.0 / n_b ** 2) * norm_h.view(-1, 1) - (2.0 / n_b) * prod_h_c + norm_c.view(1, -1)
            scores = - se

            # Predict and compute Structured losses
            idx_pred = torch.argmax(scores)

            # F1
            y_pred = Y_candidates[idx_pred]
            a = torch.sum((y_pred + y_te == 2)).item()
            b = torch.sum((y_pred + y_te >= 1)).item()
            if a + b > 0:
                f1 = 2 * a / (a + b)
            else:
                f1 = 0.
            structured_losses.append(100 * f1)

            y_pred = Y_candidates[idx_pred]

        structured_loss = np.mean(structured_losses)
        structured_std = np.std(structured_losses) / np.sqrt(n_te)
       
        return structured_loss, structured_std
    
def metabolites_scores(Y_tr, K_y, K_x_tr_te, Y_test,
                       output_kernel, Omega_block_diag, n_b,
                       formulas_te, keys_te, n_max=1e5, verbose=1):

        """
           Predict structured object by decoding, then compute structured losses
        """
        
        # Estimation and Decoding
        n_te = K_x_tr_te.shape[1]
        acc_te = np.array([0., 0., 0., 0.])
        rbf_loss_te = []
        ham_loss_te = []
        n_pred = 0
        per = -1

        for i in range(n_te):
            
            # print % of the test set's size predicted
            new_per = int(i / n_te * 100)
            if new_per // 10 > per // 10 and verbose > 0:
                per = new_per
                print(f'{per}%', end=' ', flush=True)
            if i == n_te - 1 and verbose > 0:
                print('100 %', flush=True)
                
            k_x_tr_te = K_x_tr_te[:, i]
            formula = formulas_te[i]
            key = keys_te[i]
            y_te = Y_test[i].reshape(1, -1)
            
            # Load fingerprints/keys candidates
            Y_candidates, keys = load_candidates(formula, n_max)
            if type(Y_candidates) == int:  # Loading error
                continue
            Y_candidates = Y_candidates.astype(np.float64)
            Y_candidates = torch.from_numpy(Y_candidates).float()

            # Compute distances:
            # Compute |Ph(x)|^2 ,  <Ph(x) | psi(y_c)>, and |psi(y_c)|^2 
            A = k_x_tr_te.T @ Omega_block_diag
            
            K_h_c = A @ output_kernel.compute_gram(Y_tr, Y_candidates)
            prod_h_c = K_h_c
            K_h_h = A @ K_y @ A.T
            norm_h = K_h_h
            norm_c = torch.diag(output_kernel.compute_gram(Y_candidates))

            se = (1.0 / n_b ** 2) * norm_h.view(-1, 1) - (2.0 / n_b) * prod_h_c + norm_c.view(1, -1)
            scores = - se
            scores = scores.data.numpy()[0]
            
            # k-NN
            top_k = 20
            sort_score = np.argsort(scores)
            idx_pred = sort_score[-top_k:]
            Keys_predicted = np.array(keys)[idx_pred]
            Y_pred = Y_candidates[idx_pred[-1]].view(1, -1)

            # Compute 3 Structured loss

            # 1) Top-k accuracies
            top_k_loss = np.array([key in Keys_predicted[-20:], key in Keys_predicted[-10:], key in Keys_predicted[-5:],
                                   key in Keys_predicted[-1:]])
            acc_te += top_k_loss

            # 2) Kernel induced loss
            k_pred_te = output_kernel.compute_gram(y_te, Y_pred)
            k_pred_te = k_pred_te.item()
            rbf = 2 * (1 - k_pred_te)
            rbf_loss_te.append(rbf)

            # 3) Hamming
            ham = torch.norm(Y_pred - y_te) ** 2
            ham_loss_te.append(ham.item())

            n_pred += 1

        acc_te /= n_pred
        acc_te *= 100
        rbf_std_te = np.std(rbf_loss_te) / np.sqrt(n_pred)
        rbf_loss_te = np.mean(rbf_loss_te)
        ham_te = np.mean(ham_loss_te)
        #print(f'n predictions : {n_pred}')
       
        return rbf_loss_te, rbf_std_te, acc_te, ham_te

def eps_ridge(Omega, K_x_tr_ba, K_y_tr_ba, K_y_ba_ba, K_y, eps):
    
    n_ba = K_x_tr_ba.shape[1]
    
    norm_h_square = torch.diag(torch.diag(K_x_tr_ba.T @ Omega @ K_y @ Omega.T @ K_x_tr_ba
                                       - 2 * K_x_tr_ba.T @ Omega @ K_y_tr_ba + K_y_ba_ba))
    
    norm_h = norm_h_square ** (1/2)
    
    diag_eps = eps * torch.eye(n_ba)
    
    zero = torch.zeros_like(norm_h_square)
    
    maxi = torch.max(norm_h - diag_eps, zero) ** 2
        
    eps_ridge_loss = (1.0/n_ba) * torch.trace(maxi)

    return eps_ridge_loss

# class PinballIntegral(Cost):
#
#     def __init__(self, lbda_nc):
#         super(PinballIntegral, self).init(signature_primal=ploss_with_crossing(lbda_nc),
#                                           signature_dual=)
