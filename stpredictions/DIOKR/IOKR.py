from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import scipy as sp
import torch
from stpredictions.DIOKR.utils import load_candidates
from time import time


class IOKR(object):
    """
        Main class implementing IOKR + OEL
    """

    def __init__(self, path_to_candidates=None):

        # OEL
        self.oel_method = None
        self.oel = None

        # Output Embedding Estimator
        self.L = None
        self.input_gamma = None
        self.input_kernel = None
        self.linear = False
        self.output_kernel = None
        self.Omega = None
        self.n_anchors_krr = -1

        # Data
        self.n_tr = None
        self.X_tr = None
        self.Y_tr = None
        self.UY_tr = None
        self.K_x = None
        self.K_y = None
        
        # vv-norm
        self.vv_norm = None

        # Decoding
        self.decode_weston = False
        self.path_to_candidates = path_to_candidates

    def fit(self, X_s, Y_s, L=None, input_gamma=None,
            input_kernel=None, linear=False,
            output_kernel=None, Omega=None, oel_method=None,
            K_X_s=None, K_Y_s=None, verbose=0):
        """
            Fit OEE (Output Embedding Estimator = KRR) and OEL with supervised/unsupervised data
        """

        # Saving
        self.X_tr = X_s.clone()
        self.Y_tr = Y_s.clone()
        self.L = L
        self.input_gamma = input_gamma
        self.input_kernel = input_kernel
        self.linear = linear
        self.output_kernel = output_kernel
        self.oel_method = oel_method

        # Training OEE
        t0 = time()

        # Gram computation
        if not self.linear:
            if K_X_s is None:
                self.K_x = self.input_kernel.compute_gram(self.X_tr)
            else:
                self.K_x = K_X_s
        
        if K_Y_s is None:
            self.K_y = self.output_kernel.compute_gram(self.Y_tr)
        else:
            self.K_y = K_Y_s

        # KRR computation (standard or Nystrom approximated)
        if not self.linear:
            self.n_tr = self.K_x.shape[0]
            if Omega is None:
                if self.n_anchors_krr == -1:
                    M = self.K_x + self.n_tr * self.L * torch.eye(self.n_tr)
                    self.Omega = torch.inverse(M)
                else:
                    n_anchors = self.n_anchors_krr
                    idx_anchors = torch.from_numpy(np.random.choice(self.n_tr, n_anchors, replace=False)).int()
                    K_nm = self.K_x[:, idx_anchors]
                    K_mm = self.K_x[np.ix_(idx_anchors, idx_anchors)]
                    M = K_nm.T @ K_nm + self.n_tr * self.L * K_mm
                    self.Omega = K_nm @ torch.inverse(M)
                    self.X_tr = self.X_tr[idx_anchors]
            else:
                self.Omega = Omega
        else:
            m = self.input_kernel.model_forward(self.X_tr).shape[1]
            M = self.input_kernel.model_forward(self.X_tr).T @ self.input_kernel.model_forward(self.X_tr) + m * self.L * torch.eye(m)
            self.Omega = torch.inverse(M) @ self.input_kernel.model_forward(self.X_tr).T
            
        # vv-norm computation
        if self.n_anchors_krr == -1:
            if not self.linear:
                self.vv_norm = torch.trace(self.Omega @ self.K_x @ self.Omega @ self.K_y)
            else:
                self.vv_norm = torch.trace(self.Omega.T @ self.Omega @ self.K_y)

        if verbose > 0:
            print(f'KRR training time: {time() - t0}', flush=True)

        # Training time
        t0 = time()

        
        if verbose > 0:
            print(f'Training time: {time() - t0}', flush=True)

    def predict(self, X):

        """
            KRR prediction function if finite dimensional output space
        """

        K_x_tr_te = self.input_kernel.compute_gram(self.X_tr, Y=X)
        A = K_x_tr_te.T @ self.Omega
        h_pred = A @ self.Y_tr
        return h_pred
    
    def sloss(self, K_x, K_y):
        
        """
            Compute the square loss (train MSE)
        """
        
        n_te = K_x.shape[1]
        if self.n_anchors_krr == -1:
            A = K_x.T @ self.Omega
        else:
            A = self.Omega @ K_x
            A = A.T

        # \|psi(y)\|^2
        norm_y = torch.diag(K_y)

        product_h_y = torch.einsum('ij, ji -> i', A, K_y)
        norm_h = torch.einsum('ij, jk, ki -> i', A, K_y, A.T)
        se = norm_h - 2 * product_h_y + norm_y

        mse = torch.mean(se)

        return mse
    
    def sloss_batch(self, K_x_tr_ba, K_y_tr_ba, K_y_ba_ba, verbose=0):

        """
            Compute test MSE TO BE DELETED
        """
        
        # MSE
        n_ba = K_x_tr_ba.shape[1]
        if self.n_anchors_krr == -1:
            A = K_x_tr_ba.T @ self.Omega
        else:
            A = self.Omega @ K_x_tr_ba
            A = A.T

        # \|psi(y)\|^2
        norm_y = torch.diag(K_y_ba_ba)

        product_h_y = torch.einsum('ij, ji -> i', A, K_y_tr_ba)
        norm_h = torch.einsum('ij, jk, ki -> i', A, self.K_y, A.T)
        se = norm_h - 2 * product_h_y + norm_y

        if verbose == 1:
            print(torch.sqrt(norm_h[:5]), product_h_y[:5], torch.sqrt(norm_y[:5]))
            print(se[:5])

        mse = torch.mean(se)

        return mse
    
    def sloss_batch_linear(self, X_ba, K_y_tr_ba, K_y_ba_ba, verbose=0):

        """
            Compute test MSE TO BE DELETED
        """
        
        # MSE
        n_ba = X_ba.shape[0]
        A = X_ba @ self.Omega

        # \|psi(y)\|^2
        norm_y = torch.diag(K_y_ba_ba)

        product_h_y = torch.einsum('ij, ji -> i', A, K_y_tr_ba)
        norm_h = torch.einsum('ij, jk, ki -> i', A, self.K_y, A.T)
        se = norm_h - 2 * product_h_y + norm_y

        if verbose == 1:
            print(torch.sqrt(norm_h[:5]), product_h_y[:5], torch.sqrt(norm_y[:5]))
            print(se[:5])

        mse = torch.mean(se)

        return mse

    def mse(self, K_x_tr_te, K_y_tr_te, K_y_te_te, verbose=0):

        """
            Compute test MSE TO BE DELETED
        """

        # MSE
        n_te = K_x_tr_te.shape[1]
        if self.n_anchors_krr == -1:
            A = K_x_tr_te.T @ self.Omega
        else:
            A = self.Omega @ K_x_tr_te
            A = A.T

        # \|psi(y)\|^2
        norm_y = torch.diag(K_y_te_te)

        product_h_y = torch.einsum('ij, ji -> i', A, K_y_tr_te)
        norm_h = torch.einsum('ij, jk, ki -> i', A, self.K_y, A.T)
        se = norm_h - 2 * product_h_y + norm_y

        if verbose == 1:
            print(torch.sqrt(norm_h[:5]), product_h_y[:5], torch.sqrt(norm_y[:5]))
            print(se[:5])

        mse = torch.mean(se)
        std = torch.std(se) / n_te**(1/2)

        return mse.item(), std.item()
    
    def mse_linear(self, X_te, K_y_tr_te, K_y_te_te, verbose=0):

        """
            Compute test MSE TO BE DELETED
        """

        # MSE
        n_te = X_te.shape[0]
        A = X_te @ self.Omega

        # \|psi(y)\|^2
        norm_y = torch.diag(K_y_te_te)

        product_h_y = torch.einsum('ij, ji -> i', A, K_y_tr_te)
        norm_h = torch.einsum('ij, jk, ki -> i', A, self.K_y, A.T)
        se = norm_h - 2 * product_h_y + norm_y

        if verbose == 1:
            print(torch.sqrt(norm_h[:5]), product_h_y[:5], torch.sqrt(norm_y[:5]))
            print(se[:5])

        mse = torch.mean(se)
        std = torch.std(se) / n_te**(1/2)

        return mse.item(), std.item()

    def decode_structured_loss(self, K_x_tr_te, Y_test, Y_candidates=None, problem='multi_label'):

        """
           Predict structured object by decoding, then compute structured losses
        """
        
        t0 = time()
        
        if Y_candidates is None:
            Y_candidates = self.Y_tr.data

        n_te = K_x_tr_te.shape[1]
        structured_losses = []
        kernel_losses = []
        Y_pred = []

        K_y = self.K_y

        for i in range(n_te):

            k_x = K_x_tr_te[:, i]
            y_te = Y_test[i]

            # Compute distances:
            # Compute |Ph(x)|^2 ,  <Ph(x) | psi(y_c)>, and |psi(y_c)|^2  (or |P psi(y_c)|^2 (Weston et al.))
            if self.n_anchors_krr == -1:
                A = k_x.T @ self.Omega
            else:
                A = self.Omega @ k_x
                A = A.T

            K_h_c = A @ self.output_kernel.compute_gram(self.Y_tr, Y_candidates)
            prod_h_c = K_h_c
            K_h_h = A @ K_y @ A.T
            norm_h = K_h_h
            norm_c = torch.diag(self.output_kernel.compute_gram(Y_candidates))

            se = norm_h.view(-1, 1) - 2 * prod_h_c + norm_c.view(1, -1)
            scores = - se

            # Predict and compute Structured losses
            idx_pred = torch.argmax(scores)

            # F1 if multi-label problem
            if problem == 'multi_label':
                y_pred = Y_candidates[idx_pred]
                a = torch.sum((y_pred + y_te == 2)).item()
                b = torch.sum((y_pred + y_te >= 1)).item()
                if a + b > 0:
                    f1 = 2 * a / (a + b)
                else:
                    f1 = 0.
                structured_losses.append(100 * f1)

                y_pred = Y_candidates[idx_pred]
                k_pred_te = self.output_kernel.compute_gram(y_te.view(1, -1), y_pred.view(1, -1))
                kernel_loss = 2 - 2 * k_pred_te
                kernel_losses.append(kernel_loss)

            elif problem == 'image_reconstruction':
                structured_losses.append(-1)
                y_pred = Y_candidates[idx_pred]
                k_pred_te = self.output_kernel.compute_gram(y_te.view(1, -1), y_pred.view(1, -1))
                Y_pred.append(y_pred)
                kernel_loss = 2 - 2 * k_pred_te
                kernel_losses.append(kernel_loss)

        kernel_loss = np.mean(kernel_losses)
        kernel_loss_std = np.std(kernel_losses) / np.sqrt(n_te)
        structured_loss = np.mean(structured_losses)
        structured_std = np.std(structured_losses) / np.sqrt(n_te)
        
        #print(f'Decoding time in minutes: {(time() - t0)/60.0}', flush=True)

        return Y_pred, kernel_loss, kernel_loss_std, structured_loss, structured_std
    
    def decode_structured_loss_metabolites(self, K_x_tr_te, Y_test,formulas_te,
                                           keys_te, n_max=1e5, verbose=1):

        """
           Predict with decoding and compute structured losses
        """

        # Estimation and Decoding
        n_te = K_x_tr_te.shape[1]
        acc_te = np.array([0., 0., 0., 0.])
        rbf_loss_te = []
        ham_loss_te = []
        n_pred = 0
        per = -1
        
        K_y = self.K_y

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

            # Estimate and Compute scores <Ph(x) | psi(y_c)>
            A = k_x_tr_te.T @ self.Omega

            K_h_c = A @ self.output_kernel.compute_gram(self.Y_tr, Y_candidates)
            prod_h_c = K_h_c
            K_h_h = A @ K_y @ A.T
            norm_h = K_h_h
            norm_c = torch.diag(self.output_kernel.compute_gram(Y_candidates))

            se = norm_h.view(-1, 1) - 2 * prod_h_c + norm_c.view(1, -1)
            scores = - se
            scores = scores.data.numpy()

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
            k_pred_te = self.output_kernel.compute_gram(y_te, Y_pred)
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
        print(f'n predictions : {n_pred}')

        return rbf_loss_te, rbf_std_te, acc_te, ham_te
    
    def decode_structured_loss_linear(self, X_test, Y_test, Y_candidates=None, problem='multi_label'):

        """
           Predict structured object by decoding, then compute structured losses
        """
        
        t0 = time()
        
        if Y_candidates is None:
            Y_candidates = self.Y_tr.data

        n_te = X_test.shape[0]
        structured_losses = []
        kernel_losses = []
        Y_pred = []

        K_y = self.K_y
        
        X_te = self.input_kernel.model_forward(X_test)

        for i in range(n_te):

            x = X_te[i]
            y_te = Y_test[i]

            # Compute distances:
            # Compute |Ph(x)|^2 ,  <Ph(x) | psi(y_c)>, and |psi(y_c)|^2  (or |P psi(y_c)|^2 (Weston et al.))
            A = x @ self.Omega

            K_h_c = A @ self.output_kernel.compute_gram(self.Y_tr, Y_candidates)
            prod_h_c = K_h_c
            K_h_h = A @ K_y @ A.T
            norm_h = K_h_h
            norm_c = torch.diag(self.output_kernel.compute_gram(Y_candidates))

            se = norm_h.view(-1, 1) - 2 * prod_h_c + norm_c.view(1, -1)
            scores = - se

            # Predict and compute Structured losses
            idx_pred = torch.argmax(scores)

            # F1 if multi-label problem
            if problem == 'multi_label':
                y_pred = Y_candidates[idx_pred]
                a = torch.sum((y_pred + y_te == 2)).item()
                b = torch.sum((y_pred + y_te >= 1)).item()
                if a + b > 0:
                    f1 = 2 * a / (a + b)
                else:
                    f1 = 0.
                structured_losses.append(100 * f1)

                y_pred = Y_candidates[idx_pred]
                k_pred_te = self.output_kernel.compute_gram(y_te.view(1, -1), y_pred.view(1, -1))
                kernel_loss = 2 - 2 * k_pred_te
                kernel_losses.append(kernel_loss)

            elif problem == 'image_reconstruction':
                structured_losses.append(-1)
                y_pred = Y_candidates[idx_pred]
                k_pred_te = self.output_kernel.compute_gram(y_te.view(1, -1), y_pred.view(1, -1))
                Y_pred.append(y_pred)
                kernel_loss = 2 - 2 * k_pred_te
                kernel_losses.append(kernel_loss)

        kernel_loss = np.mean(kernel_losses)
        kernel_loss_std = np.std(kernel_losses) / np.sqrt(n_te)
        structured_loss = np.mean(structured_losses)
        structured_std = np.std(structured_losses) / np.sqrt(n_te)
        
        #print(f'Decoding time in minutes: {(time() - t0)/60.0}', flush=True)

        return Y_pred, kernel_loss, kernel_loss_std, structured_loss, structured_std

    def cross_validation_score(self, X_s, Y_s, Y_u=None, K_X_s=None, n_folds=5, L=None, input_gamma=None, input_kernel=None,
                               output_kernel=None, method=None, q=None, decode=False, problem=None):

        res = np.zeros((n_folds, 4))
        
        kfold = KFold(n_splits=n_folds)
        
        i=0

        test_size = 0.2
        for train_index, test_index in kfold.split(X_s):
            # Split
            X_train, X_test = X_s[train_index], X_s[test_index]
            Y_train, Y_test = Y_s[train_index], Y_s[test_index]


            # Fit
            self.fit(X_s=X_train, Y_s=Y_train, Y_u=Y_u, K_X_s=K_X_s, L=L,
                     oel_method=method, q=q, input_gamma=input_gamma, input_kernel=input_kernel, output_kernel=output_kernel)

            # MSE (and structured loss)
            K_x_tr_te = self.input_kernel.compute_gram(self.X_tr, Y=X_test)

            if method is None:
                K_y_tr_te = self.output_kernel.compute_gram(Y_train, Y_test)
                K_y_te_te = self.output_kernel.compute_gram(Y_test, Y_test)
            else:
                UY_test = self.oel.transform(Y_test)
                K_y_tr_te = self.UY_tr.dot(UY_test.T)
                K_y_te_te = self.output_kernel(Y_test, Y_test)

            # mse_train, mse_std_train = self.mse(self.K_x, self.K_y, self.output_kernel(Y_train, Y_train))
            mse_train, mse_std_train = -1, -1
            mse_test, mse_std_test = self.mse(K_x_tr_te, K_y_tr_te, K_y_te_te)
            
            Y_candidates = Y_train

            rbf_loss_test, structured_loss_test = -1, -1
            if decode:
                Y_pred, rbf_loss_test, rbf_std_test, structured_loss_test, structured_std_test \
                    = self.decode_structured_loss(K_x_tr_te, Y_test, q=q, Y_candidates=Y_candidates,
                                                  problem=problem)

            res[i, 0] = mse_train  # Train
            res[i, 1] = mse_test  # Test
            res[i, 2] = structured_loss_test  # Test
            res[i, 3] = rbf_loss_test  # Test
            
            i += 1

        res_avg = np.mean(res, axis=0)

        return res_avg

    def cross_validation_score_list(self, X_s, Y_s, Y_u=None, n_folds=5, L=None, input_gamma=None, output_kernel=None,
                                    oel_method=None, qs=None, c=None, center_output=False):

        n_params = len(qs)
        q_max = max(qs)

        if oel_method is None:
                qs = [None]
                q_max = None
                n_params = 1
        res = np.zeros((n_folds, 2, n_params))


        test_size = 0.2
        for i in range(n_folds):
            # Split
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_s, Y_s, test_size=test_size, random_state=i)

            # Fit
            self.fit(X_s=X_train, Y_s=Y_train, Y_u=Y_u, L=L, oel_method=oel_method, q=q_max, c=c,
                     input_gamma=input_gamma, output_kernel=output_kernel, center_output=center_output)

            # MSE
            for i_q, q in enumerate(qs):

                # K_x_tr_tr = rbf_kernel(X_train, Y=X_train, gamma=self.input_gamma)
                # mse_train, mse_std_train = self.mse2(K_x_tr_tr, Y_train, Y_train, q,
                #                                     center=center_output, center_before=center_before)
                mse_train, mse_std_train = -1, -1

                K_x_tr_te = rbf_kernel(X_train, Y=X_test, gamma=self.input_gamma)
                mse_test, mse_std_test = self.mse2(K_x_tr_te, Y_train, Y_test, q, center_output=center_output)

                res[i, 0, i_q] = mse_train  # Train
                res[i, 1, i_q] = mse_test  # Test


        res_avg = np.mean(res, axis=0)

        return res_avg