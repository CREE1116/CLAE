"""
Define models here
"""
import world
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np
from scipy import sparse
from time import time
import gc


def get_valid_score(model, dataset):
    return 0.0, 0.0


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

    def _to_torch_sparse(self, scipy_matrix):
        """Scipy CSR/CSC matrix를 Torch Sparse COO Tensor로 변환 (toarray 방지)"""
        scipy_matrix = scipy_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((scipy_matrix.row, scipy_matrix.col)).astype(np.int64))
        values = torch.from_numpy(scipy_matrix.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(scipy_matrix.shape))

    def _get_batch_ratings(self, users, test_matrix, W_gpu):
        """Sparse-Dense 행렬곱을 활용한 효율적인 등급 계산"""
        # 1. 필요한 사용자 행만 슬라이싱 (여전히 Scipy Sparse)
        batch_sp = test_matrix[users.cpu().numpy()]
        # 2. Torch Sparse로 변환하여 GPU로 전송
        batch_torch = self._to_torch_sparse(batch_sp).to(W_gpu.device)
        # 3. GPU에서 Sparse-Dense 행렬곱 수행
        ratings = torch.sparse.mm(batch_torch, W_gpu)
        return ratings.cpu()

    def _compute_aspire_gram(self, X_sp, alpha, eps=1e-12):
        """ASPIRE (A Structured Proxy IPS) 가중치가 적용된 Gram Matrix 계산 (CPU)"""
        print(f"  computing ASPIRE Gram Matrix (alpha={alpha})...")
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        u_weights = (1.0 / (np.power(n_u, alpha) + eps)).astype(np.float32)
        n_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        i_weights = np.power(n_i + eps, -alpha / 2.0).astype(np.float32)
        X_weighted = sparse.diags(u_weights) @ X_sp
        G = (X_sp.T @ X_weighted).toarray()
        G = G * i_weights[:, None] * i_weights[None, :]
        del n_u, n_i, u_weights, X_weighted
        gc.collect()
        return G, i_weights

    def _compute_dan_gram(self, X_sp, alpha, beta, eps=1e-12):
        """DAN (Degree-Aware Normalization) 가중치가 적용된 Gram Matrix 계산 (CPU)"""
        print(f"  computing DAN Gram Matrix (alpha={alpha}, beta={beta})...")
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        u_weights = np.power(n_u + eps, -beta).astype(np.float32)
        n_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        X_weighted = sparse.diags(u_weights) @ X_sp
        G = (X_sp.T @ X_weighted).toarray()
        return G, n_i


# ──────────────────────────────────────────────────────────────────────────────
# Base Models
# ──────────────────────────────────────────────────────────────────────────────

class EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.diag_const = config.get('diag_const', True)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        print(f"Fitting EASE on {self.device}...")
        G_sp = X.T @ X
        G = torch.from_numpy(G_sp.toarray()).to(torch.float32).to(self.device)
        G.diagonal().add_(self.reg_p)
        P = torch.linalg.inv(G)
        if self.diag_const:
            self.W_gpu = P / (-torch.diagonal(P) + 1e-12)
        else:
            self.W_gpu = P * -self.reg_p
        self.W_gpu.diagonal().zero_()
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class LAE(BasicModel):
    """Linear AutoEncoder: Ridge Regression without diagonal constraint"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(LAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        print(f"Fitting LAE on {self.device}...")
        G_raw = (X.T @ X).toarray()
        G = G_raw.copy()
        G[np.diag_indices(G.shape[0])] += self.reg_p
        P_inv = np.linalg.inv(G)
        # W = (X.T X + lambda I)^-1 * X.T X
        W = P_inv @ G_raw
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class RLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.xi = config.get('xi', 0.0)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        G = (X.T @ X).toarray()
        G[np.diag_indices(X.shape[1])] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        condition = (1 - self.reg_p * diag_P) > self.xi
        lagrangian = ((1 - self.xi) / (diag_P + 1e-12) - self.reg_p) * condition.astype(float)
        W = P * -(lagrangian + self.reg_p)
        np.fill_diagonal(W, 0)
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class DLAE(BasicModel):
    """Dropout LAE (Unconstrained)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.dropout_p = config.get('dropout_p', 0.3)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        print(f"Fitting DLAE on {self.device}...")
        G_raw = (X.T @ X).toarray()
        p = min(self.dropout_p, 0.99)
        # Lambda = p/(1-p) * D_I
        Lambda = (p / (1.0 - p)) * np.diag(G_raw)
        G = G_raw.copy()
        G[np.diag_indices(G.shape[0])] += Lambda
        P_inv = np.linalg.inv(G)
        # W = (X.T X + Lambda)^-1 * X.T X
        W = P_inv @ G_raw
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

# ──────────────────────────────────────────────────────────────────────────────
# DAN Models
# ──────────────────────────────────────────────────────────────────────────────

class DAN_EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAN_EASE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.5)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        G, n_i = self._compute_dan_gram(X, self.alpha, self.beta)
        G[np.diag_indices(X.shape[1])] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        W = -P / (diag_P + 1e-12)
        item_power_term = np.power(n_i + 1e-12, -(1 - self.alpha))
        W = W * (1.0/(item_power_term + 1e-12)).reshape(-1, 1) * item_power_term
        np.fill_diagonal(W, 0)
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class DAN_LAE(BasicModel):
    """DAN-LAE (Unconstrained)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAN_LAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.5)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        print(f"Fitting DAN_LAE on {self.device}...")
        P_dan_raw, _ = self._compute_dan_gram(X, self.alpha, self.beta)
        G = P_dan_raw.copy()
        G[np.diag_indices(G.shape[0])] += self.reg_p
        P_inv = np.linalg.inv(G)
        # W = (P_dan + lambda I)^-1 * P_dan
        W = P_inv @ P_dan_raw
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class DAN_RLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAN_RLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.5)
        self.xi = config.get('xi', 0.0)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        G, n_i = self._compute_dan_gram(X, self.alpha, self.beta)
        G[np.diag_indices(X.shape[1])] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        condition = (1 - self.reg_p * diag_P) > self.xi
        lagrangian = ((1 - self.xi) / (diag_P + 1e-12) - self.reg_p) * condition.astype(float)
        W = P * -(lagrangian + self.reg_p)
        item_power_term = np.power(n_i + 1e-12, -(1 - self.alpha))
        W = W * (1.0/(item_power_term + 1e-12)).reshape(-1, 1) * item_power_term
        np.fill_diagonal(W, 0)
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class DAN_DLAE(BasicModel):
    """DAN-DLAE (Unconstrained, Removed reg_p)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAN_DLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.5)
        self.dropout_p = config.get('dropout_p', 0.3)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        print(f"Fitting DAN_DLAE on {self.device}...")
        P_dan_raw, _ = self._compute_dan_gram(X, self.alpha, self.beta)
        p = min(self.dropout_p, 0.99)
        w_dropout = (p / (1.0 - p)) * np.diag(P_dan_raw)
        G = P_dan_raw.copy()
        G[np.diag_indices(G.shape[0])] += w_dropout
        P_inv = np.linalg.inv(G)
        # W = (P_dan + Lambda)^-1 * P_dan
        W = P_inv @ P_dan_raw
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# ASPIRE Models
# ──────────────────────────────────────────────────────────────────────────────

class ASPIRE_EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(ASPIRE_EASE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.train_matrix = X_sp
        train_start = time()
        G, _ = self._compute_aspire_gram(X_sp, self.alpha)
        G[np.diag_indices_from(G)] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        P /= -(diag_P + 1e-12)
        np.fill_diagonal(P, 0)
        self.W_gpu = torch.tensor(P, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class ASPIRE_LAE(BasicModel):
    """ASPIRE-LAE (Unconstrained)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(ASPIRE_LAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.train_matrix = X_sp
        train_start = time()
        print(f"Fitting ASPIRE_LAE on {self.device}...")
        G_aspire, _ = self._compute_aspire_gram(X_sp, self.alpha)
        G = G_aspire.copy()
        G[np.diag_indices_from(G)] += self.reg_lambda
        P_inv = np.linalg.inv(G)
        # W = (G_aspire + lambda I)^-1 * G_aspire
        W = P_inv @ G_aspire
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class ASPIRE_RLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(ASPIRE_RLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.xi         = config.get('xi', 0.0)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.train_matrix = X_sp
        train_start = time()
        G, _ = self._compute_aspire_gram(X_sp, self.alpha)
        G[np.diag_indices_from(G)] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        condition = (1.0 - self.reg_lambda * diag_P) > self.xi
        lagrangian = ((1.0 - self.xi) / (diag_P + 1e-12) - self.reg_lambda) * condition.astype(float)
        W = P * -(lagrangian + self.reg_lambda)
        np.fill_diagonal(W, 0)
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)

class ASPIRE_DLAE(BasicModel):
    """ASPIRE-DLAE (Unconstrained, Removed reg_lambda)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(ASPIRE_DLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.alpha      = config.get('alpha', 0.5)
        self.dropout_p  = config.get('dropout_p', 0.3)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.train_matrix = X_sp
        train_start = time()
        print(f"Fitting ASPIRE_DLAE on {self.device}...")
        G_aspire, _ = self._compute_aspire_gram(X_sp, self.alpha)
        p = min(self.dropout_p, 0.99)
        w_dropout = (p / (1.0 - p)) * np.diag(G_aspire)
        G = G_aspire.copy()
        G[np.diag_indices_from(G)] += w_dropout
        P_inv = np.linalg.inv(G)
        # W = (G_aspire + Lambda)^-1 * G_aspire
        W = P_inv @ G_aspire
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.train_matrix, self.W_gpu)


class RDLAE(BasicModel):
    """RDLAE: Lagrangian Dropout LAE (Removed reg_p)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RDLAE, self).__init__()
        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.drop_p = config['drop_p']
        self.xi = config['xi']
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        G = np.array(X.T.dot(X).toarray())
        # gamma = p/(1-p) * D_I (Removed reg_p)
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p)
        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)
        diag_C = np.diag(C)
        condition = (1 - gamma * diag_C) > self.xi
        lagrangian = ((1 - self.xi) / (diag_C + 1e-12) - gamma) * condition.astype(float)
        self.W = C * -(gamma + lagrangian)
        self.W[np.diag_indices(self.num_items)] = 0
        train_end = time()
        self.train_time = train_end - train_start

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()
        input_matrix = np.array(self.train_matrix[users].toarray())
        eval_output = input_matrix @ self.W
        return torch.FloatTensor(eval_output)


class EDLAE(BasicModel):
    """EDLAE: EASE-style Dropout LAE (Removed reg_p)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EDLAE, self).__init__()
        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.drop_p = config['drop_p']
        self.diag_const = config.get('diag_const', True)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        train_start = time()
        G = np.array(X.T.dot(X).toarray())
        # gamma = p/(1-p) * D_I (Removed reg_p)
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p)
        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)
        if self.diag_const:
            self.W = C / (-np.diag(C) + 1e-12)
        else:
            self.W = -C * gamma
        self.W[np.diag_indices(self.num_items)] = 0
        train_end = time()
        self.train_time = train_end - train_start

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()
        input_matrix = np.array(self.train_matrix[users].toarray())
        eval_output = input_matrix @ self.W
        return torch.FloatTensor(eval_output)
    

class GFCF(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(GFCF, self).__init__()
        self.dataset = dataset
        self.adj_mat = dataset.UserItemNet.tolil()
        self.alpha = config.get('alpha', 0.0)
        self.__init_weight()
    
    def __init_weight(self):
        adj_mat = self.adj_mat
        train_start = time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sparse.diags(1/(d_inv+1e-12))
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        import scipy.sparse.linalg as linalg
        _, _, self.vt = linalg.svds(self.norm_adj, 256)
        train_end = time()
        self.train_time = train_end - train_start
        
    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if world.dataset == 'abook':
            ret = U_2
        else:
            U_1 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + self.alpha * U_1
        return torch.FloatTensor(ret)
