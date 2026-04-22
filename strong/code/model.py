"""
Define models here
"""
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np
from scipy import sparse
from time import time
from Procedure import get_valid_score
import gc


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

    def _compute_daspire_gram(self, X_sp, alpha, beta, eps=1e-12):
        """DAspire (Decoupled ASPIRE) 가중치가 적용된 Gram Matrix 계산 (CPU)
        alpha: Item degree exponent (tuned in 0~0.5)
        beta: User degree exponent (tuned in 0~1.0)
        """
        print(f"  computing DAspire Gram Matrix (alpha={alpha}, beta={beta})...")
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        u_weights = (1.0 / (np.power(n_u, beta) + eps)).astype(np.float32)
        
        n_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        i_weights = np.power(n_i + eps, -alpha).astype(np.float32)
        
        X_weighted = sparse.diags(u_weights) @ X_sp
        G = (X_sp.T @ X_weighted).toarray()
        G = G * i_weights[:, None] * i_weights[None, :]
        
        del n_u, n_i, u_weights, X_weighted
        gc.collect()
        return G, i_weights

    def _compute_dan_gram(self, X_sp, alpha, beta, eps=1e-12):
        """DAN (Degree-Aware Normalization) 가중치가 적용된 Gram Matrix 계산 (CPU)"""
        print(f"  computing DAN Gram Matrix (alpha={alpha}, beta={beta})...")
        # 1. User Degree Weighting: D_U^-beta
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        u_weights = np.power(n_u + eps, -beta).astype(np.float32)
        X_weighted = sparse.diags(u_weights) @ X_sp
        G_raw = (X_sp.T @ X_weighted).toarray()
        
        # 2. Item Degree Scaling: D_I^{-(1-alpha)} * G_raw * D_I^-alpha
        n_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        d_left = np.power(n_i + eps, -(1.0 - alpha))
        d_right = np.power(n_i + eps, -alpha)
        G = d_left[:, None] * G_raw * d_right[None, :]
        
        return G, n_i

    def _compute_inv_propensity(self, X, wbeta, wtype='logsigmoid'):
        """공식 코드의 인기도 가중치 계산 로직"""
        freqs = np.ravel(X.sum(axis=0)).astype(np.float32)
        
        if wtype == 'logsigmoid':
            log_freqs = np.log(freqs + 1)
            min_log = np.min(log_freqs)
            max_log = np.max(log_freqs)
            # 공식 코드의 alpha 및 logits 계산
            alpha = -wbeta * (min_log + max_log) / 2
            logits = alpha + wbeta * log_freqs
            p_i = 1 / (1 + np.exp(-logits))
            inv_p = 1 / (p_i + 1e-12)
        elif wtype == 'powerlaw':
            # 공식 코드의 powerlaw 계산
            norm_pop = freqs / (np.max(freqs) + 1e-12)
            p = np.power(norm_pop, wbeta)
            inv_p = 1 / (p + 1e-12)
        else:
            inv_p = np.ones_like(freqs)
            
        return inv_p

# ──────────────────────────────────────────────────────────────────────────────
# Base Models
# ──────────────────────────────────────────────────────────────────────────────

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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
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
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class EASE(BasicModel):
    """EASE: Ridge Regression WITH diagonal constraint (diag=0)"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config.get('reg_p', 100.0)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting EASE on {self.device}...")
        G_sp = X.T @ X
        G = G_sp.toarray()
        G[np.diag_indices(G.shape[0])] += self.reg_p
        P = np.linalg.inv(G)
        # Constrained: B = I - P * diag(P)^-1
        diag_P = np.diag(P)
        W = -P / (diag_P + 1e-12)
        np.fill_diagonal(W, 0)
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class RLAE(BasicModel):
    """RLAE: Lagrangian Relaxation with diagonal constraint"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
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
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DLAE(BasicModel):
    """Dropout LAE: (X.T X + Lambda)^-1 * X.T X where Lambda = p/(1-p) * D_I"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.dropout_p = config.get('dropout_p', 0.3)
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DLAE on {self.device}...")
        
        G_raw = (X.T @ X).toarray()
        n_i = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        
        p = min(self.dropout_p, 0.99)
        # Lambda = p/(1-p) * D_I
        Lambda = (p / (1.0 - p)) * n_i
        
        G = G_raw.copy()
        G[np.diag_indices(G.shape[0])] += Lambda
        P_inv = np.linalg.inv(G)
        
        # W = (X.T X + Lambda)^-1 * X.T X
        W = P_inv @ G_raw
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# DAN Models
# ──────────────────────────────────────────────────────────────────────────────

class DAN_LAE(BasicModel):
    """DAN-LAE: (P_dan + lambda I)^-1 * P_dan"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
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
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DAN_EASE(BasicModel):
    """DAN-EASE: Constrained DAN (diag=0)"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAN_EASE on {self.device}...")
        
        P_dan, _ = self._compute_dan_gram(X, self.alpha, self.beta)
        G = P_dan.copy()
        G[np.diag_indices(G.shape[0])] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        W = -P / (diag_P + 1e-12)
        np.fill_diagonal(W, 0)
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAN_RLAE on {self.device}...")
        
        P_dan, _ = self._compute_dan_gram(X, self.alpha, self.beta)
        G = P_dan.copy()
        G[np.diag_indices(G.shape[0])] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        condition = (1 - self.reg_p * diag_P) > self.xi
        lagrangian = ((1 - self.xi) / (diag_P + 1e-12) - self.reg_p) * condition.astype(float)
        W = P * -(lagrangian + self.reg_p)
        np.fill_diagonal(W, 0)
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DAN_DLAE(BasicModel):
    """DAN-DLAE: Combined DAN and Dropout Penalty (Unconstrained)"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAN_DLAE on {self.device}...")
        
        P_dan_raw, n_i = self._compute_dan_gram(X, self.alpha, self.beta)
        
        p = min(self.dropout_p, 0.99)
        # Lambda = p/(1-p) * diag(P_dan)
        w_dropout = (p / (1.0 - p)) * np.diag(P_dan_raw)
        
        G = P_dan_raw.copy()
        G[np.diag_indices(G.shape[0])] += w_dropout
        P_inv = np.linalg.inv(G)
        
        # W = (P_dan + Lambda)^-1 * P_dan
        W = P_inv @ P_dan_raw
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# ASPIRE Models
# ──────────────────────────────────────────────────────────────────────────────

class ASPIRE_LAE(BasicModel):
    """ASPIRE-LAE: Unconstrained ASPIRE"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
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
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class ASPIRE_EASE(BasicModel):
    """ASPIRE-EASE: Constrained ASPIRE"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        G, _ = self._compute_aspire_gram(X_sp, self.alpha)
        G[np.diag_indices_from(G)] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        P /= -(diag_P + 1e-12)
        np.fill_diagonal(P, 0)
        self.W_gpu = torch.tensor(P, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
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
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class ASPIRE_DLAE(BasicModel):
    """ASPIRE-DLAE: Unconstrained ASPIRE with Dropout Penalty"""
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
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting ASPIRE_DLAE on {self.device}...")
        
        G_aspire, _ = self._compute_aspire_gram(X_sp, self.alpha)
        
        p = min(self.dropout_p, 0.99)
        # Lambda = p/(1-p) * diag(G_aspire)
        w_dropout = (p / (1.0 - p)) * np.diag(G_aspire)
        
        G = G_aspire.copy()
        G[np.diag_indices_from(G)] += w_dropout
        P_inv = np.linalg.inv(G)
        
        # W = (G_aspire + Lambda)^-1 * G_aspire
        W = P_inv @ G_aspire
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DAspire_EASE(BasicModel):
    """DAspire-EASE: Decoupled ASPIRE with Constrained Ridge Regression"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAspire_EASE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5) # Item degree
        self.beta       = config.get('beta', 0.5)  # User degree
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAspire_EASE on {self.device}...")
        G, _ = self._compute_daspire_gram(X_sp, self.alpha, self.beta)
        G[np.diag_indices_from(G)] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        P /= -(diag_P + 1e-12)
        np.fill_diagonal(P, 0)
        self.W_gpu = torch.tensor(P, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DAspire_LAE(BasicModel):
    """DAspire-LAE: Decoupled ASPIRE with Unconstrained Ridge Regression"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAspire_LAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.beta       = config.get('beta', 0.5)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAspire_LAE on {self.device}...")
        G_aspire, _ = self._compute_daspire_gram(X_sp, self.alpha, self.beta)
        
        G = G_aspire.copy()
        G[np.diag_indices_from(G)] += self.reg_lambda
        P_inv = np.linalg.inv(G)
        
        # W = (G_aspire + lambda I)^-1 * G_aspire
        W = P_inv @ G_aspire
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DAspire_RLAE(BasicModel):
    """DAspire-RLAE: Decoupled ASPIRE with Lagrangian Relaxation"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAspire_RLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.beta       = config.get('beta', 0.5)
        self.xi         = config.get('xi', 0.0)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAspire_RLAE on {self.device}...")
        G, _ = self._compute_daspire_gram(X_sp, self.alpha, self.beta)
        G[np.diag_indices_from(G)] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        condition = (1.0 - self.reg_lambda * diag_P) > self.xi
        lagrangian = ((1.0 - self.xi) / (diag_P + 1e-12) - self.reg_lambda) * condition.astype(float)
        W = P * -(lagrangian + self.reg_lambda)
        np.fill_diagonal(W, 0)
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class DAspire_DLAE(BasicModel):
    """DAspire-DLAE: Decoupled ASPIRE with Dropout Penalty"""
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DAspire_DLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.alpha      = config.get('alpha', 0.5)
        self.beta       = config.get('beta', 0.5)
        self.dropout_p  = config.get('dropout_p', 0.3)
        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        print(f"Fitting DAspire_DLAE on {self.device}...")
        
        G_aspire, _ = self._compute_daspire_gram(X_sp, self.alpha, self.beta)
        
        p = min(self.dropout_p, 0.99)
        # Lambda = p/(1-p) * diag(G_aspire)
        w_dropout = (p / (1.0 - p)) * np.diag(G_aspire)
        
        G = G_aspire.copy()
        G[np.diag_indices_from(G)] += w_dropout
        P_inv = np.linalg.inv(G)
        
        # W = (G_aspire + Lambda)^-1 * G_aspire
        W = P_inv @ G_aspire
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy / Others
# ──────────────────────────────────────────────────────────────────────────────

class GFCF(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(GFCF, self).__init__()
        self.dataset = dataset
        self.alpha = config['alpha']
        self.__init_weight()
    
    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        
        # Simple GFCF implementation (SVD on Normalized Adj)
        import scipy.sparse.linalg as linalg
        
        rowsum = np.array(X_sp.sum(axis=1)).ravel()
        d_inv = np.power(rowsum + 1e-12, -0.5)
        D_u = sparse.diags(d_inv)
        
        colsum = np.array(X_sp.sum(axis=0)).ravel()
        d_inv_i = np.power(colsum + 1e-12, -0.5)
        D_i = sparse.diags(d_inv_i)
        
        A_norm = D_u @ X_sp @ D_i
        U, S, Vt = linalg.svds(A_norm.tocsc(), k=256)
        
        self.W = (D_i @ Vt.T @ Vt @ sparse.diags(1.0/(d_inv_i + 1e-12))).toarray()
        self.W_gpu = torch.tensor(self.W, dtype=torch.float32, device=self.device)
        
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)

class EASE_DAN(DAN_EASE): 
    pass

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
        self.train_time = time() - train_start
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()
        input_matrix = np.array(self.test_matrix[users].toarray())
        eval_output = input_matrix @ self.W
        return torch.FloatTensor(eval_output)

    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()
        input_matrix = np.array(self.valid_matrix[users].toarray())
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
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        train_start = time()
        G = np.array(X.T.dot(X).toarray())
        # gamma = p/(1-p) * D_I (Removed reg_p)
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p)
        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)
        self.W = C / (-np.diag(C) + 1e-12)
        self.W[np.diag_indices(self.num_items)] = 0
        self.train_time = time() - train_start
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()
        input_matrix = np.array(self.test_matrix[users].toarray())
        eval_output = input_matrix @ self.W
        return torch.FloatTensor(eval_output)

    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()
        input_matrix = np.array(self.valid_matrix[users].toarray())
        eval_output = input_matrix @ self.W
        return torch.FloatTensor(eval_output)

class IPS_LAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(IPS_LAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 100.0)
        self.wbeta = config.get('wbeta', 0.5)
        self.wtype = config.get('wtype', 'logsigmoid')
        self.__init_weight()

    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        
        G_raw = (X.T @ X).toarray()
        G = G_raw.copy()
        G[np.diag_indices(G.shape[0])] += self.reg_lambda
        P_inv = np.linalg.inv(G)
        W = P_inv @ G_raw
        
        inv_p = self._compute_inv_propensity(X, self.wbeta, self.wtype)
        W = W * inv_p  # Numpy broadcasting
        np.fill_diagonal(W, 0)
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class IPS_EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(IPS_EASE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 100.0)
        self.wbeta = config.get('wbeta', 0.5)
        self.wtype = config.get('wtype', 'logsigmoid')
        self.__init_weight()

    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        
        G = (X.T @ X).toarray()
        G[np.diag_indices(G.shape[0])] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        W = P / -(diag_P + 1e-12)
        
        inv_p = self._compute_inv_propensity(X, self.wbeta, self.wtype)
        W = W * inv_p
        np.fill_diagonal(W, 0)
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class IPS_RLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(IPS_RLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_lambda = config.get('reg_lambda', 100.0)
        self.wbeta = config.get('wbeta', 0.5)
        self.wtype = config.get('wtype', 'logsigmoid')
        self.xi = config.get('xi', 0.0)
        self.__init_weight()

    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        
        G = (X.T @ X).toarray()
        G[np.diag_indices(X.shape[1])] += self.reg_lambda
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        condition = (1 - self.reg_lambda * diag_P) > self.xi
        lagrangian = ((1 - self.xi) / (diag_P + 1e-12) - self.reg_lambda) * condition.astype(float)
        W = P * -(lagrangian + self.reg_lambda)
        
        inv_p = self._compute_inv_propensity(X, self.wbeta, self.wtype)
        W = W * inv_p
        np.fill_diagonal(W, 0)
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)


class IPS_DLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(IPS_DLAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        self.dropout_p = config.get('dropout_p', 0.3)
        self.wbeta = config.get('wbeta', 0.5)
        self.wtype = config.get('wtype', 'logsigmoid')
        self.__init_weight()

    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        train_start = time()
        
        G_raw = (X.T @ X).toarray()
        n_i = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        p = min(self.dropout_p, 0.99)
        Lambda = (p / (1.0 - p)) * n_i
        
        G = G_raw.copy()
        G[np.diag_indices(G.shape[0])] += Lambda
        P_inv = np.linalg.inv(G)
        W = P_inv @ G_raw
        
        inv_p = self._compute_inv_propensity(X, self.wbeta, self.wtype)
        W = W * inv_p
        np.fill_diagonal(W, 0)
        
        self.W_gpu = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)
