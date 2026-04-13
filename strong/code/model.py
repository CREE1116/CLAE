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


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError


class GFCF(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(GFCF, self).__init__()

        self.dataset = dataset
        self.alpha = config['alpha']
        self.__init_weight()
    
    def __init_weight(self):
        self.valid_matrix = self.dataset.validUserItemNet
        self.test_matrix = self.dataset.testUserItemNet
        
        adj_mat = self.dataset.UserItemNet.tolil()
        train_start = time()
        
        # generate normalized train matrix
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
        self.d_mat_i_inv = sparse.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        
        import scipy.sparse.linalg as linalg
        # svd for normalized train matrix
        _, _, self.vt = linalg.svds(norm_adj, 256)
        
        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)
        
    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        norm_adj = self.norm_adj
        batch_users = np.array(self.test_matrix[users,:].toarray())
        ret = batch_users @ (norm_adj.T @ norm_adj + self.alpha * self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv)
        
        return torch.FloatTensor(ret)
    
    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()

        norm_adj = self.norm_adj
        batch_users = np.array(self.valid_matrix[users,:].toarray())
        ret = batch_users @ (norm_adj.T @ norm_adj + self.alpha * self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv)

        return torch.FloatTensor(ret)


class RLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.xi = config['xi']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        
        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        G[np.diag_indices(self.num_items)] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)

        condition = (1 - self.reg_p * diag_P) > self.xi
        assert condition.sum() > 0
        lagrangian = ((1 - self.xi) / diag_P - self.reg_p) * condition.astype(float)
        
        self.W = P * -(lagrangian + self.reg_p)
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

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
    

class RDLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RDLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.drop_p = config['drop_p']
        self.xi = config['xi']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        
        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p) + self.reg_p
        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)
        diag_C = np.diag(C)
        
        condition = (1 - gamma * diag_C) > self.xi
        assert condition.sum() > 0
        lagrangian = ((1 - self.xi) / diag_C - gamma) * condition.astype(float)

        self.W = C * -(gamma + lagrangian)
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

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


class EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.diag_const = config['diag_const']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.best_epoch = 0

        train_start = time()
        G = np.array(X.T.dot(X).toarray())
        G[np.diag_indices(self.num_items)] += self.reg_p
        P = np.linalg.inv(G)

        if self.diag_const:
            self.W = P / (-np.diag(P))
        else:
            self.W = P * -self.reg_p
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")
        
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
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EDLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items

        self.reg_p = config['reg_p']
        self.drop_p = config['drop_p']
        self.diag_const = config['diag_const']

        self.__init_weight()

    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()

        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p) + self.reg_p

        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)

        if self.diag_const:
            self.W = C / (-np.diag(C))
        else:
            self.W = C * -gamma
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

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


class CLAE(BasicModel):
    """
    CLAE (Causal Linear AutoEncoder)
    - Stage 1: User-side Fractional IPW (beta) -> Variance Stabilization
    - Stage 2: Item-side Geometric Ensemble Normalization (alpha) -> Causal Ensemble
    - Stage 3: Ridge Regression via Solve -> Faster and more stable
    """
    def __init__(self, config:dict, dataset:BasicDataset):
        super(CLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items

        from world import device
        self.device = device

        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.beta       = config.get('beta', 0.5)
        self.eps        = 1e-12

        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()

        train_start = time()
        print(f"Fitting CLAE (lambda={self.reg_lambda}, alpha={self.alpha}, beta={self.beta}) on {self.device}...")

        # ── Stage 1: User-side Fractional IPW (CPU Sparse) ───────────────────────────
        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        user_weights = np.power(n_u + self.eps, -self.beta)
        D_U_inv = sparse.diags(user_weights)

        X_weighted = D_U_inv @ X_sp                 
        G_U = (X_sp.T @ X_weighted).toarray()       

        # ── Stage 2: Item-side Geometric Ensemble Normalization ──────────────────
        A_i = G_U.diagonal().copy()
        scale = np.power(A_i + self.eps, -self.alpha / 2.0)
        G_tilde = G_U * scale[:, None] * scale[None, :]

        # ── Stage 3: Ridge Regression via Solve (GPU) ────────────────
        K = self.num_items
        G_torch = torch.tensor(G_tilde, dtype=torch.float32, device=self.device)
        A_mat = G_torch + self.reg_lambda * torch.eye(K, device=self.device)

        try:
            # W = (G_tilde + lambda*I)^{-1} @ G_tilde
            self.W = torch.linalg.solve(A_mat, G_torch)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular matrix, applying stronger regularization.")
            A_mat.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            self.W = torch.linalg.solve(A_mat, G_torch)

        # Post-masking (Prevent self-recommendation)
        # self.W.diagonal().zero_()

        # Keep W on CPU for prediction to be consistent with other models if needed, 
        # but let's see if we can keep it on GPU. 
        # The other models use numpy for self.W. 
        self.W = self.W.cpu().numpy()

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

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


class DCLAE(BasicModel):
    """
    DCLAE (Dropout Causal Linear AutoEncoder)
    - Stage 1: User-side Fractional IPW (beta) -> Variance Stabilization
    - Stage 2: Item-side Geometric Ensemble Normalization (alpha) -> Causal Ensemble
    - 🔥 Dropout Regularization: Adds (p/(1-p)) * A_i to diagonal
    - Stage 3: Ridge Regression via Solve -> Faster and more stable
    """
    def __init__(self, config:dict, dataset:BasicDataset):
        super(DCLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items

        from world import device
        self.device = device

        self.reg_lambda = config.get('reg_lambda', 10.0) 
        self.alpha      = config.get('alpha', 0.5)
        self.beta       = config.get('beta', 0.5)
        self.dropout_p  = config.get('dropout_p', 0.3)
        self.eps        = 1e-12

        self.__init_weight()

    def __init_weight(self):
        X_sp = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()

        train_start = time()
        print(f"Fitting DCLAE (lambda={self.reg_lambda}, alpha={self.alpha}, beta={self.beta}, dropout={self.dropout_p}) on {self.device}...")

        # ── Stage 1: User-side Fractional IPW (CPU Sparse) ───────────────────────────
        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        user_weights = np.power(n_u + self.eps, -self.beta)
        D_U_inv = sparse.diags(user_weights)

        X_weighted = D_U_inv @ X_sp                 
        G_U = (X_sp.T @ X_weighted).toarray()       

        # ── Stage 2: Item-side Geometric Ensemble Normalization ──────────────────
        A_i = G_U.diagonal().copy()
        scale = np.power(A_i + self.eps, -self.alpha / 2.0)
        G_tilde = G_U * scale[:, None] * scale[None, :]

        # ── 🔥 Dropout Regularization ──
        p = min(self.dropout_p, 0.99)
        w_dropout = (p / (1.0 - p)) * A_i

        # ── Stage 3: Ridge Regression via Solve (GPU) ────────────────
        K = self.num_items
        G_torch = torch.tensor(G_tilde, dtype=torch.float32, device=self.device)
        
        # diagonal에 dropout 추가
        A_mat = G_torch + torch.diag(
            torch.tensor(w_dropout + self.reg_lambda, dtype=torch.float32, device=self.device)
        )

        try:
            # W = (G_tilde + lambda*I + Dropout)^{-1} @ G_tilde
            self.W = torch.linalg.solve(A_mat, G_torch)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular matrix, applying stronger regularization.")
            A_mat.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            self.W = torch.linalg.solve(A_mat, G_torch)

        self.W = self.W.cpu().numpy()

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

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
class EASE_DAN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE_DAN, self).__init__()
        self.config = config
        self.dataset = dataset
        from world import device
        self.device = device
        self.reg_p = config['reg_p']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        
        train_start = time()
        item_counts = np.array(X.sum(axis=0))
        user_counts = np.array(X.sum(axis=1))
        X_T = X.multiply(np.power(user_counts, -self.beta)).T
        G = X_T.dot(X).toarray()
        lmbda = self.reg_p + (self.config.get('drop_p', 0.0) / (1 - self.config.get('drop_p', 0.0) + 1e-10)) * item_counts
        G[np.diag_indices(X.shape[1])] += lmbda.reshape(-1)
        
        P = np.linalg.inv(G)
        B_DLAE = np.eye(X.shape[1]) - P / np.diag(P)
        item_power_term = np.power(item_counts, -(1 - self.alpha))
        W = B_DLAE * (1/item_power_term).reshape(-1, 1) * item_power_term
        W[np.diag_indices(X.shape[1])] = 0
        
        self.W_gpu = torch.FloatTensor(W).to(self.device)
        self.train_time = time() - train_start
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)
 
class IPS_LAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(IPS_LAE, self).__init__()
        self.dataset = dataset
        from world import device
        self.device = device
        
        # 하이퍼파라미터 설정
        self.reg_lambda = config.get('reg_lambda', 500.0)
        self.wbeta = config.get('wbeta', 0.4)
        self.wtype = config.get('wtype', 'logsigmoid')  # powerlaw | logsigmoid
        self.eps = 1e-12
        self.__init_weight()

    def _compute_inv_propensity(self, X):
        """아이템별 역 성향 점수(Inverse Propensity Score)를 계산합니다."""
        pop = np.array(X.sum(axis=0)).flatten()
        
        if self.wtype == 'powerlaw':
            # Power-law 기반 성향 점수 계산
            norm_pop = pop / (np.max(pop) + self.eps)
            p = np.power(norm_pop, self.wbeta)
        elif self.wtype == 'logsigmoid':
            # Log-sigmoid 기반 성향 점수 계산 (아이템 빈도 로그 스케일링)
            log_freqs = np.log(pop + 1)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / 2
            p = 1 / (1 + np.exp(-(alpha_logit + self.wbeta * log_freqs)))
        else:
            p = np.ones_like(pop)
            
        # 역 성향 점수 반환 (GPU 텐서)
        return torch.tensor(1 / (p + self.eps), dtype=torch.float32, device=self.device)

    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        
        train_start = time()
        
        # 1. Gram Matrix 계산 및 정규화 (CPU)
        print("  computing gram matrix and inverting...")
        G = (X.T @ X).toarray()
        G[np.diag_indices(X.shape[1])] += self.reg_lambda
        
        # 2. EASE 해법 (Closed-form solution)
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        W_np = -P / (diag_P + self.eps)
        np.fill_diagonal(W_np, 0)
        
        W_torch = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        
        # 3. IPS Weighting 적용 (GPU)
        # 아이템별로 p_j에 반비례하도록 가중치 행렬의 컬럼을 스케일링
        inv_p = self._compute_inv_propensity(X)
        self.W_gpu = W_torch * inv_p.view(1, -1)
        
        self.train_time = time() - train_start
        print(f"costing {self.train_time}s for training")
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        return self._get_batch_ratings(users, self.test_matrix, self.W_gpu)

    def getvalidUsersRating(self, users):
        return self._get_batch_ratings(users, self.valid_matrix, self.W_gpu)
