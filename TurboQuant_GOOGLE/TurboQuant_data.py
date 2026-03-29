#向量数据库的压缩存储
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import math

class TurboQuant:
    def __init__(self, d, b, use_prod=True):
        """
        d: 向量维度
        b: 比特数 per coordinate (对于 prod 模式，实际比特数会多 1 用于残差)
        use_prod: 如果 True，使用无偏内积的两阶段量化；否则只使用 MSE 量化。
        """
        self.d = d
        self.b = b
        self.use_prod = use_prod

        # 1. 生成随机旋转矩阵 (正交矩阵),
        self.Pi = self._random_orthogonal_matrix(d)

        # 2. 预计算标量量化码本 (基于高斯分布)
        # 高维下坐标分布近似 N(0, 1/d)，我们缩放为标准正态
        self.scale = 1.0 / np.sqrt(d)   # 坐标标准差
        self.codebook = self._optimal_scalar_codebook(b)

        # 3. 如果是 prod 模式，还需要 QJL 的随机矩阵
        if use_prod:
            self.S = np.random.randn(d, d)   # QJL 随机投影矩阵

    def _random_orthogonal_matrix(self, n):
        """生成随机正交矩阵 (通过 QR 分解)"""
        H = np.random.randn(n, n)
        Q, R = np.linalg.qr(H)
        return Q

    def _optimal_scalar_codebook(self, b):
        """生成最优标量码本 (基于标准正态分布)"""
        if b == 1:
            # 最优 1-bit 码本: ± sqrt(2/pi)
            c = math.sqrt(2.0 / math.pi)
            return np.array([-c, c])
        elif b == 2:
            # 2-bit 码本近似值 (来自论文)
            return np.array([-1.51, -0.453, 0.453, 1.51]) * self.scale
        elif b == 3:
            # 3-bit 需要求解 Lloyd-Max，这里使用近似值 (可自行数值优化)
            # 我们先用 8 个等距分位数近似，实际应用可预计算
            quantiles = np.linspace(0, 1, 2**b + 1)
            points = norm.ppf(quantiles[1:-1])
            # 调整使码本中心对称
            points = (points - np.mean(points)) / np.std(points) * self.scale
            return points
        elif b == 4:
            quantiles = np.linspace(0, 1, 2**b + 1)
            points = norm.ppf(quantiles[1:-1])
            points = (points - np.mean(points)) / np.std(points) * self.scale
            return points
        else:
            # 对于 b>4，使用均匀量化近似，也可以使用 Panter-Dite 公式生成
            # 这里简单用标准正态分位数
            quantiles = np.linspace(0, 1, 2**b + 1)
            points = norm.ppf(quantiles[1:-1])
            points = (points - np.mean(points)) / np.std(points) * self.scale
            return points

    def quantize_mse(self, x):
        """MSE 量化: 返回索引数组和残差范数 (如果需要)"""
        # 随机旋转
        y = self.Pi @ x
        # 对每个坐标独立量化
        idx = np.zeros(self.d, dtype=int)
        for j in range(self.d):
            # 找到最近码本
            diff = np.abs(y[j] - self.codebook)
            idx[j] = np.argmin(diff)
        # 重建
        y_hat = self.codebook[idx]
        # 反旋转得到重建向量
        x_hat = self.Pi.T @ y_hat
        # 残差
        r = x - x_hat
        r_norm = np.linalg.norm(r)
        return idx, r_norm, x_hat

    def dequantize_mse(self, idx):
        y_hat = self.codebook[idx]
        return self.Pi.T @ y_hat

    def quantize_prod(self, x):
        """两阶段量化，返回 (idx, qjl, gamma)"""
        # 第一阶段: b-1 比特 MSE 量化 (如果 b=1 则无法使用 prod，这里处理)
        if self.b == 1:
            # 当 b=1 时，只使用 QJL 单比特量化
            qjl = np.sign(self.S @ x)
            gamma = np.linalg.norm(x)
            idx = None
        else:
            # 先使用 b-1 比特量化
            idx, _, x_mse = self.quantize_mse(x)
            r = x - x_mse
            gamma = np.linalg.norm(r)
            if gamma > 0:
                r_unit = r / gamma
            else:
                r_unit = np.zeros(self.d)
            qjl = np.sign(self.S @ r_unit)
        return idx, qjl, gamma

    def dequantize_prod(self, idx, qjl, gamma):
        """重建向量"""
        if self.b == 1:
            # 只用 QJL 重建
            x_qjl = (np.sqrt(math.pi / 2) / self.d) * gamma * (self.S.T @ qjl)
            return x_qjl
        else:
            x_mse = self.dequantize_mse(idx)
            x_qjl = (np.sqrt(math.pi / 2) / self.d) * gamma * (self.S.T @ qjl)
            return x_mse + x_qjl

    def compress(self, vectors):
        """压缩一批向量，返回压缩表示列表"""
        compressed = []
        for x in vectors:
            if self.use_prod:
                compressed.append(self.quantize_prod(x))
            else:
                idx, _, _ = self.quantize_mse(x)
                compressed.append(idx)
        return compressed

    def decompress(self, compressed):
        """解压一批向量"""
        reconstructed = []
        for item in compressed:
            if self.use_prod:
                idx, qjl, gamma = item
                reconstructed.append(self.dequantize_prod(idx, qjl, gamma))
            else:
                idx = item
                reconstructed.append(self.dequantize_mse(idx))
        return np.array(reconstructed)

    def inner_product(self, query, compressed):
        """计算查询向量与压缩数据库向量的内积估计（无偏）"""
        # 仅当 use_prod=True 时有效
        if not self.use_prod:
            raise ValueError("Inner product estimation requires use_prod=True")
        # 预先将查询向量旋转
        query_rot = self.Pi @ query
        results = []
        for item in compressed:
            idx, qjl, gamma = item
            # 第一阶段内积
            if self.b == 1:
                inner1 = 0.0
            else:
                y_hat = self.codebook[idx]
                inner1 = np.dot(query_rot, y_hat)
            # 第二阶段内积 (QJL)
            # 根据论文，QJL 反量化后的向量为 (sqrt(pi/2)/d) * gamma * S^T * qjl
            # 内积为 (sqrt(pi/2)/d) * gamma * (qjl^T * (S @ query))
            Sz = self.S @ query
            inner2 = (np.sqrt(math.pi / 2) / self.d) * gamma * np.dot(qjl, Sz)
            results.append(inner1 + inner2)
        return np.array(results)



if __name__ == "__main__":
    # 参数设置
    d = 128  # 向量维度
    b = 3  # 每坐标比特数（对于 prod 模式，实际使用 b-1 比特 + 1 比特残差）
    n = 10000  # 数据库向量数量

    # 生成随机数据库向量 (假设已归一化，或存储范数)
    np.random.seed(42)
    db_vectors = np.random.randn(n, d)
    db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)  # 单位化

    # 初始化 TurboQuant (prod 模式)
    tq = TurboQuant(d, b, use_prod=True)

    # 压缩数据库
    compressed_db = tq.compress(db_vectors)

    # 存储大小评估
    if b == 1:
        # 每个向量存储: qjl (d 比特) + gamma (一个浮点数)
        bits_per_vector = d + 32  # gamma 用 32 位浮点
    else:
        # 每个向量存储: idx (d * (b-1) 比特) + qjl (d 比特) + gamma (浮点)
        bits_per_vector = d * (b - 1) + d + 32
    print(f"压缩后每向量约 {bits_per_vector} 比特，原始每向量 {d * 32} 比特，压缩比 {d * 32 / bits_per_vector:.2f}x")

    # 查询向量
    query = np.random.randn(d)
    query = query / np.linalg.norm(query)

    # 计算与所有压缩向量的内积估计
    inner_estimates = tq.inner_product(query, compressed_db)

    # 真实内积
    true_inner = db_vectors @ query

    # 误差分析
    mse_inner = np.mean((true_inner - inner_estimates) ** 2)
    print(f"内积估计 MSE: {mse_inner:.6f}")

    # 召回率评估 (Top-10)
    topk = 10
    true_topk = np.argsort(-true_inner)[:topk]
    est_topk = np.argsort(-inner_estimates)[:topk]
    recall = len(set(true_topk) & set(est_topk)) / topk
    print(f"Recall@{topk}: {recall:.4f}")