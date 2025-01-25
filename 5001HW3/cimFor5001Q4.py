import numpy as np
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt


class QAIA:
    r"""
    The base class of QAIA.

    This class contains the basic and common functions of all the algorithms.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N x 1)`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1):
        """Construct a QAIA algorithm."""
        self.J = J
        if h is not None and len(h.shape) < 2:
            h = h[:, np.newaxis]
        self.h = h
        self.x = x
        # The number of spins
        self.N = self.J.shape[0]
        self.n_iter = n_iter
        self.batch_size = batch_size

    def initialize(self):
        """Randomly initialize spin values."""
        self.x = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)

    def calc_cut(self, x=None):
        r"""
        Calculate cut value.

        Args:
            x (numpy.array): The spin value with shape :math:`(N x batch_size)`.
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if x is None:
            sign = np.sign(self.x)
        else:
            sign = np.sign(x)

        return 0.25 * np.sum(self.J.dot(sign) * sign, axis=0) - 0.25 * self.J.sum()

    def calc_energy(self, x=None):
        r"""
        Calculate energy.

        Args:
            x (numpy.array): The spin value with shape :math:`(N x batch_size)`.
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if x is None:
            sign = np.sign(self.x)
        else:
            sign = np.sign(x)

        if self.h is None:
            return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0)
        return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0, keepdims=True) - self.h.T.dot(sign)


class OverflowException(Exception):
    r"""
    Custom exception class for handling overflow errors in numerical calculations.

    Args:
        message: Exception message string, defaults to "Overflow error".
    """

    def __init__(self, message="Overflow error"):
        self.message = message
        super().__init__(self.message)


class CIM(QAIA):
    r"""
    Simulated Coherent Ising Machine.

    Reference: `Annealing by simulating the coherent Ising
    machine <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-7-10288&id=408024>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        momentum (float): momentum factor. Default: ``0.9``.
        sigma (float): The standard deviation of noise. Default: ``0.03``.
        pt (float): Pump parameter. Default: ``6.5``.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=12000,
        batch_size=10,
        dt=0.01,
        momentum=0,
        sigma=1.75,
        a=0.5,
        y=1.6,
    ):
        """Construct CIM algorithm."""
        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        self.dt = dt
        self.momentum = momentum
        self.sigma = sigma
        self.a = a
        self.y = y
        self.avrCutSize = []  # 新增，保存每次迭代的cut值
        self.initialize()

    def initialize(self):
        """Initialize spin."""
        # Initialization of spin value
        if self.x is None:
            self.x = np.zeros((self.N, self.batch_size))
            # self.x = np.random.rand(self.N, self.batch_size) - 0.5
            self.avrCutSize.append(0)  # 保存当前cut值
        # gradient
        self.dx = np.zeros_like(self.x)
        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")



    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        for _ in range(self.n_iter):
            # 计算 dx 的值
            linear_term = self.a * self.x
            cubic_term = -self.x ** 3
            interaction_term = self.J.dot(self.x)
            
            # 生成白噪音
            noise = np.random.normal(0, self.sigma, self.x.shape)  
            # 均值为0，标准差为sigma

            # 更新 dx
            newdc = linear_term + cubic_term + interaction_term + noise*self.y
            
            # 更新自旋值
            self.dx = self.dx * self.momentum + newdc * self.dt
            # self.dx = (self.a * self.x - self.x ** 3 + self.J.dot(self.x) + np.random.normal(0, self.sigma, self.x.shape)*self.y) * self.dt
            # ind = (np.abs(self.x + self.dx) < 1.0).astype(np.int64 or np.int32)
            # self.x += self.dx * ind
            self.x += self.dx
            cutValue = self.calc_cut()
            self.avrCutSize.append(np.mean(cutValue))  # 保存当前cut value



def load_gset(filename):
    with open(filename, 'r') as f:
        # Skip the first line or parse it to get number of nodes, edges
        lines = f.readlines()[1:]
        edges = [list(map(int, line.split())) for line in lines]
    return edges


def create_adjacency_matrix(edges, size):
    J = np.zeros((size, size))
    for (i, j, weight) in edges:
        J[i-1, j-1] = J[j-1, i-1] = -weight
    return J


if __name__ == "__main__":
    filename = r"C:\Users\user\Desktop\G1.txt"
    edges = load_gset(filename)
    size = max(max(i, j) for (i, j, _) in edges)
    J = create_adjacency_matrix(edges, size)
    # print(J)
    ntime = time.time()

    # 使用CIM算法，设置耦合矩阵、样本次数、迭代次数、迭代步长、噪声标准差
    s = CIM(J)
    s.update()

    # 绘图代码
    plt.figure(figsize=(10, 6))
    plt.plot(range(s.n_iter+1), s.avrCutSize, label=f'average cut-size versus t')
        
    plt.title(f'Average cut-size versus t over 10 samples')
    plt.xlabel('Time (iterations)')
    plt.ylabel('Average cut-size')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend(loc='lower right')  # Set a specific location for the legend
    plt.show()

    stime = time.time()

    print(f"CIM dynamical evolution complete, \trun time {stime - ntime}")