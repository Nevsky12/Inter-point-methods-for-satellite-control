import matplotlib.pyplot as plt
import numpy as np
import proxsuite
import time

def compute_lqr_matrices_optimized(A, B, P, Q, R, x0, N):
    """
    Computes H and c matrices for the LQR problem with horizon N, optimized for matrix powers.

    Parameters:
    - A  (ndarray): State transition matrix (r x r)
    - B  (ndarray): Input            matrix (r x m)
    - P  (ndarray): Terminal cost    matrix (r x r)
    - Q  (ndarray): State    cost    matrix (r x r)
    - R  (float  ): Control  cost    scalar
    - x0 (ndarray): Initial  state   vector (r x 1)
    - N  (int    ): Horizon length

    Returns:
    - H (ndarray): Quadratic cost matrix (N*m x N*m)
    - c (ndarray): Linear    cost vector (N*m x 1  )
    """
    r, m = B.shape

    powers = [B]
    for i in range(1, N):
        powers.append(A @ powers[-1])  # [B, A B, A^2 B, ..., A^{N-1} B]

    x_k = [x0]
    for k in range(1, N + 1):
        x_k.append(A @ x_k[-1])  # [x0, A x0, A^2 x0, ..., A^N x0]

    # Compute phi_N = [A^{N-1} B, A^{N-2} B, ..., B]
    phiN = np.hstack(powers[::-1])
    H = phiN.T @ P @ phiN  # Terminal cost contribution
    c = x_k[N].T @ P @ phiN  # Terminal cost linear term

    # Stage costs
    for k in range(1, N):
        # phi_k = [A^{k-1} B, A^{k-2} B, ..., B]
        phi_k = np.hstack(powers[:k][::-1])
        H[:k * m, :k * m] += phi_k.T @ Q @ phi_k  # Stage cost quadratic term
        c[:k * m] += x_k[k].T @ Q @ phi_k  # Stage cost linear term

    H += np.eye(N * m) * R  # Control cost contribution
    c = c.flatten()
    return H, c



# load a qp object using qp problem dimensions
T = 50.0
N = 1000
h = T / N

# System matrices
A = np.array([[1, h], [-h, 1]])
B = np.array([[0], [h]])

# Cost weights
P = np.array([[2, 0], [0, 1]])
Q = P.copy()
R = 6.0

# Initial state
x0 = np.array([15.0, 5.0])
# generate a random QP

H, c = compute_lqr_matrices_optimized(A, B, P, Q, R, x0, N)

H, g, A_1, b, C, l, u = H, c, np.zeros((0, 0)), np.zeros(0), np.eye(N), -np.ones(N), np.ones(N)
# initialize the model of the problem to solve
qp = proxsuite.proxqp.dense.QP(N, 0, N)
qp.init(H, g, A_1, b, C, l, u)
# solve without warm start
start_time = time.time()
qp.solve()
end_time = time.time()

elapsed_time = end_time - start_time
print('Elapsed time: ', elapsed_time)

info = qp.results.info

# print(dir(info))
print("количество итераций: ", info.iter)
print("минимизируемая функция: ", info.objValue)

u_opt = qp.results.x[:N]

# time axis
t = np.linspace(0, T, N + 1)
x_traj = np.zeros((2, N + 1))
x_traj[:, 0] = x0
for k in range(N):
    u = u_opt[k]
    x_traj[:, k + 1] = A @ x_traj[:, k] + B.flatten() * u
# plot
plt.figure(figsize=(12,6))
plt.plot(t[:-1], u_opt, label='control')
plt.plot(t, x_traj[0, :], '--', label='state-1')
plt.plot(t, x_traj[1, :], '-.', label='state-2')
plt.xlabel('t (seconds)')
plt.ylabel('Amplitude')
plt.title('Optimal control and system state response')
plt.legend()
plt.grid(True)
plt.show()
