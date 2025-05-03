import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm import solve_box_qp
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
import time as time
import matplotlib.pyplot as plt


# --- create problem data
n_x = 1000
m = 1
n_batch = 1
n_samples = 2 * n_x
n_sims = 1
tol = [1e-8]#, 1e-3, 1e-5]


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


# --- set models:
QP_fp = []
QP_optnet = []

for i in range(len(tol)):
    QP_fp.append(SolveBoxQP(control=box_qp_control(eps_rel=tol[i], eps_abs=tol[i])))


models = {"ADMM FP 1": QP_fp[0]}#, "ADMM FP 3": QP_fp[1], "ADMM FP 5": QP_fp[2],


model_names = list(models.keys())

# --- storage:
forward_times = np.zeros((n_sims, len(model_names)))
forward_times = pd.DataFrame(forward_times, columns=model_names)
backward_times = np.zeros((n_sims, len(model_names)))
backward_times = pd.DataFrame(backward_times, columns=model_names)
total_times = np.zeros((n_sims, len(model_names)))
total_times = pd.DataFrame(total_times, columns=model_names)



# --- main loop:

for i in range(n_sims):
    H, c = compute_lqr_matrices_optimized(A, B, P, Q, R, x0, N)
    e = np.ones(n_x)
    A = np.zeros((2, 2))
    b = np.zeros(2)
    dict = {}
    start_time = time.time()
    u_opt = solve_box_qp(H, c, A=None, b=None, lb=-1, ub=1, control=dict)
    end_time = time.time()
   # print(u_opt)

    elapsed_time = end_time - start_time
    print('Elapsed time: ', elapsed_time)
    np.save("ADMM_runtime", elapsed_time)
    print(u_opt['Loss'])
    np.save("ADMM_iter", u_opt['iter'])
    np.save("ADMM_loss", u_opt['Loss'])


    t = np.linspace(0, T, N + 1)
    x_traj = np.zeros((2, N + 1))
    x_traj[:, 0] = x0
    for k in range(N):
        u = u_opt['x'][k]
        x_traj[:, k + 1] = A @ x_traj[:, k] + B.flatten() * u
    # plot
    plt.figure(figsize=(12,6))
    plt.plot(t[:-1], u_opt['x'], label='control')
    plt.plot(t, x_traj[0, :], '--', label='state-1')
    plt.plot(t, x_traj[1, :], '-.', label='state-2')
    plt.xlabel('t (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Optimal control and system state response')
    plt.legend()
    plt.grid(True)
    plt.show()

