import matplotlib.pyplot as plt
import numpy as np

admm_it = np.load("ADMM_iter.npy")
admm_runtime = np.load("ADMM_runtime.npy")
admm_loss = np.load("ADMM_loss.npy", allow_pickle=True)

print(admm_loss)

intP_it = np.load("intPoint_iterations.npy")
intP_runtime = np.load("intPoint_runtime.npy")
intP_loss = np.load("intPoint_loss.npy")

plt.figure()
plt.plot(np.arange(admm_it+1), admm_loss)
plt.plot(np.arange(intP_it), intP_loss)
plt.xlabel("iterations")
plt.ylabel("objective value")
plt.legend(['ADMM', "Interior point"])
#plt.plot(np.linspace(0, admm_runtime, admm_it), admm_loss)
plt.grid()

plt.figure()
plt.plot(np.linspace(0, admm_runtime, admm_it+1), admm_loss)
plt.plot(np.linspace(0, intP_runtime, intP_it), intP_loss)
plt.xlabel("runtime")
plt.ylabel("objective value")
plt.legend(['ADMM', "Interior point"])
#plt.plot(np.linspace(0, admm_runtime, admm_it), admm_loss)
plt.grid()
plt.show()
