import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

admm_it = np.load("ADMM_iter.npy")
admm_runtime = np.load("ADMM_runtime.npy")
admm_loss = np.load("ADMM_loss.npy", allow_pickle=True)

intP_it = np.load("intPoint_iterations.npy")
intP_runtime = np.load("intPoint_runtime.npy")
intP_loss = np.load("intPoint_loss.npy")

proxQP_it = np.load("iter_proxQP.npy")
proxQP_runtime = np.load("elapsed_time_proxQP.npy")

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


times_ADMM = np.load("ADMM_times.npy")
iters_ADMM = np.load("ADMM_iters.npy")

times_intP = np.load("times_intP.npy")
iters_intP = np.load("iters_intP.npy")

model_names = ['ADMM', 'proxQP', 'interior point']
total_times = [times_ADMM.mean(), proxQP_runtime, times_intP.mean()]
total_iters = [admm_it, proxQP_it, intP_it]
total_times_df = pd.DataFrame({'models': model_names, 'time': total_times})
total_iters_df = pd.DataFrame({'models': model_names, 'time': total_iters})

# Создаем bar plot
plt.figure(figsize=(8, 6))  # задаем размер фигуры
bars = plt.bar(total_times_df['models'], total_times_df['time'], color=['blue', 'green', 'red'])

# Добавляем подписи значений над столбцами
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

# Добавляем заголовок и подписи осей
plt.title('Comparison of Algorithm Runtimes, 30 simulations')
plt.xlabel('Optimization Models')
plt.ylabel('Time (seconds)')

# Сохраняем график
plt.tight_layout()  # чтобы подписи не обрезались
plt.savefig("runtime_comparison.png")


plt.figure(figsize=(8, 6))  # задаем размер фигуры
bars = plt.bar(total_iters_df['models'], total_iters_df['time'], color=['blue', 'green', 'red'])

# Добавляем подписи значений над столбцами
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

# Добавляем заголовок и подписи осей
plt.title('Comparison of Algorithm iterations')
plt.xlabel('Optimization Models')
plt.ylabel('Iterations')

# Сохраняем график
plt.tight_layout()  # чтобы подписи не обрезались
plt.savefig("iterations_comparison.png")
plt.show()