import random
from multiprocessing import Process, Manager

# Define the network as a dictionary of lists (adjacency list)
network = {
    'A': ['C', 'F'],
    'B': ['D', 'E'],
    'C': ['A', 'D', 'G'],
    'D': ['B', 'C', 'F', 'G'],
    'E': ['B', 'G'],
    'F': ['A', 'D', 'G', 'H'],
    'G': ['C', 'D', 'E', 'F', 'H'],
    'H': ['F', 'G']
}

# Initialize particles on each point
initial_particles = {
    'A': 7,
    'B': 9,
    'C': 12,
    'D': 15,
    'E': 16,
    'F': 17,
    'G': 22,
    'H': 24
}

def random_walk(point, particles, new_particles, lock):
    for _ in range(particles[point]):
        neighbor = random.choice(network[point])
        with lock:
            new_particles[neighbor] += 1  # 安全更新邻居的粒子数量

def simulate_random_walk(time_steps, initial_particles):
    manager = Manager()
    new_particles = manager.dict({key: 0 for key in initial_particles})  # 共享的新粒子字典
    lock = manager.Lock()  # 用于同步

    for t in range(time_steps):
        processes = []
        for point in initial_particles.keys():
            p = Process(target=random_walk, args=(point, initial_particles, new_particles, lock))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()  # 等待所有进程完成

        # 更新粒子分布
        initial_particles = dict(new_particles)  # 更新粒子分布
        new_particles = manager.dict({key: 0 for key in initial_particles})  # 重置新粒子字典

    return initial_particles

if __name__ == '__main__':
    final_distribution = simulate_random_walk(1000, initial_particles)
    print(final_distribution)