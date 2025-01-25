from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each processor generates a random integer
    np.random.seed(rank)  # Seed for reproducibility
    local_value = np.random.randint(0, 100)

    # Print the initial value
    print(f"Processor {rank} generated value: {local_value}")

    # Autonomous sorting through swapping
    sorted = False
    while not sorted:
        sorted = True
        # Compare with the next processor
        if rank < size - 1:
            # Send and receive values
            comm.send(local_value, dest=rank + 1)
            received_value = comm.recv(source=rank + 1)

            # Swap if needed
            if local_value > received_value:
                local_value, received_value = received_value, local_value
                sorted = False

        # Compare with the previous processor
        if rank > 0:
            # Send and receive values
            comm.send(local_value, dest=rank - 1)
            received_value = comm.recv(source=rank - 1)

            # Swap if needed
            if local_value < received_value:
                local_value, received_value = received_value, local_value
                sorted = False

    # Final sorted value
    print(f"Processor {rank} final sorted value: {local_value}")

if __name__ == "__main__":
    main()