from mpi4py import MPI

def compute_partial_sum(start, end):
    """Compute the sum of 1/n^2 from start to end (inclusive)."""
    partial_sum = 0.0
    for n in range(start, end + 1):
        partial_sum += 1.0 / (n * n)
    return partial_sum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define the total number of terms to compute
    total_terms = 10000000  # Adjust this value for more precision

    # Determine the range of terms for each process
    terms_per_process = total_terms // size
    start = rank * terms_per_process + 1
    end = start + terms_per_process - 1

    # Handle the last process which may have a different range
    if rank == size - 1:
        end = total_terms

    # Compute the partial sum for this process
    partial_sum = compute_partial_sum(start, end)

    # Reduce all partial sums to the root process
    total_sum = comm.reduce(partial_sum, op=MPI.SUM, root=0)

    # Only the root process will compute and print the final result
    if rank == 0:
        pi_squared = 6 * total_sum
        print(f"Computed value of pi square: {pi_squared}")

if __name__ == "__main__":
    main()