# Abridged (m, n)-Queens Problem Solver

This program solves a variation of the n-Queens problem using a multi-process approach in C. Given an `m x n` chessboard, it attempts to determine:
- The maximum number of non-attacking queens that can be placed.
- The number of distinct solutions that exist.
- The number of dead ends encountered during the search.

The program uses `fork()`, `waitpid()`, and `pipe()` to implement a parallelized brute-force solution, creating a process tree to explore possible board configurations.

## Compilation Modes

The program supports several compilation modes:

1. **Parallel Mode**: Default mode with full parallelization.
   ```bash
   gcc -Wall -Werror -o parallel.out main.c
   ```

2. **No Parallel Mode**: Runs without parallel execution, waiting for each child process to complete before starting the next. Useful for testing and deterministic output.
   ```bash
   gcc -Wall -Werror -D NO_PARALLEL -o no-parallel.out main.c
   ```

3. **Quiet Parallel Mode**: Only displays the first two lines and final summary, suppressing intermediate output for larger boards.
   ```bash
   gcc -Wall -Werror -D QUIET -o quiet-parallel.out main.c
   ```

4. **Quiet No Parallel Mode**: Combines quiet and non-parallel modes.
   ```bash
   gcc -Wall -Werror -D QUIET -D NO_PARALLEL -o quiet-no-parallel.out main.c
   ```

## Usage

Run the program with the board dimensions `m` and `n` as command-line arguments:

```bash
./parallel.out <m> <n>
```

### Example

For a `3x3` board:
```bash
./parallel.out 3 3
```

### Output Format

The program displays output for each process, including:
- Possible moves for each row and number of child processes created.
- Notifications for end-states (solutions or dead ends).
- A final summary with counts of distinct end-states by queen placement.

### Error Handling

If invalid arguments are provided:
```bash
ERROR: Invalid argument(s)
USAGE: hw2.out <m> <n>
```

If an error occurs during process creation, the program will terminate using `abort()` to propagate the error back to the top-level process.

## Implementation Details

- **Process Tree**: A new child process is forked for each valid queen placement, creating a tree structure where each leaf node represents an end-state.
- **IPC**: A single pipe is used to communicate results from child processes back to the top-level parent.
- **Memory Management**: The board is dynamically allocated with `calloc()` for each row and freed after usage, ensuring no memory leaks.

## Notes

- **Quiet Mode** is recommended for larger boards to reduce output clutter.
- **No Parallel Mode** can be used for debugging and testing on smaller boards.
