#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/wait.h>

// check if placing this queen attack another queen on board
bool CheckNewQueen(int row, int col, char **board, int m, int n);

// create board
int BoardCreation(char **board, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        *(board + i) = calloc(n, sizeof(char));
        if (*(board + i) == NULL)
        {
            fprintf(stderr, "ERROR: col calloc\n");
            fflush(stderr);
            for (int j = 0; j < i; j++)
            {
                free(*(board + j));
            }
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

// free memory for board and array for max queens for each i
void FreeMemories(char **board, int *maxQueens, int m, int n);

// use to wait for child
void BlockOnWaitpid(int pid);

// updat ethe array for max queens in pipe
void UpdateMaxQueens(int *pipefd, int *maxQueens, int i, int m)
{
    if (read(*(pipefd), maxQueens, sizeof(int) * (m + 1)) == -1)
    {
        fprintf(stderr, "read error\n");
        abort();
    }
    *(maxQueens + i) += 1;

    if (write(*(pipefd + 1), maxQueens, sizeof(int) * (m + 1)) == -1)
    {
        fprintf(stderr, "ERROR: write\n");
        abort();
    }
}

// explore the possibilities for new queen placement
// for current row
int Expand(int row, char **board, int m, int n, int *maxQueens, int queen, int *pipefd)
{
    // base case for recursion
    if (row >= m)
    {
        UpdateMaxQueens(pipefd, maxQueens, queen, m);
        if (queen == m)
        {

#ifndef QUIET
            printf("P%d: found a solution; notifying top-level parent\n", getpid());
            fflush(stdout);
#endif
        }
        return EXIT_SUCCESS;
    }

    int possibleMove = 0;

    int *possible = calloc(n, sizeof(int));

    // checking all possibilities for this row
    for (int col = 0; col < n; col++)
    {
        if (CheckNewQueen(row, col, board, m, n))
        {

#if DEBUG_MODE
            printf("P%d: %d %d %d\n\n", getpid(), row, col, children);
            fflush(stdout);

#endif

            *(possible + col) = 1;

            possibleMove++;
        }
    }

    // when the no possible move
    if (possibleMove == 0)
    {
        UpdateMaxQueens(pipefd, maxQueens, queen, m);
#ifndef QUIET
        printf("P%d: dead end at row #%d; notifying top-level parent\n", getpid(), row);
        fflush(stdout);
#endif
    }
    else
    {
        // singular and plural case
        if (possibleMove == 1)
        {
#ifndef QUIET
            printf("P%d: %d possible move at row #%d; creating %d child process...\n", getpid(), possibleMove, row, possibleMove);
            fflush(stdout);
#endif
        }
        else
        {
#ifndef QUIET
            printf("P%d: %d possible moves at row #%d; creating %d child processes...\n", getpid(), possibleMove, row, possibleMove);
            fflush(stdout);
#endif
        }
#ifndef NO_PARALLEL

        int children = 0;
#endif

        // creating child process
        for (int i = 0; i < n; i++)
        {
            if (*(possible + i) == 1)
            {
                // creating child for explorations
                pid_t pid = fork();

#if NO_PARALLEL
                BlockOnWaitpid(pid);
#endif

                if (pid == -1)
                {
                    // fork error
                    fprintf(stderr, "ERROR: fork\n");
                    fflush(stderr);
                    exit(-1);
                }

                if (pid == 0)
                {
                    // child process
                    // resetting col and number of possible row
                    // move to next row
                    *(*(board + row) + i) = 'Q';
                    Expand(row + 1, board, m, n, maxQueens, queen + 1, pipefd);
                    free(possible);
                    return EXIT_SUCCESS;
                }
                else if (pid > 0)
                {

#ifndef NO_PARALLEL
                    children++;
#endif
                }
            }
        }

        close(*(pipefd));
        close(*(pipefd + 1));

#ifndef NO_PARALLEL

        // wait for child
        while (children > 0)
        {
            BlockOnWaitpid(-1);
            children--;
        }
#endif
    }
    free(possible);

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    // command line arguments
    if (argc != 3 || atoi(*(argv + 1)) < 1 || atoi(*(argv + 2)) < 1)
    {
        fprintf(stderr, "ERROR: Invalid argument(s)\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    int m = atoi(*(argv + 1));
    int n = atoi(*(argv + 2));

    // constrains on input
    if (n < m)
    {
        int tmp = m;
        m = n;
        n = tmp;
    }

    if (m > 256)
    {
        fprintf(stderr, "ERROR: m > 256\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    int *pipefd = calloc(2, sizeof(int));

    int rc = pipe(pipefd);

    if (rc == -1)
    {
        fprintf(stderr, "ERROR: pipe\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    char **board = calloc(m, sizeof(char *));
    if (board == NULL)
    {
        fprintf(stderr, "ERROR: row calloc\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    int *maxQueens = calloc(m + 1, sizeof(int));

    if (maxQueens == NULL)
    {
        free(board);
        fprintf(stderr, "ERROR: maxQueens calloc\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    // attempt to create necessary board
    if (BoardCreation(board, m, n) == EXIT_FAILURE)
    {
        FreeMemories(board, maxQueens, m, n);
        return EXIT_FAILURE;
    }

    // write empty array to pipe for possible queens
    // with all entries with value 0
    write(*(pipefd + 1), maxQueens, sizeof(int) * (m + 1));

    printf("P%d: solving the Abridged (m,n)-Queens problem for %dx%d board\n", getpid(), m, n);
    fflush(stdout);

    int row = 0, queen = 0;
    int possibleMove = 0;

    int *possible = calloc(n, sizeof(int));

    if (possible == NULL)
    {
        free(board);
        free(maxQueens);
        fprintf(stderr, "ERROR: possible calloc\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    // if no parallel then no point to record number of child
    // since parent will just wait for each child
#ifndef NO_PARALLEL
    int children = 0;
#endif

    // explore the possible moves for current row
    for (int col = 0; col < n; col++)
    {
        if (CheckNewQueen(row, col, board, m, n))
        {
            *(possible + col) = 1;
            possibleMove++;
        }
    }

    // whe no more move
    if (possibleMove == 0)
    {
        UpdateMaxQueens(pipefd, maxQueens, queen, m);
    }
    else
    {
        // singular vs plural
        if (possibleMove == 1)
        {
            printf("P%d: %d possible move at row #%d; creating %d child process...\n", getpid(), possibleMove, row, possibleMove);
            fflush(stdout);
        }
        else
        {
            printf("P%d: %d possible moves at row #%d; creating %d child processes...\n", getpid(), possibleMove, row, possibleMove);
            fflush(stdout);
        }

        for (int i = 0; i < n; i++)
        {
            if (*(possible + i) == 1)
            {
                // creating childrens
                pid_t pid = fork();

#if NO_PARALLEL
                // if no parallel then parent will wait for each children
                // before breeding any new children
                BlockOnWaitpid(pid);
#endif

                if (pid == -1)
                {
                    fprintf(stderr, "ERROR: fork\n");
                    fflush(stderr);

                    exit(-1);
                }

                if (pid == 0)
                {

                    // child process
                    // resetting col and number of possible row
                    // move to next row
                    *(*(board + row) + i) = 'Q';
                    Expand(row + 1, board, m, n, maxQueens, queen + 1, pipefd);

                    FreeMemories(board, maxQueens, m, n);
                    close(*(pipefd));
                    close(*(pipefd + 1));
                    free(pipefd);
                    free(possible);

                    return EXIT_SUCCESS;
                }
                else if (pid > 0)
                {

#ifndef NO_PARALLEL
                    children++;
#endif
                }
            }
        }

#ifndef NO_PARALLEL
        while (children > 0)
        {
            BlockOnWaitpid(-1);
            children--;
        }
#endif
    }

    BlockOnWaitpid(-1);
    close(*(pipefd + 1));

    printf("P%d: search complete\n", getpid());
    fflush(stdout);

    read(*(pipefd), maxQueens, sizeof(int) * (m + 1));
    close(*(pipefd));

    for (int i = 1; i <= m; i++)
    {
        printf("P%d: number of %d-Queen end-states: %d\n", getpid(), i, *(maxQueens + i));
        fflush(stdout);
    }

    FreeMemories(board, maxQueens, m, n);
    free(pipefd);
    free(possible);

    return EXIT_SUCCESS;
}

void FreeMemories(char **board, int *maxQueens, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        free(*(board + i));
    }
    free(board);
    free(maxQueens);
}

void BlockOnWaitpid(int pid)
{
    int status;
    pid_t result = waitpid(pid, &status, 0);
    if (result > 0)
    {
        if (WIFSIGNALED(status))
        {
            int signalNumber = WTERMSIG(status);
            fprintf(stderr, "Child process %d terminated with signal %d\n", result, signalNumber);
            fflush(stderr);
            abort();
        }
        else if (WIFEXITED(status))
        {

#if DEBUG_MODE
            int exit_status = WEXITSTATUS(status);
            printf("PARENT: ...normally with exit status %d\n", exit_status);
            fflush(stdout);
#endif
        }
    }
}

// check if placing this new queen will face another
bool CheckNewQueen(int row, int col, char **board, int m, int n)
{
    // Check adjacent positions
    int adj_positions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int i = 0; i < 4; i++) {
        int new_row = row + adj_positions[i][0];
        int new_col = col + adj_positions[i][1];
        if (new_row >= 0 && new_row < m && new_col >= 0 && new_col < n) {
            if (*(*(board + new_row) + new_col) == 'Q') {
                return false;
            }
        }
    }
    // Check all directions (vertical, horizontal, and diagonals)
    int directions[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {1, 1}, {-1, 1}, {1, -1}
    };

    for (int d = 0; d < 8; d++) {
        int i = 1;
        while (true) {
            int new_row = row + i * directions[d][0];
            int new_col = col + i * directions[d][1];
            if (new_row < 0 || new_row >= m || new_col < 0 || new_col >= n) {
                break;
            }
            if (*(*(board + new_row) + new_col) == 'Q') {
                return false;
            }
            i++;
        }
    }

    return true;
}
