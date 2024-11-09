# Code Samples Overview

This repository contains a collection of my code samples, across various domains including machine learning, parallel processing, and concurrent programming.

## Folder Structure

### 1. **DigitClassification**
Contains Python code for classifying handwritten digits. Implements various machine learning algorithms to classify digits, with a focus on differentiating between the digits "1" and "5." Methods include linear regression, logistic regression (using gradient descent and stochastic gradient descent), and polynomial transformations to improve accuracy. Plots and error calculations for training and test datasets are also included.

### 2. **ParallelC**
Contains C code for solving a variation of the n-Queens problem using concurrent programming techniques. The solution leverages multi-processing, with `fork()`, `waitpid()`, and `pipe()` functions to create a synchronized parallel solution. Each process explores different paths in the solution tree, with intermediate and final states communicated via inter-process communication (IPC). Demonstrates the application of process management, synchronization, and dynamic memory allocation in C.

---

Each folder has a dedicated README with additional details on compilation, execution instructions, and etc.

--- 