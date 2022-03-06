#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List, Tuple
import snoop
from pprint import pprint
import os


# ## write_file
# 
# Writes a file to the `io` directory.
# 
# * **fname**: The file name as string.
# * **command**: The type of command issued as a string.
# 
# **returns**: None.

# In[2]:


def write_file(fname: str, command: str)-> None:
    with open(f"io/{fname}.txt", "w") as f: # not appending on purpose
        f.write(command)


# ## read_file
# 
# Read a file from the `io` directory and return the file contents.
# 
# * **fname**: The file name as string.
# 
# **returns**: Contents of the file.

# In[3]:


def read_file(fname: str)-> str:
    with open(f"io/{fname}.txt") as f:
        return f.read()


# ## write_logging_files
# 
# Takes in file names and checks if the files exist in the `io` directory. If files don't exist, files are created.
# 
# * **exploit_file_3x3**: String representing a filename.
# * **exploit_file_3x3**: String representing a filename.
# * **set_file_5x5**: String representing a filename.
# 
# **returns**: None

# In[4]:


def write_logging_files(exploit_file_3x3: str, exploit_file_5x5: str, set_file_5x5: str)-> None:
    files = [exploit_file_3x3, exploit_file_5x5, set_file_5x5]
    for f in files:
        if not os.path.isfile(f"io/{f}.txt"):
            write_file(f, "")


# ## get_latest_move
# 
# Method takes the diff of two matrices and identifies the additional move made in the latter matrix by its location. The location is represented as $(i, j)$, where $i$ is the row and $j$ is the column starting from index 0. In the tic-tac-toe game board representation, $0$ is empty, $1$ is $x$ and $2$ is $o$.
# 
# * **prev_state**: The previous state of the game board.
# * **current_state**: The current state of the game board.
# 
# **returns**: A tuple of ints.

# In[5]:


def get_latest_move(prev_state: List[List[int]], current_state: List[List[int]])-> Tuple[int, int]:            
    return [(i, j) for i in range(0, len(prev_state)) for j in range(0, len(current_state[0])) if prev_state[i][j] != current_state[i][j]][0]

board_old = [
    [0, 1, 0],
    [0, 0, 2],
    [0, 0, 0]]
board_new = [
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0]]
# 0 = empty, 1 = X, 2 = O
assert get_latest_move(board_old, board_new) == (0,1)


# ## horizontal_left_search
# 
# Searches horizontal left of the $(i,j)$ position.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[6]:


def horizontal_left_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i
        _j = j - n
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [1, 1, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
assert horizontal_left_search(i=2, j=1, matrix=board, num_neighbors=2, symbol=1) == False
assert horizontal_left_search(i=2, j=1, matrix=board, num_neighbors=1, symbol=1) == True


# ## horizontal_right_search
# 
# Search horizontal right of the $(i,j)$ position.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[7]:


def horizontal_right_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i
        _j = j + n
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board1 = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 1, 1, 1, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
board2 = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11,  1, 0, 1, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
assert horizontal_right_search(i=2, j=1, matrix=board1, num_neighbors=2, symbol=1) == True
assert horizontal_right_search(i=2, j=1, matrix=board2, num_neighbors=2, symbol=1) == False


# ## vertical_up_search
# 
# Search vertical up from the $(i,j)$ position.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[8]:


def vertical_up_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i - n
        _j = j
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
assert vertical_up_search(i=2, j=1, matrix=board, num_neighbors=2, symbol=1) == False


# ## vertical_down_search
# 
# Search vertical down from $(i,j)$ position.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[9]:


def vertical_down_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i + n
        _j = j
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board1 = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 1, 13, 14, 15],
    [16, 1, 18, 19, 20],
    [21, 1, 23, 24, 25]]

board2 = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 1, 13, 14, 15],
    [16, 0, 18, 19, 20],
    [21, 1, 23, 24, 25]]
assert vertical_down_search(i=2, j=1, matrix=board1, num_neighbors=2, symbol=1) == True
assert vertical_down_search(i=2, j=1, matrix=board2, num_neighbors=2, symbol=1) == False


# ## diagonal_left_search
# 
# Search diagonal left from the $(i,j)$ position (i.e., the first quadrant)
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[10]:


def diagonal_left_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i - n
        _j = j - n 
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  1,  8,  9, 10],
    [11, 12, 1, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
assert diagonal_left_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1) == True


# ## diagonal_right_search
# 
# Search diagonal right from $(i,j)$ position (i.e., the third quadrant)
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[11]:


def diagonal_right_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i + n
        _j = j + n 
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  12, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18,  1, 20],
    [21, 22, 23, 24,  1]]
assert diagonal_right_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1) == True


# ## antidiagonal_left_search
# 
# Search the antidiagonal left from $(i,j)$ position (i.e., 4th quadrant).
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[12]:


def antidiagonal_left_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i + n
        _j = j - n 
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 1, 18, 19, 20],
    [1, 22, 23, 24, 25]]
assert antidiagonal_left_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1) == True


# ## antidiagonal_right_search
# 
# Search the antidiagonal right from the $(i,j)$ position (i.e., second quadrant).
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[13]:


def antidiagonal_right_search(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors+1):
        _i = i - n
        _j = j + n 
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  1],
    [ 6,  7,  8,  1, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
assert antidiagonal_right_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1) == True


# ## check_move_made_inbetween_two_moves
# 
# Checks for moves made in between two moves as a failsafe to ensure that winning move is not made by linking two moves together, i.e., it checks from moves made in the form of x **X** x. Method checks in all possible directions and if any of those directional checks return a `True`, the method returns `True` and `False` otherwise.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[14]:


def check_move_made_inbetween_two_moves(i: int, j: int, matrix: List[List[int]], symbol: int)-> bool:
        horizontal = all([horizontal_left_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                         horizontal_right_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
        vertical = all([vertical_up_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                        vertical_down_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
        diagonal = all([diagonal_right_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                       diagonal_left_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
        antidiagonal = all([antidiagonal_right_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                        antidiagonal_left_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
        return any([horizontal, vertical, diagonal, antidiagonal])  

board3 = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 0, 0]]    
assert check_move_made_inbetween_two_moves(i=0, j=1, matrix=board3, symbol=1) == True
assert check_move_made_inbetween_two_moves(i=1, j=1, matrix=board3, symbol=1) == False


# ##  check_move_made_inbetween_three_moves
# 
# Checks for moves made in between three moves as a failsafe to ensure that winning move is not made by linking moves together, i.e., it checks from moves made in the form of x **X** x x or x x **X** x. Method checks in all possible directions and if any of those directional checks return a `True`, the method returns `True` and `False` otherwise.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool..

# In[15]:


def check_move_made_inbetween_three_moves(i: int, j: int, matrix: List[List[int]], symbol: int)-> bool:
    # scenario x X x x
    horizontal1 = all([horizontal_left_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                     horizontal_right_search(i, j, matrix, num_neighbors=2, symbol=symbol)])
    vertical1 = all([vertical_up_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                    vertical_down_search(i, j, matrix, num_neighbors=2, symbol=symbol)])
    diagonal1 = all([diagonal_right_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                   diagonal_left_search(i, j, matrix, num_neighbors=2, symbol=symbol)])
    antidiagonal1 = all([antidiagonal_right_search(i, j, matrix, num_neighbors=1, symbol=symbol),
                    antidiagonal_left_search(i, j, matrix, num_neighbors=2, symbol=symbol)])
    # scenario x x X x
    horizontal2 = all([horizontal_left_search(i, j, matrix, num_neighbors=2, symbol=symbol),
                     horizontal_right_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
    vertical2 = all([vertical_up_search(i, j, matrix, num_neighbors=2, symbol=symbol),
                    vertical_down_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
    diagonal2 = all([diagonal_right_search(i, j, matrix, num_neighbors=2, symbol=symbol),
                   diagonal_left_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
    antidiagonal2 = all([antidiagonal_right_search(i, j, matrix, num_neighbors=2, symbol=symbol),
                    antidiagonal_left_search(i, j, matrix, num_neighbors=1, symbol=symbol)])
    return any([horizontal1, vertical1, diagonal1, antidiagonal1, horizontal2, vertical2, diagonal2, antidiagonal2])  

board5_1 = [
    [ 0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0]]
board5_2 = [
    [ 0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0]]
board5_3 = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0]]
board5_4 = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0]]
assert check_move_made_inbetween_three_moves(i=1, j=2, matrix=board5_1, symbol=1) == True
assert check_move_made_inbetween_three_moves(i=1, j=3, matrix=board5_2, symbol=1) == True
assert check_move_made_inbetween_three_moves(i=2, j=2, matrix=board5_3, symbol=1) == True
assert check_move_made_inbetween_three_moves(i=3, j=2, matrix=board5_4, symbol=1) == False


# ## eval_player_move
# 
# Method evaluates a player's move from the given $(i,j)$ position in every direction and returns true if any of the directions meet the stated requirements for a play.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **num_neighbors**: Given a position on the board, the number of neighbors in any given direction (not including the current position).
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[16]:


def eval_player_move(i: int, j: int, matrix: List[List[int]], num_neighbors: int, symbol: int)-> bool:       
    return any([
        horizontal_left_search(i, j, matrix, num_neighbors, symbol),
        horizontal_right_search(i, j, matrix, num_neighbors, symbol),
        vertical_up_search(i, j, matrix, num_neighbors, symbol),
        vertical_down_search(i, j, matrix, num_neighbors, symbol),
        diagonal_right_search(i, j, matrix, num_neighbors, symbol),
        diagonal_left_search(i, j, matrix, num_neighbors, symbol),
        antidiagonal_right_search(i, j, matrix, num_neighbors, symbol),
        antidiagonal_left_search(i, j, matrix, num_neighbors, symbol)])


# ## is_first_move
# 
# Checks if this is the first made move by an $X$/attacker in the game.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **symbol**: symbol representing the board squares, with $0$ as empty, $1$ as $x$ and $2$ as $o$. 
# 
# **returns**: bool.

# In[17]:


def is_first_move(i: int, j: int, matrix: List[List[int]], symbol: int)-> bool:
    count = 0
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            if matrix[i][j] == symbol:
                count += 1
            if count >= 2:
                return False
    return True

board3_1 = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]] 
assert is_first_move(i=1, j=0, matrix=board3_1, symbol=1) == True
board3_2 = [
    [0, 0, 0],
    [1, 2, 0],
    [0, 0, 1]]
assert is_first_move(i=1, j=0, matrix=board3_2, symbol=1) == False


# ## eval_attacker_3x3
# 
# Series of if/then tests to evaluate moves by the attacker:
# 1. If the move is a first move made by attacker then the result is a port scan.
# 2. If the move is a move made with no adjacent $X$ moves nearby, then move is a NOP.
# 3. If move is a move made with in the form x x **X** or **X** x x or x **X** then it is a game winning move.
# 4. If move is made in the form of x **X**, i.e., chaining two $X$s, then move initiates an exploit with `use` and `set` commands.
# 5. Otherwise move is a NOP.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **exploit_file**: filename for exploit file. 
# 
# **returns**: str.

# In[18]:


def eval_attacker_3x3(i: int, j: int, matrix: List[List[int]], exploit_file: str)-> str:
    symbol=1
    if is_first_move(i, j, matrix, symbol=symbol):
        # save scanned ports to a list
        return "port scan"
    if not eval_player_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        return "NOP"
    if eval_player_move(i, j, matrix, num_neighbors=2, symbol=symbol) or check_move_made_inbetween_two_moves(i, j, matrix, symbol):
        if read_file(exploit_file) == "exploit initiated":
            return "run exploit -- game over, attacker wins!"
        return "exploit initiated and parameters set, run exploit -- game over, attacker wins!"
    if eval_player_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        if read_file(exploit_file) == "exploit initiated":
            return "NOP -- exploit already in progress"
        write_file(exploit_file, "exploit initiated")
        # retrieve command based on port from a list of ports and command from db of commands
        return "exploit initiated and parameters set"
    return "NOP"


# ## eval_defender_3x3
# 
# Defender side if/then evaluation of move:
# 1. If move is not adjacent to an $x$ or an $o$ then move is a NOP.
# 2. If move is three $O$s chained then move is a game winning move.
# 3. If move is blocking of two chained $X$s, then move is a block and it kills a process.
# 4. Otherwise move is a NOP.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **exploit_file**: filename for exploit file. 
# 
# **returns**: str.

# In[19]:


def eval_defender_3x3(i: int, j: int, matrix: List[List[int]], exploit_file: str)-> str:
    if not any([eval_player_move(i, j, matrix, num_neighbors=1, symbol=2), eval_player_move(i, j, matrix, num_neighbors=1, symbol=1)]):
        return "NOP"
    if eval_player_move(i, j, matrix, num_neighbors=2, symbol=2) or check_move_made_inbetween_two_moves(i, j, matrix, symbol=2):
        write_file(exploit_file, "")
        return "kill daemon -- defender wins!"
    if eval_player_move(i, j, matrix, num_neighbors=2, symbol=1) or check_move_made_inbetween_two_moves(i, j, matrix, symbol=1):
        write_file(exploit_file, "")
        return("defender blocks attacker -- kill process")
    return "NOP"
    
m = [
    [0, 0, 0],
    [1, 2, 1],
    [0, 0, 2]]
assert eval_defender_3x3(1, 1, m, "exploit_3x3.txt") == 'defender blocks attacker -- kill process'


# ## eval_defender_5x5
# 
# Defender side if/then evaluation of move:
# 1. If move is not adjacent to an $x$ or an $o$ then move is a NOP.
# 2. If move is four $O$s chained then move is a game winning move.
# 3. If move is blocking of three chained $X$s, then move is a block and it kills a process.
# 4. If move is blocking of two chained $X$s, then move is a block and it kills a process.
# 5. Otherwise move is a NOP.
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **exploit_file**: filename for exploit file. 
# * **set_file**: filename for set file. 
# 
# **returns**: str.

# In[20]:


def eval_defender_5x5(i: int, j: int, matrix: List[List[int]], exploit_file: str, set_file: str)-> str:
    if not any([eval_player_move(i, j, matrix, num_neighbors=1, symbol=2), eval_player_move(i, j, matrix, num_neighbors=1, symbol=1)]):
        return "NOP"
    if eval_player_move(i, j, matrix, num_neighbors=3, symbol=2) or check_move_made_inbetween_three_moves(i, j, matrix, symbol=2):
        write_file(exploit_file, "")
        write_file(set_file, "")
        return "kill daemon -- defender wins!"
    if eval_player_move(i, j, matrix, num_neighbors=3, symbol=1) or check_move_made_inbetween_three_moves(i, j, matrix, symbol=1):
        write_file(exploit_file, "")
        write_file(set_file, "")
        return "defender blocks move -- kill process"
    if eval_player_move(i, j, matrix, num_neighbors=2, symbol=1) or check_move_made_inbetween_two_moves(i, j, matrix, symbol=1):
        write_file(exploit_file, "")
        write_file(set_file, "")
        return "defender blocks attacker -- kill process"
    return "NOP"

    
m = [
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  2]]
assert eval_defender_5x5(i=4, j=4, matrix=m , exploit_file="exploit_5x5.txt", set_file="set_5x5.txt") == "NOP"


# ## eval_attacker_5x5
# 
# Series of if/then tests to evaluate moves by the attacker:
# 1. If the move is a first move made by attacker then the result is a port scan.
# 2. If the move is a move made with no adjacent $X$ moves nearby, then move is a NOP.
# 3. If move is a move made with in the form x x x **X** or **X** x x x or  x **X** x x or x x **X** x then it is a game winning move.
# 4. If move is made in the form of x x **X** or **X** x x or x **X** x, i.e., chaining three $X$s then $X$s, then move initiates a set
# 5. If move is made in the form of x **X**, i.e., chaining two $X$s, then move initiates an exploit via the `use` command.
# 6. Otherwise move is a NOP.
# 
# 
# * **i**: The $i_{th}$ row of the matrix.
# * **j**: The $j_{th}$ column of the matrix.
# * **matrix**: Current state of the matrix.
# * **exploit_file**: filename for exploit file. 
# * **set_file**: filename for set file. 
# 
# **returns**: str.

# In[21]:


def eval_attacker_5x5(i: int, j: int, matrix: List[List[int]], exploit_file: str, set_file: str)-> str:
    symbol=1
    if is_first_move(i, j, matrix, symbol=symbol):
        # save scanned ports to a list
        return "port scan"
    if not eval_player_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        return "NOP"
    if eval_player_move(i, j, matrix, num_neighbors=3, symbol=symbol) or check_move_made_inbetween_three_moves(i, j, matrix, symbol):
        if read_file(set_file) == "parameters set":
            return "run exploit -- game over, attacker wins!"
        return "set parameters, run exploit -- game over, attacker wins!"
    if eval_player_move(i, j, matrix, num_neighbors=2, symbol=symbol) or check_move_made_inbetween_two_moves(i, j, matrix, symbol):
        if read_file(exploit_file) == "exploit initiated":
            if read_file(set_file) == "parameter set":
                return "NOP -- parameters already set"
            write_file(set_file, "parameters set")
            return "parameters set"
        write_file(exploit_file, "exploit initiated")
        write_file(set_file, "parameters set")
        return "init exploit, parameters set"
    if eval_player_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        if read_file(exploit_file) == "exploit initiated":
            return "NOP -- exploit already in progress"
        write_file(exploit_file, "exploit initiated")
        return "exploit initiated"
    return "NOP"


# ## eval_move
# 
# Main method for the graph traversal program. Takes in a previous and current state and returns the action performed in the cyber realm.
# * **prev_state**: The previous state of the game board.
# * **current_state**: The current state of the game board.
# * **debug**: Bool flag to print debug statements.
#     
# **returns**: str.

# In[22]:


@snoop
def eval_move(prev_state: List[List[int]], current_state: List[List[int]], debug: bool=False):
    exploit_file_3x3 = "exploit_3x3"
    exploit_file_5x5 = "exploit_5x5"
    set_file_5x5 = "set_5x5"
    write_logging_files(exploit_file_3x3, exploit_file_5x5, set_file_5x5)
    move = get_latest_move(prev_state, current_state)
    i = move[0]
    j = move[1]
    if current_state[i][j] == 1: # attacker
        if debug: print("attacker move")
        if len(current_state) == 3:
            return eval_attacker_3x3(i, j, current_state, exploit_file_3x3)
        if len(current_state) == 5:
            return eval_attacker_5x5(i, j, current_state, exploit_file_5x5, set_file_5x5)
    elif current_state[i][j] == 2: # defender
        if debug: print("defender move")
        if len(current_state) == 3:
            return eval_defender_3x3(i, j, current_state, exploit_file_5x5)
        if len(current_state) == 5:
            return eval_defender_5x5(i, j, current_state, exploit_file_5x5, set_file_5x5)
    else:
        return "something has gone terribly wrong" # ruh roh

board_old = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0]]
board_new = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  2,  0,  0],
    [ 0,  0,  0,  0,  0]]
assert eval_move(board_old, board_new) == "NOP"


# In[23]:


board_old = [
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 2]]
board_new = [
    [0, 0, 1],
    [0, 1, 0],
    [2, 0, 2]]
assert eval_move(board_old, board_new) == "defender blocks attacker -- kill process"

