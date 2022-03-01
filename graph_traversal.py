#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List, Tuple
import snoop
from pprint import pprint
import os


# In[2]:


def write_file(fname, command):
    with open(f"io/{fname}.txt", "w") as f: # not appending on purpose
        f.write(command)
        
def read_file(fname):
    with open(f"io/{fname}.txt") as f:
        return f.read()


# In[3]:


board_old = [
    [0, 1, 0],
    [0, 0, 2],
    [0, 0, 0]]
board_new = [
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0]]
# 0 = empty, 1 = X, 2 = O


# In[4]:


def get_latest_move(prev_state: List[List[int]], current_state: List[List[int]])-> Tuple[int, int]:            
    return [(i, j) for i in range(0, len(prev_state)) for j in range(0, len(current_state[0])) if prev_state[i][j] != current_state[i][j]][0]


# In[5]:


assert get_latest_move(board_old, board_new) == (0,1)


# In[6]:


def adjacency_check(i, j, matrix, num_neighbors=2):
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    start_pos_i = i + (i - min_i) if i - num_neighbors < min_i else i - num_neighbors
    start_pos_j = j + (j - min_j) if j - num_neighbors < min_j else j - num_neighbors
    end_pos_i = i + (max_i - i) if i + num_neighbors > max_i else i + num_neighbors
    end_pos_j = j + (max_j - j) if j + num_neighbors > max_j else j + num_neighbors  

    locs = []
    for _i in range(start_pos_i, end_pos_i + 1):
        for _j in range(start_pos_j, end_pos_j + 1): 
            locs.append((_i, _j))
    horizontal = [loc for loc in locs if loc[0]==i]
    vertical = [loc for loc in locs if loc[1]==j]
    diagonal = [loc for loc in locs if loc[0]==loc[1]]
    antidiagonal = [loc for loc in locs if loc[0]+loc[1]==(i+j)]
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "diagonal": diagonal,
        "antidiagonal": antidiagonal
    }


# In[7]:


board_adj = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]

ba = adjacency_check(i=3, j=3, matrix=board_adj)
assert ba["horizontal"] == [(3, 1), (3, 2), (3, 3), (3, 4)]
assert ba["vertical"] == [(1, 3), (2, 3), (3, 3), (4, 3)]
assert ba["diagonal"] == [(1, 1), (2, 2), (3, 3), (4, 4)]
assert ba["antidiagonal"] == [(2, 4), (3, 3), (4, 2)]
pprint(ba)


# In[8]:


# num_neighbors 0 == the initial x square
# num_neighbors 1 == the initial x square plus 1 neighbor
def horizontal_left_search(i, j, matrix, num_neighbors, symbol):
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


# In[9]:


def horizontal_right_search(i, j, matrix, num_neighbors, symbol):
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


# In[10]:


def vertical_up_search(i, j, matrix, num_neighbors, symbol):
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


# In[11]:


def vertical_down_search(i, j, matrix, num_neighbors, symbol):
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


# In[12]:


def diagonal_left_search(i, j, matrix, num_neighbors, symbol):
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


# In[13]:


def diagonal_right_search(i, j, matrix, num_neighbors, symbol):
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
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
diagonal_right_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1)


# In[14]:


def antidiagonal_left_search(i, j, matrix, num_neighbors, symbol):
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(1, num_neighbors):
        _i = i - n
        _j = j - n 
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors+1:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
antidiagonal_left_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1)


# In[15]:


def antidiagonal_right_search(i, j, matrix, num_neighbors, symbol):
    min_i = 0 
    min_j = 0
    max_i = len(matrix) - 1
    max_j = len(matrix[0]) - 1
    
    xs = []
    for n in range(0, num_neighbors+1):
        _i = i + n
        _j = j + n 
        if _i >= min_i and _j >= min_j and _i <= max_i and _j <= max_j:
            xs.append(matrix[_i][_j])
    if len(xs) != num_neighbors+1:
        return False
    return all(item == symbol for item in xs)

board = [
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]
antidiagonal_right_search(i=2, j=2, matrix=board, num_neighbors=2, symbol=1)


# In[16]:


def check_move_made_inbetween_two_moves(i, j, matrix, symbol):
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


# In[17]:


def check_move_made_inbetween_three_moves(i, j, matrix, symbol):
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


# In[18]:


def eval_attacker_move(i, j, matrix, num_neighbors, symbol):       
    return any([
        horizontal_left_search(i, j, matrix, num_neighbors, symbol),
        horizontal_right_search(i, j, matrix, num_neighbors, symbol),
        vertical_up_search(i, j, matrix, num_neighbors, symbol),
        vertical_down_search(i, j, matrix, num_neighbors, symbol),
        diagonal_right_search(i, j, matrix, num_neighbors, symbol),
        diagonal_left_search(i, j, matrix, num_neighbors, symbol),
        antidiagonal_right_search(i, j, matrix, num_neighbors, symbol),
        antidiagonal_left_search(i, j, matrix, num_neighbors, symbol)])


# In[19]:


def is_first_move(i, j, matrix, symbol):
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


# In[25]:


@snoop
def eval_attacker_3x3(i, j, matrix):
    exploit_file = "exploit_3x3"
    symbol=1
    if is_first_move(i, j, matrix, symbol=symbol):
        # save scanned ports to a list
        return "port scan"
    if not eval_attacker_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        return "NOP"
    if eval_attacker_move(i, j, matrix, num_neighbors=2, symbol=symbol):
        try:
            if read_file(exploit_file) == "exploit initiated":
                return "run exploit -- game over, attacker wins!"
        except FileNotFoundError:
            return "use exploit and set parameters, run exploit -- game over, attacker wins!"
    if check_move_made_inbetween_two_moves(i, j, matrix, symbol):
        try:
            if read_file(exploit_file) == "exploit initiated":
                return "run exploit -- game over, attacker wins!"
        except FileNotFoundError:
            return "use exploit and set parameters, run exploit -- game over, attacker wins!"
    if eval_attacker_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        try:
            if read_file(exploit_file) == "exploit initiated":
                return "NOP -- exploit already in progress"
        except FileNotFoundError:
            write_file(exploit_file, "exploit initiated")
            # retrieve command based on port from a list of ports and command from db of commands
            return "use exploit and set parameters"
    return "NOP"


# In[36]:


@snoop
def eval_attacker_5x5(i, j, matrix):
    exploit_file = "exploit_5x5"
    set_file = "set_5x5"
    symbol=1
    if is_first_move(i, j, matrix, symbol=symbol):
        # save scanned ports to a list
        return "port scan"
    if not eval_attacker_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        return "NOP"
    if eval_attacker_move(i, j, matrix, num_neighbors=3, symbol=symbol):
        try:
            if read_file(set_file) == "parameters set":
                return "run exploit -- game over, attacker wins!"
        except FileNotFoundError:
            return "set parameters, run exploit -- game over, attacker wins!"
    if check_move_made_inbetween_three_moves(i, j, matrix, symbol):
        try:
            if read_file(set_file) == "parameters set":
                return "run exploit -- game over, attacker wins!"
        except FileNotFoundError:
            return "set parameters, run exploit -- game over, attacker wins!"
    if eval_attacker_move(i, j, matrix, num_neighbors=2, symbol=symbol):
        try:
            if read_file(exploit_file) == "exploit initiated":
                try:
                    if read_file(set_file) == "parameter set":
                        return "NOP -- parameters already set"
                except FileNotFoundError:
                    write_file(set_file, "parameters set")
                    return "parameters set"
        except FileNotFoundError:
            write_file(exploit_file, "exploit initiated")
            write_file(set_file, "parameters set")
            return "init exploit, parameters set"
    if check_move_made_inbetween_two_moves(i, j, matrix, symbol):
        try:
            if read_file(exploit_file) == "exploit initiated":
                try:
                    if read_file(set_file) == "parameter set":
                        return "NOP -- parameters already set"
                except FileNotFoundError:
                    write_file(set_file, "parameters set")
                    return "parameters set"
        except FileNotFoundError:
            write_file(exploit_file, "exploit initiated")
            write_file(set_file, "parameters set")
            return "init exploit, parameters set"
    if eval_attacker_move(i, j, matrix, num_neighbors=1, symbol=symbol):
        try:
            if read_file(exploit_file) == "exploit initiated":
                return "NOP -- exploit already in progress"
        except FileNotFoundError:
            write_file(exploit_file, "exploit initiated")
            return "use command to commence exploit"
    return "NOP"


# In[46]:


@snoop
def eval_move(prev_state, current_state):
    move = get_latest_move(prev_state, current_state)
    i = move[0]
    j = move[1]
    if current_state[i][j] == 1: # attacker
        if len(current_state) == 3:
            return eval_attacker_3x3(i, j, current_state)
        if len(current_state) == 5:
            return eval_attacker_5x5(i, j, current_state)
    elif current_state[i][j] == 2: # defender
        return "defender moves go here..."
    else:
        return "something has gone terribly wrong" # ruh roh

board_old = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  0]]
board_new = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  1,  1,  1,  0]]
eval_move(board_old, board_new)


# In[42]:


@snoop
def eval_move(prev_state, current_state):
    move = get_latest_move(prev_state, current_state)
    i = move[0]
    j = move[1]
    if current_state[i][j] == 1: # attacker
        if len(current_state) == 3:
            return eval_attacker_3x3(i, j, current_state)
        if len(current_state) == 5:
            return eval_attacker_5x5(i, j, current_state)
    elif current_state[i][j] == 2: # defender
        return "defender moves go here..."
    else:
        return "something has gone terribly wrong" # ruh roh

board_old = [
    [1, 0, 1],
    [2, 2, 0],
    [1, 2, 0]]
board_new = [
    [1, 1, 1],
    [2, 2, 0],
    [1, 0, 0]]
eval_move(board_old, board_new)


# In[ ]:





# In[24]:


import itertools
set(itertools.permutations([1, 0, 1]))


# In[ ]:




