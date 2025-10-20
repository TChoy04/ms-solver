import cv2
import numpy as np
from mss import mss
import keyboard
import time
import pyautogui
import random

sct = mss()
monewItor = sct.monitors[1]
pyautogui.PAUSE = 0.001
tile = cv2.imread('./samples/tile.png', cv2.IMREAD_GRAYSCALE)
cleared = cv2.imread('./samples/cleared.png', cv2.IMREAD_GRAYSCALE)
flag = cv2.imread('./samples/flag.png', cv2.IMREAD_GRAYSCALE)
win = cv2.imread('./samples/win.png', cv2.IMREAD_GRAYSCALE)
loss = cv2.imread('./samples/loss.png', cv2.IMREAD_GRAYSCALE)
directions = [
    (-1, 0), (1, 0),
    (0, -1), (0, 1),
    (-1, -1), (-1, 1),
    (1, -1), (1, 1)
]

number_templates = {}
for n in range(1, 9):
    img = cv2.imread(f'./samples/{n}.png', cv2.IMREAD_GRAYSCALE)
    if img is not None:
        number_templates[n] = img


def capture_screen():
    img = np.array(sct.grab(monewItor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def find_board(gray):
    res = cv2.matchTemplate(gray, tile, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.8)
    pts = sorted(zip(*loc[::-1]))
    filtered = []
    for x, y in pts:
        if not filtered or abs(x - filtered[-1][0]) > 5 or abs(y - filtered[-1][1]) > 5:
            filtered.append((x, y))
    xs = np.array(sorted(list(set(x for x, y in filtered))))
    ys = np.array(sorted(list(set(y for x, y in filtered))))
    dx = int(np.median(np.diff(xs)))
    dy = int(np.median(np.diff(ys)))
    return xs, ys, dx, dy, len(ys), len(xs)


def cell_center(i, j, xs, ys, dx, dy):
    return int(xs[j] + dx / 2), int(ys[i] + dy / 2)


def classify_cells(gray_board, xs, ys, dx, dy):
    rows, cols = len(ys), len(xs)
    board = np.full((rows, cols), -1, dtype=int)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            x1, y1 = int(x), int(y)
            x2 = int(min(x + dx, gray_board.shape[1]))
            y2 = int(min(y + dy, gray_board.shape[0]))
            cell = gray_board[y1:y2, x1:x2]
            if cell.size == 0:
                continue

            res_tile = cv2.matchTemplate(cell, tile, cv2.TM_CCOEFF_NORMED)
            res_cleared = cv2.matchTemplate(cell, cleared, cv2.TM_CCOEFF_NORMED)
            tile_val = res_tile.max() if res_tile is not None else 0
            cleared_val = res_cleared.max() if res_cleared is not None else 0
            res_flag = cv2.matchTemplate(cell, flag, cv2.TM_CCOEFF_NORMED)
            flag_val = res_flag.max() if res_flag is not None else 0
            if flag_val > tile_val and flag_val > cleared_val and flag_val > 0.7:
                board[i, j] = 9 
                continue
            if tile_val > cleared_val and tile_val > 0.7:
                board[i, j] = -1
                continue
            if tile_val > cleared_val and tile_val > 0.7:
                board[i, j] = -1
                continue

            best_val, best_num = 0, None
            for n, tmpl in number_templates.items():
                if cell.shape[0] < tmpl.shape[0] or cell.shape[1] < tmpl.shape[1]:
                    continue
                res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
                val = res.max()
                if val > best_val:
                    best_val, best_num = val, n

            if best_val > 0.65:
                board[i, j] = best_num
            elif cleared_val > 0.5:
                board[i, j] = 0
            else:
                board[i, j] = -1
    return board


print("Press space to start...")
keyboard.wait("space")

img = capture_screen()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
xs, ys, dx, dy, rows, cols = find_board(gray)

def add_flags(board):
    for i in range(rows):
        for j in range(cols):
            if board[i][j] > 0 and board[i][j] != 9:
                unknowns = set()
                flag_count = 0
                for dr, dc in directions:
                    ni, nj = i + dr, j + dc
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if board[ni][nj] == -1:
                            unknowns.add((ni, nj))
                        elif board[ni][nj] == 9:
                            flag_count += 1
                remaining = board[i][j] - flag_count
                if remaining > 0 and len(unknowns) == remaining:
                    for ni, nj in unknowns:
                        if board[ni][nj] == -1: 
                            board[ni][nj] = 9
                            x, y = cell_center(ni, nj, xs, ys, dx, dy)
                            pyautogui.moveTo(x, y)
                            pyautogui.rightClick()
                            # time.sleep(0.02)



def safe_clicks(board):
    n, m = len(board), len(board[0])
    dirs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(n):
        for j in range(m):
            v=board[i][j]
            if v<=0 or v>8:continue
            hidden=[]
            flagcount=0
            for dx,dy in dirs:
                x,y=i+dx,j+dy
                if 0<=x<n and 0<=y<m:
                    if board[x][y]==9:flagcount+=1
                    elif board[x][y]==-1:hidden.append((x,y))
            if flagcount==v and hidden:
                for x,y in hidden:
                    if board[x][y]!=9:
                        board[x][y]=0
                        cx,cy=cell_center(x,y,xs,ys,dx,dy)
                        pyautogui.moveTo(cx,cy)
                        pyautogui.click()
                        # print(f"Clicked safe cell ({x},{y})")
                        # time.sleep(0.02)

def get_probabilities(board):
    n, m = len(board), len(board[0])
    prob_map = np.full((n, m), np.nan)
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(n):
        for j in range(m):
            v = board[i][j]
            if v <= 0 or v > 8:
                continue
            hidden = []
            flagcount = 0
            for dx, dy in dirs:
                x, y = i+dx, j+dy
                if 0 <= x < n and 0 <= y < m:
                    if board[x][y] == 9:
                        flagcount += 1
                    elif board[x][y] == -1:
                        hidden.append((x, y))
            remaining = v - flagcount
            if remaining > 0 and hidden:
                p = remaining / len(hidden)
                for (x, y) in hidden:
                    if np.isnan(prob_map[x, y]) or p > prob_map[x, y]:
                        prob_map[x, y] = p
    return prob_map

def guess_move(board, xs, ys, dx, dy):
    prob_map = get_probabilities(board)
    candidates = [(i, j, prob_map[i, j]) for i in range(len(board)) for j in range(len(board[0])) if board[i][j] == -1 and not np.isnan(prob_map[i, j])]
    if not candidates:
        candidates = [(i, j, 1.0) for i in range(len(board)) for j in range(len(board[0])) if board[i][j] == -1]
    if not candidates:
        return False  
    i, j, p = min(candidates, key=lambda x: x[2]) 
    cx, cy = cell_center(i, j, xs, ys, dx, dy)
    pyautogui.moveTo(cx, cy)
    pyautogui.click()
    print(f"Guessed ({i},{j}) with estimated mine prob {p:.2f}")
    return True

rand_i = random.randint(0, rows - 1)
rand_j = random.randint(0, cols - 1)
x, y = cell_center(rand_i, rand_j, xs, ys, dx, dy)
pyautogui.moveTo(x, y)
pyautogui.click()
# time.sleep(0.5)
def restart(gray_board):
    res_loss = cv2.matchTemplate(gray_board, loss, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res_loss)
    x, y = max_loc
    cx, cy = x + loss.shape[1] // 2, y + loss.shape[0] // 2
    pyautogui.moveTo(cx, cy)
    pyautogui.click()
    time.sleep(0.5)
def detect_win_loss(gray_board):
    res_win = cv2.matchTemplate(gray_board, win, cv2.TM_CCOEFF_NORMED)
    res_loss = cv2.matchTemplate(gray_board, loss, cv2.TM_CCOEFF_NORMED)
    win_val = res_win.max() if res_win is not None else 0
    loss_val = res_loss.max() if res_loss is not None else 0
    if win_val > 0.9:
        return "win"
    if loss_val > 0.9:
        return "loss"
    return None
while True:
    img = capture_screen()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if detect_win_loss(gray)=="win":
        break
    if detect_win_loss(gray) == "loss":
        restart(gray)
        time.sleep(1)
        rand_i = random.randint(0, rows - 1)
        rand_j = random.randint(0, cols - 1)
        x, y = cell_center(rand_i, rand_j, xs, ys, dx, dy)
        pyautogui.moveTo(x, y)
        pyautogui.click()
        time.sleep(0.3)
        continue
    board = classify_cells(gray, xs, ys, dx, dy)
    prev_board = board.copy()
    add_flags(board)
    safe_clicks(board)
    if np.array_equal(prev_board, board):
        guess_move(board, xs, ys, dx, dy)
    # for row in board:
    #     print(" ".join(str(x).rjust(2) for x in row))
    # print("br")
    # time.sleep(0.4)
