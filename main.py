import cv2
import numpy as np
from mss import mss
import keyboard
import time

sct = mss()
monitor = sct.monitors[1]

tile = cv2.imread('./samples/tile.png', cv2.IMREAD_GRAYSCALE)
cleared = cv2.imread('./samples/cleared.png', cv2.IMREAD_GRAYSCALE)
directions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1)
]
number_templates = {}
for n in range(1, 9):
    img = cv2.imread(f'./samples/{n}.png', cv2.IMREAD_GRAYSCALE)
    if img is not None:
        number_templates[n] = img

def capture_screen():
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def find_board(gray):
    res = cv2.matchTemplate(gray, tile, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.8)
    pts = sorted(zip(*loc[::-1]))
    filtered = []
    for x, y in pts:
        if not filtered or abs(x - filtered[-1][0]) > 5 or abs(y - filtered[-1][1]) > 5:
            filtered.append((x, y))
    xs = np.array([x for x, y in filtered])
    ys = np.array([y for x, y in filtered])
    dx = np.median(np.diff(np.sort(np.unique(xs))))
    dy = np.median(np.diff(np.sort(np.unique(ys))))
    cols = len(np.unique(xs))
    rows = len(np.unique(ys))
    return xs, ys, dx, dy, rows, cols

def classify_cells(gray_board, xs, ys, dx, dy):
    sorted_xs = np.sort(np.unique(xs))
    sorted_ys = np.sort(np.unique(ys))
    board = np.full((len(sorted_ys), len(sorted_xs)), -1, dtype=int)

    for r, y in enumerate(sorted_ys):
        for c, x in enumerate(sorted_xs):
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

            if tile_val > cleared_val and tile_val > 0.7:
                board[r, c] = -1
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
                board[r, c] = best_num
            elif cleared_val > 0.5:
                board[r, c] = 0
            else:
                board[r, c] = -1

    return board

print("Press space to start...")
keyboard.wait("space")

img = capture_screen()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
xs, ys, dx, dy, rows, cols = find_board(gray)
print(f"Detected grid: {rows}x{cols}")

def add_flags(board):
     for i in range(len(board)):
        for j in range(len(board[i])):
            if(board[i][j]!=0 and board[i][j]!=-1):
                undiscovered = set()          
                for dr, dc in directions:
                    newI = i + dr
                    newJ = j+dc
                    if 0 <= newI < rows and 0 <= newJ < cols and board[newI][newJ]==-1:
                        undiscovered.add((newI,newJ))
                if len(undiscovered) == board[i][j]:
                    for newI,newJ in undiscovered:
                        board[newI][newJ] = 9



while True:
    img = capture_screen()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board = classify_cells(gray, xs, ys, dx, dy)

    print("\n" + "=" * 40)
    add_flags(board)
    for row in board:
        print(" ".join(f"{x:2}" for x in row))
    print("=" * 40)
    time.sleep(1)
