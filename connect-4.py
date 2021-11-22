from collections import Counter
import numpy as np

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4
DEPTH = 3


def valid_moves(free_column):
	# """Returns columns where a disc may be played"""
	# return [n for n in range(NUM_COLUMNS) if board[n, COLUMN_HEIGHT - 1] == 0]
	return np.argwhere(free_column < COLUMN_HEIGHT).T[0].tolist()


# def play(board, column, player):
#     """Updates `board` as `player` drops a disc in `column`"""
#     (index,) = next((i for i, v in np.ndenumerate(board[column]) if v == 0))
#     board[column, index] = player
def play(board, column, player, free_column):
	index = free_column[column]
	free_column[column] += 1
	board[column, index] = player


def take_back(board, column, free_column):
	# """Updates `board` removing top disc from `column`"""
	# (index,) = [i for i, v in np.ndenumerate(board[column]) if v != 0][-1]
	free_column[column] -= 1
	index = free_column[column]
	board[column, index] = 0


def four_in_a_row(board, player):
	"""Checks if `player` has a 4-piece line"""
	return (
			any(
				all(board[c, r] == player)
				for c in range(NUM_COLUMNS)
				for r in (list(range(n, n + FOUR)) for n in range(COLUMN_HEIGHT - FOUR + 1))
			)
			or any(
		all(board[c, r] == player)
		for r in range(COLUMN_HEIGHT)
		for c in (list(range(n, n + FOUR)) for n in range(NUM_COLUMNS - FOUR + 1))
	)
			or any(
		np.all(board[diag] == player)
		for diag in (
			(range(ro, ro + FOUR), range(co, co + FOUR))
			for ro in range(0, NUM_COLUMNS - FOUR + 1)
			for co in range(0, COLUMN_HEIGHT - FOUR + 1)
		)
	)
			or any(
		np.all(board[diag] == player)
		for diag in (
			(range(ro, ro + FOUR), range(co + FOUR - 1, co - 1, -1))
			for ro in range(0, NUM_COLUMNS - FOUR + 1)
			for co in range(0, COLUMN_HEIGHT - FOUR + 1)
		)
	)
	)


def _mc(board, player, free_column):
	p = -player
	while valid_moves(free_column):
		p = -p
		c = np.random.choice(valid_moves(free_column))
		play(board, c, p, free_column)
		if four_in_a_row(board, p):
			return p
	return 0


def montecarlo(board, player, free_column):
	montecarlo_samples = 100
	cnt = Counter(_mc(np.copy(board), player, free_column.copy()) for _ in range(montecarlo_samples))
	return (cnt[1] - cnt[-1]) / montecarlo_samples


def eval_board(board, player, free_column):
	if four_in_a_row(board, 1):
		# Alice won
		return 1
	elif four_in_a_row(board, -1):
		# Bob won
		return -1
	else:
		# Not terminal, let's simulate...
		return montecarlo(board, player, free_column)


def min_max(board, player, depth, alpha, beta):
	evaluation = eval_board(board, player, free_column)
	possible = valid_moves(free_column)
	if evaluation == 1 or evaluation == -1 or not possible or depth == DEPTH:
		return None, evaluation
	if player == -1:
		tree = np.full(NUM_COLUMNS, 2.0)
		for column in possible:
			play(board, column, player, free_column)
			_, val = min_max(board, -player, depth + 1, alpha, beta)
			take_back(board, column, free_column)
			alpha = max(alpha, val)
			tree[column] = val
			if val >= beta:
				break
		i = tree.argmin()
	else:
		tree = np.full(NUM_COLUMNS, -2.0)
		for column in possible:
			play(board, column, player, free_column)
			_, val = min_max(board, -player, depth + 1, alpha, beta)
			take_back(board, column, free_column)
			beta = min(beta, val)
			tree[column] = val
			if val <= alpha:
				break
		i = tree.argmax()
	return i, tree[i]


board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)
free_column = np.zeros(NUM_COLUMNS, dtype = int)
play(board, 5, 1, free_column)
play(board, 1, -1, free_column)
play(board, 4, 1, free_column)
print(board)

p = -1
i, _ = min_max(board, p, 1, -2, 2)
play(board, i, p, free_column)
print(board)

