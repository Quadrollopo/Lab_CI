# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 42
STEADY_STATE = 1000


class Tsp:

	def __init__(self, num_cities: int, seed: Any = None) -> None:
		if seed is None:
			seed = num_cities
		self._num_cities = num_cities
		self._graph = nx.DiGraph()
		np.random.seed(seed)
		for c in range(num_cities):
			self._graph.add_node(c, pos = (np.random.random(), np.random.random()))

	def distance(self, n1, n2) -> int:
		pos1 = self._graph.nodes[n1]['pos']
		pos2 = self._graph.nodes[n2]['pos']
		return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0]) ** 2 +
														 (pos1[1] - pos2[1]) ** 2))

	def evaluate_solution(self, solution: np.array) -> float:
		total_cost = 0
		tmp = solution.tolist() + [solution[0]]
		for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
			total_cost += self.distance(n1, n2)
		return total_cost

	def plot(self, path: np.array = None) -> None:
		if path is not None:
			self._graph.remove_edges_from(list(self._graph.edges))
			tmp = path.tolist() + [path[0]]
			for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
				self._graph.add_edge(n1, n2)
		plt.figure(figsize = (12, 5))
		nx.draw(self._graph,
				pos = nx.get_node_attributes(self._graph, 'pos'),
				with_labels = True,
				node_color = 'pink')
		if path is not None:
			plt.title(f"Current path: {self.evaluate_solution(path):,}")
		plt.show()

	def compute_distances(self):
		distances = []
		for i in range(self._num_cities):
			for j in range(i + 1, self._num_cities):
				distances.append((f'{i}-{j}', self.distance(i, j)))
		return distances

	@property
	def graph(self) -> nx.digraph:
		return self._graph


def tweak(solution: np.array, *, pm: float = .1) -> np.array:
	new_solution = solution.copy()
	p = None
	while p is None or p < pm:
		i1 = np.random.randint(0, solution.shape[0])
		i2 = np.random.randint(0, solution.shape[0])
		temp = new_solution[i1]
		new_solution[i1] = new_solution[i2]
		new_solution[i2] = temp
		p = np.random.random()
	return new_solution


def remove_node_distance(node, distances: list):
	new_distances = []
	for k, v in distances:
		nodes = k.split('-')
		if nodes[0] != node and nodes[1] != node:
			new_distances.append((k, v))
	return new_distances


def main():
	problem = Tsp(NUM_CITIES)

	solution = np.array(range(NUM_CITIES))
	pop_number = 32
	num_parents = 2
	offspring = np.empty(pop_number, dtype = tuple)
	# solution_cost = np.empty(pop_number, dtype = int)
	np.random.shuffle(solution)
	problem.plot(solution)

	for i in range(pop_number):
		son = tweak(solution, pm = .1)
		offspring[i] = (problem.evaluate_solution(son), son)
		# solution_cost.append( (problem.evaluate_solution(offspring[i]), offspring[i]))
	offspring = sorted(offspring, key = lambda a: a[0])
	parents = offspring[:num_parents]

	# history = [(0, solution_cost)]
	steady_state = 0
	step = 0
	partition = pop_number // num_parents
	pm = 0.2
	while steady_state < STEADY_STATE:
		step += 1
		steady_state += 1
		offspring[-1] = parents[0]
		for i in range(pop_number-1):
			son = tweak(parents[i // partition][1], pm = pm)
			offspring[i] = (problem.evaluate_solution(son), son)
		offspring = sorted(offspring, key = lambda a: a[0])
		if offspring[0][0] < parents[0][0]:
			steady_state = 0
		parents = offspring[:num_parents]
	winner = parents[0]
	problem.plot(winner[1])
	print(winner[0])


if __name__ == '__main__':
	logging.basicConfig(format = '[%(asctime)s] %(levelname)s: %(message)s', datefmt = '%H:%M:%S')
	logging.getLogger().setLevel(level = logging.INFO)
	main()
