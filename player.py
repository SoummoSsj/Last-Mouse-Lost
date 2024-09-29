from random import randint as rand
import random 
import board as Board 
import collections as col


class Player:
	def __init__(self, board):
		self.board = board

	def move(self):
		raise NotImplementedError


class RandomPlayer(Player):
	def move(self):
		r = rand(0, 5)
		while self.board.row_empty(r):
			r = rand(0, 5)
		return (r, rand(1, self.board.spot_avail(r)) + 1)

	def __str__(self):
		return 'Random Player'


class HumanPlayer(Player):
	def move(self):
		try:
			pr = int(input('Row: '))
			pa = int(input('Amount: '))
		except ValueError:
			pr = -1
			pa = -1
		while pr > 5 or self.board.row_empty(pr) or pa <= 0 or pr < 0:
			print('Invalid move')
			try:
				pr = int(input('Row: '))
				pa = int(input('Amount: '))
			except ValueError:
				pr = -1
				pa = -1
		return (pr, pa)

	def __str__(self):
		return 'Human Player'


class SmartPlayer(Player):
	def __init__(self, board, pn):
		Player.__init__(self, board)
		self.pn = pn

	def _rmove(self):
		r = rand(0, 5)
		while self.board.row_empty(r):
			r = rand(0, 5)
		return (r, rand(1, self.board.spot_avail(r)))

	def move(self):
		cur_b = col.Counter(self.board.num_row())
		if cur_b in [{0: 2, 1: 4}, {0: 2, 1: 3, 2: 1}, {0: 2, 1: 3, 3: 1}, {0: 2, 1: 3, 4: 1}, {0: 2, 1: 3, 5: 1}, {0: 2, 1: 3, 6: 1}, {1: 6}, {1: 5, 2: 1}, {1: 5, 3: 1}, {1: 5, 4: 1}, {1: 5, 5: 1}, {1: 5, 6: 1}, {0: 4, 1: 2}, {0: 4, 1: 1, 2: 1}, {0: 4, 1: 1, 3: 1}, {0: 4, 1: 1, 4: 1}, {0: 4, 1: 1, 5: 1}, {0: 4, 1: 1, 6: 1}]:
			m = 0
			r = 0
			for i in range(len(self.board)):
				if self.board.spot_avail(i) > m:
					m = self.board.spot_avail(i)
					r = i
			return (r, m)
		elif cur_b in [{0: 1, 1: 4, 2: 1}, {0: 1, 1: 4, 3: 1}, {0: 1, 1: 4, 4: 1}, {0: 1, 1: 4, 5: 1}, {0: 1, 1: 4, 6: 1}, {0: 3, 1: 2, 2: 1}, {0: 3, 1: 2, 3: 1}, {0: 3, 1: 2, 4: 1}, {0: 3, 1: 2, 5: 1}, {0: 3, 1: 2, 6: 1}, {0: 5, 2: 1}, {0: 5, 3: 1}, {0: 5, 4: 1}, {0: 5, 5: 1}, {0: 5, 6: 1}]:
			m = 0
			r = 0
			for i in range(len(self.board)):
				if self.board.spot_avail(i) > m:
					m = self.board.spot_avail(i)
					r = i
			return (r, m - 1)
		else:
			if self.board.b[0] != self.board.b[5]:
				return self.board.diff(0, 5)
			if self.board.b[1] != self.board.b[4]:
				return self.board.diff(1, 4)
			if self.board.b[2] != self.board.b[3]:
				return self.board.diff(2, 3)
			else:
				return self._rmove()

	def __str__(self):
		return 'Smart Player'
	

class FuzzyPlayer(Player):
    def __init__(self, board):
        super().__init__(board)

    def fuzzy_logic_move(self):
        possible_moves = self.generate_possible_moves()
        best_move = None
        best_score = float('-inf')

        for move in possible_moves:
            score = self.evaluate_move(move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def generate_possible_moves(self):
        possible_moves = []
        for r in range(len(self.board.b)):
            if not self.board.row_empty(r):
                for a in range(1, self.board.spot_avail(r) + 1):
                    possible_moves.append((r, a))
        return possible_moves

    def evaluate_move(self, move):
        r, a = move
        score = 0
        # Simple evaluation: prefer moves that maximize opponent's difficulty
        for i in range(len(self.board.b)):
            if not self.board.row_empty(i):
                score -= self.board.spot_avail(i)
        score += self.board.spot_avail(r)  # Favor the current move
        return score
    
    def move(self):
        return self.fuzzy_logic_move()
    
    def __str__(self):
        return 'Fuzzy Player'


class GeneticAlgorithmPlayer(Player):
    def __init__(self, board, population_size=20, generations=100):
        super().__init__(board)
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [self.random_move(self.board) for _ in range(5)]  # Assume a sequence of 5 moves per individual
            population.append(individual)
        return population

    def random_move(self, board):
        """Generates a random valid move for the given board state."""
        valid_moves = self.generate_possible_moves(board)
        if valid_moves:
            return random.choice(valid_moves)
        else:
            # Fallback in case no valid moves are available
            return (0, 1)  # Assuming (0, 1) as a default move, this should be adapted as per game rules

    def generate_possible_moves(self, board):
        """Generate all possible valid moves for the current board state."""
        possible_moves = []
        for r in range(len(board.b)):
            if not board.row_empty(r):
                for a in range(1, board.spot_avail(r) + 1):
                    possible_moves.append((r, a))
        return possible_moves

    def evolve_population(self):
        """Evolves the population over a set number of generations."""
        for generation in range(self.generations):
            # Evaluate all individuals in the population
            population_fitness = [(self.evaluate_individual(individual), individual) for individual in self.population]
            # Sort individuals based on fitness (higher is better)
            population_fitness.sort(reverse=True, key=lambda x: x[0])
            # Select the top half of the population to survive
            self.population = [individual for _, individual in population_fitness[:self.population_size // 2]]

            # Create new population through crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            self.population = new_population

    def crossover(self, parent1, parent2):
        """Performs crossover between two parents to produce two children."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        """Randomly mutates an individual's move with a small probability."""
        mutation_rate = 0.1
        if random.random() < mutation_rate:
            index = random.randint(0, len(individual) - 1)
            individual[index] = self.random_move(self.board)
        return individual

    def evaluate_individual(self, individual):
        """Evaluates the fitness of an individual based on their moves for the game."""
        # Copy the board to simulate the individual's moves
        simulated_board = self.copy_board(self.board)
        score = 0

        # Simulate the moves of the individual
        for move in individual:
            r, a = move
            # Check if the move is valid
            if 0 <= r < len(simulated_board.b) and 1 <= a <= simulated_board.spot_avail(r):
                simulated_board.update_b(r, a)
                if simulated_board.g_o():
                    # If the game is over, this is a losing move
                    score -= 100  # Heavy penalty for a losing move
                    break
                else:
                    # Favor moves that reduce opponent's options
                    score -= len(self.generate_possible_moves(simulated_board))
            else:
                # Invalid move, heavily penalize
                score -= 100

        # Add bonus for leaving opponent in a state with limited moves
        if not simulated_board.g_o():
            score += 50 if len(self.generate_possible_moves(simulated_board)) == 1 else 0

        return score

    def copy_board(self, board):
        """Creates a deep copy of the board to simulate game states."""
        # Assuming a method to clone the board object, might require deep copy depending on the implementation
        return board.dupe()
    
    def move(self):
        """Choose the best move from the evolved population."""
        self.evolve_population()
        best_individual = max(self.population, key=lambda ind: self.evaluate_individual(ind))
        return best_individual[0]

    def __str__(self):
        return 'Genetic Algorithm Player'
             
class AStarPlayer(Player):
    def a_star_move(self):
        open_set = set()
        open_set.add((0, None, self.board))
        closed_set = set()
        while open_set:
            current_node = min(open_set, key=lambda x: x[0])
            open_set.remove(current_node)
            _, move, current_board = current_node

            if current_board.g_o():
                return move

            closed_set.add(current_board)

            for move in self.generate_possible_moves():
                new_board = self.board.dupe()
                new_board.update_b(move[0], move[1])
                if new_board in closed_set:
                    continue
                open_set.add((self.heuristic(new_board), move, new_board))

        return self.generate_possible_moves()[0]

    def heuristic(self, board):
        return sum([self.board.spot_avail(r) for r in range(len(board.b))])

    def generate_possible_moves(self):
        possible_moves = []
        for r in range(len(self.board.b)):
            if not self.board.row_empty(r):
                for a in range(1, self.board.spot_avail(r) + 1):
                    possible_moves.append((r, a))
        return possible_moves

    def move(self):
        return self.a_star_move()

    def __str__(self):
        return 'A* Player'

class MinMaxPlayer(Player):
    def __init__(self, board, depth = 5):
        super().__init__(board)
        self.depth = depth

    def move(self):
        #max_select = self.game.max_select if self.game.max_select is not None else 1  # Use max_select from game
        max_select = 1
        move = self.minimax(self.board, self.depth, True, float('-inf'), float('inf'), max_select)
        return move[1]

    def minimax(self, board, depth, maximizing, alpha, beta, max_select):
        if depth == 0 or board.g_o():
            return self.evaluate(board), None

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            for r in range(len(board.b)):
                if not board.row_empty(r):
                    for a in range(1, board.spot_avail(r) + 1):
                        new_board = board.dupe()
                        new_board.update_b(r, a)
                        eval = self.minimax(new_board, depth - 1, False, alpha, beta, max_select)[0]
                        if eval > max_eval:
                            max_eval = eval
                            best_move = (r, a)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for r in range(len(board.b)):
                if not board.row_empty(r):
                    for a in range(1, board.spot_avail(r) + 1):
                        new_board = board.dupe()
                        new_board.update_b(r, a)
                        eval = self.minimax(new_board, depth - 1, True, alpha, beta, max_select)[0]
                        if eval < min_eval:
                            min_eval = eval
                            best_move = (r, a)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval, best_move

    def evaluate(self, board):
        # Simple evaluation function
        score = 0
        for row in board.b:
            score += row.count('o')
        return -score

    def __str__(self):
        return 'MinMax Player'