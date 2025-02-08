'''
Creating crossword mini puzzles using a genetic algorithm and a wordnet-based fitness function

Written by Paul Bodily for Computational Creativity
Completed by Andreas Kramer 
Due Date 02/07/2025
Python version 3.9.1
'''
# This is my 9th version of this code - Andreas Kramer

import random
import string
import copy
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
import matplotlib.pyplot as plt 

# Global parameters
DIMENSION = 5 # Puzzle is DIMENSION x DIMENSION (e.g., 4 for a 4x4 puzzle)
ALPHA = 0.5    # Maximum bonus for a quasi-valid word (used in the bonus version)
MAX_SCORE = DIMENSION * 2  # Traditional optimal score: 2 * DIMENSION (for a 4x4 puzzle, 8)
VOWELS = set("AEIOU")

# Global counters for crossover methods
uniform_count = 0
diagonal_count = 0
block_count = 0

def initialize_crossword():
    """Create a random DIMENSION x DIMENSION crossword puzzle."""
    return [[random.choice(string.ascii_uppercase) for _ in range(DIMENSION)] for _ in range(DIMENSION)]

def print_crossword(crossword_to_print):
    """Print the crossword puzzle in a grid format."""
    for row in crossword_to_print:
        print(' '.join(row))

def print_crossword_clues(crossword_to_clue):
    """For each across and down word, print the WordNet definition if available."""
    # Across words
    for i in range(DIMENSION):
        word = ''.join(crossword_to_clue[i])
        syns = wordnet.synsets(word)
        if syns:
            print(f"{i+1} Across: {syns[0].definition()} ({word})")
        else:
            print(f"{i+1} Across: {word} is invalid")
    print()
    # Down words
    for j in range(DIMENSION):
        word = ''.join(crossword_to_clue[i][j] for i in range(DIMENSION))
        syns = wordnet.synsets(word)
        if syns:
            print(f"{j+1} Down: {syns[0].definition()} ({word})")
        else:
            print(f"{j+1} Down: {word} is invalid")

def get_words(puzzle):
    """Return a set of unique words (across and down) from the puzzle."""
    words = set()
    # Across words
    for row in puzzle:
        words.add(''.join(row))
    # Down words
    for j in range(DIMENSION):
        word = ''.join(puzzle[i][j] for i in range(DIMENSION))
        words.add(word)
    return words

# --- Scoring Functions ---

def word_score(word, alpha=ALPHA):
    """
    (Bonus Version)
    Compute a score for a single word.
    
    If the word is valid (appears in WordNet), it is awarded a score of 1.
    Otherwise, it is given a bonus based on the vowel/consonant transitions.
    
    The bonus is calculated as:
      bonus = alpha * (number of transitions / (length(word) - 1))
    
    Thus, a valid word gets 1 point, while an invalid word can get at most alpha.
    """
    if wordnet.synsets(word):
        # Word is valid: reward a full point.
        return 1
    else:
        bonus = 0
        if len(word) > 1:
            transitions = 0
            for i in range(1, len(word)):
                if (word[i] in VOWELS) != (word[i-1] in VOWELS):
                    transitions += 1
            bonus = alpha * (transitions / (len(word) - 1))
        return bonus

def traditional_word_score(word):
    """
    (Traditional Version)
    Compute a score for a single word.
    
    The word gets a score of 1 if it is valid (appears in WordNet),
    and 0 otherwise.
    """
    return 1 if wordnet.synsets(word) else 0

def fitness(puzzle):
    """
    (Bonus Version)
    Compute the overall fitness of a puzzle.
    
    For each unique word (across and down), add its score.
    A puzzle where all words are valid will have an optimal score of 2 * DIMENSION.
    """
    words = get_words(puzzle)
    total_score = 0
    for word in words:
        total_score += word_score(word)
    return total_score

def traditional_fitness(puzzle):
    """
    (Traditional Version)
    Compute the overall fitness of a puzzle.
    
    For each unique word (across and down), add its score (1 if valid, 0 if not).
    A puzzle where all words are valid will have an optimal score of 2 * DIMENSION.
    """
    words = get_words(puzzle)
    total_score = 0
    for word in words:
        total_score += traditional_word_score(word)
    return total_score

def all_valid(puzzle):
    """
    Check if every unique word in the puzzle is a valid word.
    
    Returns True if every word is valid (i.e., found in WordNet), else False.
    """
    words = get_words(puzzle)
    return all(wordnet.synsets(word) for word in words)

# --- Crossover Operators ---

def uniform_crossover(parent1, parent2):
    """Perform a simple cell-by-cell (uniform) crossover."""
    child = copy.deepcopy(parent1)
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if random.random() < 0.5:
                child[i][j] = parent2[i][j]
    return child

def diagonal_crossover(parent1, parent2):
    """
    Perform a diagonal crossover.
    
    For cells on or above the main diagonal (i <= j), take from parent1.
    For cells below the main diagonal (i > j), take from parent2.
    """
    child = copy.deepcopy(parent1)
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if i > j:
                child[i][j] = parent2[i][j]
    return child

def block_based_crossover(parent1, parent2, block_size=2):
    """
    Perform a block-based crossover.
    
    The grid is partitioned into blocks of size block_size x block_size.
    For each block, randomly choose to copy that entire block from parent1 or parent2.
    """
    child = [[None for _ in range(DIMENSION)] for _ in range(DIMENSION)]
    for row_start in range(0, DIMENSION, block_size):
        for col_start in range(0, DIMENSION, block_size):
            row_end = min(row_start + block_size, DIMENSION)
            col_end = min(col_start + block_size, DIMENSION)
            if random.random() < 0.5:
                source = parent1
                #print(f"Block rows {row_start}-{row_end-1}, cols {col_start}-{col_end-1} from Parent 1")
            else:
                source = parent2
                #print(f"Block rows {row_start}-{row_end-1}, cols {col_start}-{col_end-1} from Parent 2")
            for i in range(row_start, row_end):
                for j in range(col_start, col_end):
                    child[i][j] = source[i][j]
    #print("Child produced by block-based crossover:")
    #print_crossword(child)
    return child

def random_diagonal_crossover(parent1, parent2):
    """
    Perform a diagonal crossover by randomly selecting among multiple diagonal strategies.
    
    Variants:
      1. Main Diagonal Variant:
         - For cells where i <= j, take from parent1.
         - For cells where i > j, take from parent2.
         
      2. Anti-Diagonal Variant:
         - Define the anti-diagonal as i + j == DIMENSION - 1.
         - For cells where i + j <= DIMENSION - 1, take from parent1.
         - For cells where i + j > DIMENSION - 1, take from parent2.
         
      3. Flipped Main Diagonal Variant:
         - For cells where i <= j, take from parent2.
         - For cells where i > j, take from parent1.
    
    The function randomly selects one of these three strategies and returns the child puzzle.
    """
    child = [[None for _ in range(DIMENSION)] for _ in range(DIMENSION)]
    
    # Randomly choose one variant (0: main, 1: anti-diagonal, 2: flipped main)
    variant = random.choice([0, 1, 2])
    #print(f"Selected diagonal variant {variant} for crossover.")  # Uncomment for debugging
    
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if variant == 0:
                # Main Diagonal Variant
                if i <= j:
                    child[i][j] = parent1[i][j]
                else:
                    child[i][j] = parent2[i][j]
            elif variant == 1:
                # Anti-Diagonal Variant
                if i + j <= DIMENSION - 1:
                    child[i][j] = parent1[i][j]
                else:
                    child[i][j] = parent2[i][j]
            else:  # variant == 2
                # Flipped Main Diagonal Variant
                if i <= j:
                    child[i][j] = parent2[i][j]
                else:
                    child[i][j] = parent1[i][j]
    
    # Optionally, you can print the chosen variant and resulting child for debugging:
    #print("Child produced by random diagonal crossover (variant", variant, "):")
    #print_crossword(child)
    
    return child


def choose_crossover(parent1, parent2):
    """
    Randomly choose one of several crossover strategies:
      - Uniform crossover
      - Diagonal crossover
      - Block-based crossover
    """
    global uniform_count, diagonal_count, block_count
    r = random.random()
    if r < 0.33:
        uniform_count += 1
        #print("Using uniform crossover.")
        return uniform_crossover(parent1, parent2)
    elif r < 0.66:
        diagonal_count += 1
        #print("Using diagonal crossover.")
        return random_diagonal_crossover(parent1, parent2)
    else:
        block_count += 1
        #print("Using block-based crossover.")
        return block_based_crossover(parent1, parent2, block_size=2)
    
def choose_crossover_adaptive(parent1, parent2, current_best):
    """
    Progressive (Adaptive) Crossover Strategy:
    
    Chooses a crossover method based on the current best fitness relative to the optimal score.
    
    - If current_best < 0.5 * MAX_SCORE: (Early Phase) favor uniform crossover (70%),
      diagonal (20%), block-based (10%).
      
    - If 0.5 * MAX_SCORE <= current_best < 0.8 * MAX_SCORE: (Mid Phase) favor diagonal crossover (50%),
      uniform (30%), block-based (20%).
      
    - If current_best >= 0.8 * MAX_SCORE: (Late Phase) favor block-based crossover (60%),
      diagonal (30%), uniform (10%).
    """
    global uniform_count, diagonal_count, block_count
    ratio = current_best / MAX_SCORE
    if ratio < 0.5:
        probs = [0.7, 0.2, 0.1]  # [uniform, diagonal, block-based]
    elif ratio < 0.8:
        probs = [0.3, 0.5, 0.2]
    else:
        probs = [0.1, 0.3, 0.6]
    
    r = random.random()
    if r < probs[0]:
        uniform_count += 1
        #print("Using uniform crossover (adaptive).")
        return uniform_crossover(parent1, parent2)
    elif r < probs[0] + probs[1]:
        diagonal_count += 1
        #print("Using diagonal crossover (adaptive).")
        return random_diagonal_crossover(parent1, parent2)
    else:
        block_count += 1
        #print("Using block-based crossover (adaptive).")
        return block_based_crossover(parent1, parent2, block_size=2)

# --- Mutation Operator ---

def mutate(puzzle, mutation_rate):
    """
    Mutate the puzzle by randomly changing letters based on the mutation_rate.
    """
    new_puzzle = copy.deepcopy(puzzle)
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if random.random() < mutation_rate:
                old_letter = new_puzzle[i][j]
                new_letter = random.choice(string.ascii_uppercase)
                new_puzzle[i][j] = new_letter
                #print(f"Mutated cell ({i},{j}) from {old_letter} to {new_letter}")
    return new_puzzle

# --- Genetic Algorithm ---

def run_ga(gens, population_size=20, children_per_generation=20, mutation_rate=0.25):
    """
    Run the genetic algorithm to evolve crossword puzzles.
    
    Returns the final population, the best fitness progress log, and the total number of generations executed.
    Uses the traditional scoring mechanism (only valid words are rewarded).
    
    Only prints a summary message whenever we obtain a new best score.
    Also prints the generation number and the current counts of each crossover method.
    """
    population = []
    #print("Initializing population...")
    for i in range(population_size):
        puzzle = initialize_crossword()
        fit = fitness(puzzle)  # Use traditional_fitness here
        population.append((puzzle, fit))
        #print(f"Individual {i} fitness: {fit}")
    #print("=" * 50)
    
    best_fitness_progress = []
    generation_count = 0  # Track the number of generations
    best_overall = 0  # To track the best fitness seen so far
    
    for gen in range(gens):
        generation_count += 1
        #print(f"\n--- Generation {gen} ---")
        population.sort(key=lambda x: x[1], reverse=True)
        best_individual = population[0]
        current_best = best_individual[1]
        best_fitness_progress.append(current_best)
        
        # Only print if we get a better score than before
        if current_best > best_overall:
            best_overall = current_best
            print(f"Generation {gen}: New best fitness = {current_best} "
                  f"(Uniform: {uniform_count}, Diagonal: {diagonal_count}, Block: {block_count})")
        
        # Early exit: terminate only if every word is truly valid (optimal score reached)
        if best_individual[1] == MAX_SCORE:
            print(f"Found perfect puzzle at generation {gen}")
            break
        
        children = []
        for child_index in range(children_per_generation):
            if population_size == 1:
                # Only mutate the single individual (no crossover possible)
                parent = population[0][0]  # Extract the individual from the tuple
                child = mutate(parent, mutation_rate)  
                child_fit = fitness(child)  
            else:
                parent_candidates = population[:max(1, population_size // 2)]
                tournament = random.sample(parent_candidates, 2)
                parent1, parent2 = tournament[0][0], tournament[1][0]
                #print(f"Child {child_index}: Selected two parents for crossover.")
                child = choose_crossover(parent1,parent2) #diagonal_crossover(parent1, parent2) #uniform_crossover(parent1,parent2) # #
                child = mutate(child, mutation_rate)
                child_fit = fitness(child)  # Use traditional fitness for scoring
                #print(f"Child {child_index} fitness: {child_fit}")
            children.append((child, child_fit))
        
        population.extend(children)
        population.sort(key=lambda x: x[1], reverse=True)
        #print("Population fitnesses after combining:")
        #for idx, individual in enumerate(population):
        #    print(f"Individual {idx}: Fitness = {individual[1]}")
        population = population[:population_size]
    
    return population, best_fitness_progress, generation_count

# --- Function to Run GA and Plot Progress ---
def run_and_plot_ga(num_runs, gens, population_size, children_per_generation, mutation_rate):
    """
    Runs the GA for num_runs times and plots the best fitness progress for a single run.
    If num_runs is 1, simply plots the progress.
    If num_runs > 1, returns the list of progress arrays so you can later compute averages manually.
    """
    # For now, if num_runs == 1, we run the GA and plot the progress.
    if num_runs == 1:
        final_population, progress, gens_executed = run_ga(
            gens,
            population_size=population_size,
            children_per_generation=children_per_generation,
            mutation_rate=mutation_rate
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(progress, label='Best Fitness Progress')
        plt.xlabel('Generation', fontsize=25)
        plt.ylabel('Best Fitness Score So Far',fontsize=25)
        plt.title('GA Best Fitness Progress', fontsize=25)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()
        
        return final_population, progress, gens_executed
    else:
        # If num_runs > 1, store each run's progress in a list.
        all_progress = []
        gens_executed_list = []
        for r in range(num_runs):
            print(f"\n--- Run {r+1} ---")
            final_population, progress, gens_executed = run_ga(
                gens,
                population_size=population_size,
                children_per_generation=children_per_generation,
                mutation_rate=mutation_rate
            )
            all_progress.append(progress)
            gens_executed_list.append(gens_executed)
        return final_population, all_progress, gens_executed_list

# --- Main Execution ---
if __name__ == '__main__':
    # Set GA parameters
    gens = 10000
    pop_size = 100
    children_per_gen = 10
    mutation_rate = 0.20

    # Change num_runs to 1 for a single run (graphing one run) 
    # or to >1 if you want to collect multiple runs for averaging later.
    num_runs = 1

    best_population, progress, gens_executed = run_and_plot_ga(num_runs, gens, pop_size, children_per_gen, mutation_rate)
    
    best_crossword = best_population[0][0]
    print("\nFinal Best Crossword:")
    print_crossword(best_crossword)
    print("Clues:")
    print_crossword_clues(best_crossword)
    print(f"Total generations executed: {gens_executed}")

    """
if __name__ == '__main__':
    gens = 10000
    pop_size = 20
    children_per_gen = 20
    mutation_rate = 0.5
    
    final_population, progress, generations_executed = run_ga(
        gens,
        population_size=pop_size,
        children_per_generation=children_per_gen,
        mutation_rate=mutation_rate
    )
    
    best_crossword = final_population[0]
    print("\nFinal Best Crossword:")
    print_crossword(best_crossword[0])
    print("Final Fitness:", best_crossword[1])
    print("Clues:")
    print_crossword_clues(best_crossword[0])
    print(f"Total generations executed: {generations_executed}")

    # --- Plotting the Best Fitness Progress ---
    plt.figure(figsize=(10, 6))
    plt.plot(progress, label='Best Fitness Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score So Far')
    plt.title('GA Best Fitness Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
