#!/usr/bin/env python3
import numpy as np
import scipy, random, itertools, sys, argparse

# For analysis purposes we track the number of iterations until convergence.
global total_iters_centrality
total_iters_centrality = 0

# Global variable to track comparison mode
global comparison_mode
comparison_mode = "scalar"  # Default mode

def rank_centrality(A, tol=1e-8, max_iters=100000):
    """
    Implements the rank centrality algorithm for pairwise comparisons.  The
    argument "A" is an n x n matrix such that A_ij / (A_ij + A_ji) represents
    the probability that item j is preferred to item i.  For example in a
    tournament, A_ij could be the number of times that player j beat player i.
    The matrix may be sparse (i.e. some (i,j) pairs may have no comparisons)
    but the graph induced by non-zero arcs must be fully connected.
    
    Returns a vector "scores" of length n such that scores[i] is proportional
    to the global preference for item i.  Its entries sum approximately to 1,
    i.e. it is a probability distribution over the n items.

    tol: iteration stops when sum(abs(scores - prev_scores)) < tol
    max_iters: the algorithm also stops after this many iterations
    """
    # Compute a normalized matrix W such that the probabilities for each (i, j)
    # pair sum to 1.
    n = A.shape[0]
    W = np.zeros((n, n))
    for (i, j) in itertools.product(range(n), range(n)):
        if A[i, j] + A[j, i] > 0:  # Only process pairs with at least one comparison
            W[i, j] = A[j, i] / (A[i, j] + A[j, i])  # FIXED: A[j,i] in numerator instead of A[i,j]
    
    # Compute a transition matrix P whose non-diagonal entries are proportional
    # to W but where every row sums to exactly 1.  To do this, we first compute
    # the maximum sum of any row of W excluding the diagonal entry.
    w_max = max(sum(W[i, j] for j in range(n) if j != i) for i in range(n))

    # Now define the transition matrix P by dividing all non-diagonal entries
    # by w_max and setting the diagonal entry to one minus the sum of the
    # non-diagonal entries.  Note that w_max has been chosen to make the
    # diagonal entries as small as possible while ensuring that no value is
    # negative.  This maximizes the convergence rate in the loop below.
    P = W / w_max
    for i in range(n):
        P[i, i] = 1 - sum(P[i, k] for k in range(n) if k != i)

    # If n is large enough, it is more efficient in the loop below to use
    # a sparse representation for the matrix P.
    if n >= 250: P = scipy.sparse.csr_array(P)
    
    # Finally, compute the stationary distribution of the Markov chain defined
    # by the transition matrix P.  We start with an arbitrary distribution
    # "scores" and iterate by applying the transition matrix repeatedly.
    prev_scores = np.ones(n) / n
    for iter in range(max_iters):
        scores = prev_scores @ P
        if np.sum(np.abs(scores - prev_scores)) < tol: break
        prev_scores = scores

    global total_iters_centrality
    total_iters_centrality += iter + 1
    return scores

def rank_elo(comparison_matrix, K=32):
    """
    Implements traditional ELO ranking for pairwise comparisons.
    
    Args:
        comparison_matrix: n x n matrix where entry (i,j) is raw weight
        K: ELO K-factor that determines how quickly ratings change
    
    Returns:
        scores: list of ELO ratings for each item
    """
    global comparison_mode
    n = len(comparison_matrix)
    scores = np.ones(n) * 1500  # Initialize all ratings to 1500
    
    # Process each comparison
    for i in range(n):
        for j in range(n):
            if i >= j or comparison_matrix[i,j] + comparison_matrix[j,i] == 0:
                continue
                
            # Calculate expected scores
            diff = scores[i] - scores[j]
            expected_i = 1 / (1 + 10 ** (-diff/400))
            expected_j = 1 - expected_i
            
            # Calculate actual scores from comparison matrix
            total_games = comparison_matrix[i,j] + comparison_matrix[j,i]
            
            if comparison_mode == "scalar":
                # In scalar mode, use the actual weights from the matrix
                actual_i = comparison_matrix[i,j] / total_games
                actual_j = comparison_matrix[j,i] / total_games
            else:
                # In binary mode, winner takes all
                actual_i = 1.0 if comparison_matrix[i,j] > comparison_matrix[j,i] else 0.0
                actual_j = 1.0 - actual_i
            
            # Update ratings
            scores[i] += K * (actual_i - expected_i)
            scores[j] += K * (actual_j - expected_j)
    
    return scores

def add_comparison(i, j, A):
    """
    Adds a comparison between nodes i and j to the comparison matrix A by
    updating the weights A_ij and A_ji appropriately.
    
    In scalar mode:
        Nodes are preferred in proportion to their numerical value plus one.
        For example, node 4 is preferred to node 6 with probability 5/(5+7)
        whereas node 6 is preferred to node 4 with probability 7/(7+5).
    
    In binary mode:
        The higher-valued node always wins (deterministic).
        If j > i, then A[i,j] = 1 and A[j,i] = 0
        If i > j, then A[i,j] = 0 and A[j,i] = 1
    """
    global comparison_mode
    
    if comparison_mode == "binary":
        # Binary comparison - higher value always wins
        if j > i:
            A[i, j] += 1  # j wins
            A[j, i] += 0  # i loses
        else:
            A[i, j] += 0  # j loses
            A[j, i] += 1  # i wins
    else:  # scalar mode
        # Weight nodes according to the node value plus one
        w = (j + 1) / (i + j + 2)
        A[i, j] += w
        A[j, i] += 1 - w

def make_comparison_matrix(n, extra_comparisons=0):
    """
    Generates a comparison matrix containing n - 1 random comparisons that
    form a spanning tree (thus ensuring graph connectivity), plus an additional
    "extra_comparisons" random comparisons.  A given pair of nodes may be
    compared more than once, in which case the comparison data is summed.
    
    Returns an n x n matrix "A" where A_ij / (A_ij + A_ji) represents the
    fraction of times that node i was preferred to node j
    """
    # First build a random spanning tree.
    result = np.zeros((n, n))
    perm = random.sample(range(n), n)
    for k in range(1, n):
        i = random.choice(perm[:k])
        j = perm[k]
        add_comparison(i, j, result)

    # Then add any extra comparisons requested.
    for _ in range(extra_comparisons):
        i = random.randrange(n)
        j = (i + 1 + random.randrange(n - 1)) % n
        add_comparison(i, j, result)

    return result

def run_test(n, extra_comparisons):
    # Generate comparison matrix once to use for all methods
    comparison_matrix = make_comparison_matrix(n, extra_comparisons)
    
    # Get rankings from both methods
    scores_centrality = rank_centrality(comparison_matrix)
    scores_elo = rank_elo(comparison_matrix)
    
    # True ranking is just the indices in reverse order (n-1 should be highest ranked)
    true_ranking = np.arange(n)[::-1]
    
    # Calculate Kendall Tau correlation with true ranking
    tau_centrality, p_centrality = scipy.stats.kendalltau(scores_centrality, true_ranking)
    tau_elo, p_elo = scipy.stats.kendalltau(scores_elo, true_ranking)
    
    print("Comparing rankings (Centrality vs ELO):")
    print("Index: Centrality Score | ELO Score")
    print("-" * 50)
    
    # Sort by centrality score and print comparison
    for i in sorted(range(n), key=lambda i: scores_centrality[i], reverse=True):
        print(f'{i}: {scores_centrality[i] * n:.4f} | {scores_elo[i]:.1f}')
    
    print(f'Iterations - Centrality: {total_iters_centrality}')
    print(f'Kendall Tau vs true ranking:')
    print(f'  Centrality: tau={tau_centrality:.4f} (p-value: {p_centrality:.4f})')
    print(f'  ELO: tau={tau_elo:.4f} (p-value: {p_elo:.4f})')

def compare_convergence(n, max_comparisons=100000000):
    """
    Compares how many comparisons ELO and rank_centrality need to achieve correct ranking.
    
    Args:
        n: number of items to rank
        max_comparisons: maximum number of extra comparisons to try
    
    Returns:
        (elo_comparisons, centrality_comparisons): tuple of comparisons needed for each method
    """
    # True ranking should be reverse order (n-1 is highest ranked)
    true_ranking = np.arange(n)[::-1]
    
    # Try increasing numbers of comparisons until both methods converge
    elo_found = False
    centrality_found = False
    elo_comparisons = max_comparisons
    centrality_comparisons = max_comparisons
    
    # Use smaller step size for more precise measurement
    step_size = max(1, n//20)
    
    for extra in range(0, max_comparisons, step_size):
        # Use the same comparison matrix for both methods to ensure fair comparison
        comparison_matrix = make_comparison_matrix(n, extra)
        
        # Check ELO ranking
        if not elo_found:
            scores_elo = rank_elo(comparison_matrix)
            
            # Check if ranking is correct (all items in descending order)
            # We're checking if each item's score is higher than the next item's score
            elo_correct = all(scores_elo[i] > scores_elo[i+1] for i in range(n-1))
            
            if elo_correct:
                elo_found = True
                elo_comparisons = extra + n - 1
                print(f"\nELO converged after {elo_comparisons} comparisons")
        
        # Check rank_centrality ranking
        if not centrality_found:
            scores_centrality = rank_centrality(comparison_matrix)
            
            # Check if ranking is correct (all items in descending order)
            centrality_correct = all(scores_centrality[i] > scores_centrality[i+1] for i in range(n-1))
            
            if centrality_correct:
                centrality_found = True
                centrality_comparisons = extra + n - 1
                print(f"\nRank Centrality converged after {centrality_comparisons} comparisons")
        
        if elo_found and centrality_found:
            break
            
        if extra % 1000 == 0:
            print(f"\nTried {extra} extra comparisons...")
            
            # Print some diagnostic information
            if not elo_found:
                incorrect_elo = [(i, scores_elo[i], scores_elo[i+1]) 
                               for i in range(n-1) 
                               if scores_elo[i] <= scores_elo[i+1]]
                print(f"ELO has {len(incorrect_elo)} incorrect pairs")
                
            if not centrality_found:
                incorrect_centrality = [(i, scores_centrality[i], scores_centrality[i+1]) 
                                      for i in range(n-1) 
                                      if scores_centrality[i] <= scores_centrality[i+1]]
                print(f"Centrality has {len(incorrect_centrality)} incorrect pairs")
    
    print(f"\nConvergence comparison for {n} items:")
    print(f"ELO needed {elo_comparisons} comparisons")
    print(f"rank_centrality needed {centrality_comparisons} comparisons")
    
    return elo_comparisons, centrality_comparisons

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare ranking algorithms')
    parser.add_argument('n', type=int, help='Number of items to rank')
    parser.add_argument('extra_comparisons', type=int, nargs='?', default=0, 
                        help='Number of extra comparisons beyond the spanning tree')
    parser.add_argument('iters', type=int, nargs='?', default=1,
                        help='Number of iterations to run')
    parser.add_argument('--binary', action='store_true', 
                        help='Use binary comparisons (higher value always wins)')
    parser.add_argument('--scalar', action='store_true',
                        help='Use scalar comparisons (default, weighted by node values)')
    
    args = parser.parse_args()
    
    # Set comparison mode based on flags
    if args.binary:
        comparison_mode = "binary"
        print("Using binary comparison mode (higher value always wins)")
    elif args.scalar or not args.binary:  # Default to scalar if neither or only scalar is specified
        comparison_mode = "scalar"
        print("Using scalar comparison mode (weighted by node values)")
    
    # Run tests
    for _ in range(args.iters):
        run_test(args.n, args.extra_comparisons)
    print(f'Total iterations: {total_iters_centrality}')
    
    # Add convergence comparison
    compare_convergence(args.n)