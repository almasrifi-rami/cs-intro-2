###########################
# 6.0002 Problem Set 1b: Space Change
# Name:
# Collaborators:
# Time:
# Author: charz, cdenise

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo={}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # TODO: Your code here
    # initialize min_eggs for later comparison
    min_eggs = float('inf')
    # define a base case
    if target_weight == 0:
        return 0
    # check whether result for subproblem is already in memo
    elif target_weight in memo:
        return memo[target_weight]
    else:    
        # loop over all eggs (because we can wait / resampling)
        for weight in egg_weights:
            # skip if the egg is bigger than the space we have
            if weight > target_weight:
                break
            # if the weight is suitable
            elif weight <= target_weight:
                # calculate the remaining weight
                remaining_weight = target_weight - weight
                # check how many eggs are needed for the remaining of the weight
                # use recursive programming
                eggs_needed_for_remaining = dp_make_weight(egg_weights,
                                                           remaining_weight,
                                                           memo)
                # add 1 to the remaining eggs and calculate the minimum at each step
                # in the for loop
                min_eggs = min(min_eggs, 1 + eggs_needed_for_remaining)
        # store the minimum number of eggs in the memo
        memo[target_weight] = min_eggs
        return min_eggs


# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()