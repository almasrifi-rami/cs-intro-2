###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # TODO: Your code here
    # initiate a dictionary to store the cows' information
    cows = {}
    # open the file and read all lines then store them in a list
    # use with to close the file after the operation is done
    with open(filename, mode='r') as fh:
        lines = fh.readlines()
    # loop over each line
    for line in lines:
        # each line refers to a cow
        cow = line
        # split the line on the "," and get the cow name and its weight
        # use strip to remove spaces and new line characters
        cow_name, cow_weight = cow.strip().split(',')
        # store the cows in the dict
        # make sure the weight is an integer
        cows[cow_name] = int(cow_weight)

    return cows


# Problem 2
def greedy_cow_transport(cows, limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    # copy dictionary to avoid mutation
    pending_transport = cows.copy()
    # create a list to store the transported cows
    transported = []
    # sort the dictionary based on the weights of the cows
    pending_transport_sorted = dict(
        sorted(
            pending_transport.items(),
            key=lambda cow: cow[1],
            reverse=True
        )
    )
    # if the pending transportation list is not empty
    while pending_transport_sorted:
        # the remaining space is reset to the original limit
        remaining_space = limit
        # initialize the trip
        trip = []
        # loop over all items in the dictionary
        for cow_name, cow_weight in pending_transport_sorted.items():
            # check  if the cow's weight is less than or equal to the remaining space
            if cow_weight <= remaining_space:
                # add it to the trip
                trip.append(cow_name)
                # update the remaining space
                remaining_space -= cow_weight
        # add the trip to the transported list
        transported.append(trip)
        # update the pending transportation by removing everyone that went on the trip
        for cow_name in trip:
            del pending_transport_sorted[cow_name]
    return transported
                

# Problem 3
def brute_force_cow_transport(cows, limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    # copy the dictionary
    pending_transport = cows.copy()
    # generate all possible trips
    partitions = get_partitions(pending_transport)
    # initiate a best partition tuple where the first element is bigger than the number of cows
    # and the second element is an empty list
    best_partition = (len(cows)+1, [])
    # loop over partitions and each trip in each set
    for partition in partitions:
        # initiate a counter for good trips
        good_trips = 0
        for trip in partition:
            # sum the weight of the cows in the trip
            trip_weight = sum([cows[cow] for cow in trip])
            # if the trip weight is within the limit
            if trip_weight <= limit:
                # increase the number of good trips
                good_trips += 1
            # otherwise break the loop as the partition is not valid
            else:
                break
        # check if the number of good trips in the partition is the same as the number of trips
        # and if the number of trips is lower than the best partition
        if good_trips == len(partition) and len(partition) < best_partition[0]:
            # change the best partition to the new partition
            best_partition = len(partition), partition
    return best_partition[1]
            
        
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # TODO: Your code here
    # load the cows data
    cows = load_cows('src/ps1/ps1_cow_data.txt')
    # start the timer
    greedy_start = time.time()
    # execute the algorithm
    greedy_trips = greedy_cow_transport(cows, limit=10)
    # record the end time
    greedy_end = time.time()
    # print the trips
    print(f'The best transport with the greedy algorithm is:\n')
    print(greedy_trips)
    # get the duration and print it
    greedy_duration = greedy_end - greedy_start
    print(f'The greedy algorithm took {greedy_duration}s to execute!')
    # do the same for the brute force algorithm
    brute_start = time.time()
    brute_trips = brute_force_cow_transport(cows, limit=10)
    brute_end = time.time()
    print(f'The best transport with the brute force alogrithm is:\n')
    print(brute_trips)
    brute_duration = brute_end - brute_start
    print(f'The brute force algorithm took {brute_duration}s to execute!')

    return None


# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    compare_cow_transport_algorithms()