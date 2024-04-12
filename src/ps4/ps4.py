# Problem Set 4: Simulating the Spread of Disease and Bacteria Population Dynamics
# Name:
# Collaborators (Discussion):
# Time:

import math
import numpy as np
import matplotlib.pyplot as plt
import random

# call a random seed for reproducibility
random.seed(0)

##########################
# End helper code
##########################

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacteria
    and ResistantBacteria classes to indicate that a bacteria cell does not
    reproduce. You should use NoChildException as is; you do not need to
    modify it or add any code.
    """


def make_one_curve_plot(x_coords, y_coords, x_label, y_label, title):
    """
    Makes a plot of the x coordinates and the y coordinates with the labels
    and title provided.

    Args:
        x_coords (list of floats): x coordinates to graph
        y_coords (list of floats): y coordinates to graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the graph
    """
    plt.figure()
    plt.plot(x_coords, y_coords)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def make_two_curve_plot(x_coords,
                        y_coords1,
                        y_coords2,
                        y_name1,
                        y_name2,
                        x_label,
                        y_label,
                        title):
    """
    Makes a plot with two curves on it, based on the x coordinates with each of
    the set of y coordinates provided.

    Args:
        x_coords (list of floats): the x coordinates to graph
        y_coords1 (list of floats): the first set of y coordinates to graph
        y_coords2 (list of floats): the second set of y-coordinates to graph
        y_name1 (str): name describing the first y-coordinates line
        y_name2 (str): name describing the second y-coordinates line
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): the title of the graph
    """
    plt.figure()
    plt.plot(x_coords, y_coords1, label=y_name1)
    plt.plot(x_coords, y_coords2, label=y_name2)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


##########################
# PROBLEM 1
##########################

class SimpleBacteria(object):
    """A simple bacteria cell with no antibiotic resistance"""

    def __init__(self, birth_prob, death_prob):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob

    def is_killed(self):
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        # generate a random float between 0 and 1 and check if it is within the 
        # death_prob
        if random.random() <= self.death_prob:
            return True
        else:
            return False

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        self.birth_prob * (1 - pop_density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacteria (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacteria: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        if random.random() <= self.birth_prob * (1 - pop_density):
            return SimpleBacteria(self.birth_prob, self.death_prob)
        else:
            raise NoChildException


class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria (list of SimpleBacteria): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        self.bacteria = bacteria
        self.max_pop = max_pop

    def get_total_pop(self):
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determine whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassign the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        # create a new list of surviving bacteria
        surviving_bacteria = [bact for bact in self.bacteria \
                              if not bact.is_killed()]
        # calculate the population density
        pop_density = len(surviving_bacteria) / self.max_pop
        # create an empty list to contain offspring bacteria
        offspring_bacteria = []
        # loop over surviving bacteria and try to append their offspring 
        # to the list
        for bact in surviving_bacteria:
            try:
                offspring_bacteria.append(bact.reproduce(pop_density))
            # if we get a NoChildException error then do nothing and continue 
            except NoChildException:
                continue
        # reassign the bacteria to the surviving and offspring bacteria
        self.bacteria = surviving_bacteria + offspring_bacteria
        return self.get_total_pop()


##########################
# PROBLEM 2
##########################

def calc_pop_avg(populations, n):
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j

    Returns:
        float: The average bacteria population size at time step n
    """
    # initiate a variable to calculate populations sum
    pop_sum = 0
    # loop over the trials in the populations
    for i in range(len(populations)):
        # add the population size for trial i at n time step to the pop sum
        pop_sum += populations[i][n]
    # average the population by dividing the sum by the number of trials
    pop_avg = pop_sum / len(populations)
    return pop_avg


def simulation_without_antibiotic(num_bacteria,
                                  max_pop,
                                  birth_prob,
                                  death_prob,
                                  num_trials):
    """
    Run the simulation and plot the graph for problem 2. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacteria
        * instantiate a Patient using the list of SimpleBacteria
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. Note
          that the first time step should contain the starting number of
          bacteria in the patient

    Then, plot the average bacteria population size (y-axis) as a function of
    elapsed time steps (x-axis) You might find the make_one_curve_plot
    function useful.

    Args:
        num_bacteria (int): number of SimpleBacteria to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    timesteps = 300
    populations = []
    for _ in range(num_trials):
        # initiate a bacteria list and a patient
        bacteria = [SimpleBacteria(birth_prob, death_prob) for _ \
                    in range(num_bacteria)]
        patient = Patient(bacteria, max_pop)
        # create a list of populations, starting with the initial population
        population_at_timestep = [patient.get_total_pop()]
        # loop over the number of time steps and append the number of population
        # - 1 because the first step is the initial population
        for _ in range(timesteps - 1):
            population_at_timestep.append(patient.update())
        # add the populations to the list
        populations.append(population_at_timestep)
    # plot the average populations by timestep
    # the xs are the timesteps from 0 to timesteps
    xs = [i for i in range(timesteps)]
    # ys is the average population across trials at timestep n
    ys = [calc_pop_avg(populations, n) for n in range(timesteps)]
    make_one_curve_plot(xs, ys, x_label='Timestep',
                        y_label='Average Population',
                        title='Without Antibiotic')
    return populations


# When you are ready to run the simulation, uncomment the next line
# populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 50)

##########################
# PROBLEM 3
##########################

def calc_pop_std(populations, t):
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root

    You may not use third-party functions that calculate standard deviation,
    such as numpy.std. Other built-in or third-party functions that do not
    calculate standard deviation may be used.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    # calculate the average population at timestep t
    pop_avg = calc_pop_avg(populations, t)
    # initiate a variable to calculate differences from the mean
    pop_diff = 0
    # loop over the trials in the populations
    for i in range(len(populations)):
        # add the squared difference between the population and the mean
        pop_diff += (populations[i][t] - pop_avg) ** 2
    # calculate the population standard deviation at time step t
    pop_std =  math.sqrt(pop_diff / len(populations))
    return pop_std


def calc_95_ci(populations, t):
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        I.e., you should return a tuple containing (mean, width)
    """
    pop_avg = calc_pop_avg(populations, t)
    pop_std = calc_pop_std(populations, t)
    # calculate the standard error of the mean
    pop_sem = pop_std / math.sqrt(len(populations))
    pop_width = 1.96 * pop_sem
    return pop_avg, pop_width


##########################
# PROBLEM 4
##########################

class ResistantBacteria(SimpleBacteria):
    """A bacteria cell that can have antibiotic resistance."""

    def __init__(self, birth_prob, death_prob, resistant, mut_prob):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        super(ResistantBacteria, self).__init__(birth_prob, death_prob)
        self.resistant = resistant
        self.mut_prob = mut_prob

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""
        return self.resistant

    def is_killed(self):
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability. If not resistant,
        the bacteria dies with the regular death probability / 4.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        if self.get_resistant():
            return SimpleBacteria.is_killed(self)
        else:
            if random.random() <= self.death_prob / 4:
                return True
            else:
                return False

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacteria, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacteria: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        # check if the bacteria will reproduce
        if random.random() <= self.birth_prob * (1 - pop_density):
            # if yes, check if the bacteria is resistant
            if self.get_resistant():
                # if yes, the offspring is also resistant
                offspring_resistant = self.resistant
            # if not, check if the bacteria mutates
            elif random.random() <= self.mut_prob * (1 - pop_density):
                # the bacteria will have resistance to antibiotics
                offspring_resistant = True
            # if not, then the bacteria will not be resistant
            else:
                offspring_resistant = False
            # return the offspring bacteria
            return ResistantBacteria(self.birth_prob, self.death_prob,
                                     offspring_resistant, self.mut_prob)
        # if it does not reproduce raise an error
        else:
            raise NoChildException


class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function should initialize self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.

        Don't forget to call Patient's __init__ method at the start of this
        method.
        """
        super(TreatedPatient, self).__init__(bacteria, max_pop)
        self.on_antibiotic = False

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotic = True

    def get_resist_pop(self):
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        return sum([bact.get_resistant() for bact in self.bacteria])

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Use the same calculation as in Patient

        4. Based on this value of population density, determine whether each
           surviving bacteria cell should reproduce and add offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        # create a new list of surviving bacteria
        surviving_bacteria = [bact for bact in self.bacteria \
                              if not bact.is_killed()]
        # if the patient is on antibiotics kill all non-resistant bacteria
        if self.on_antibiotic:
            surviving_bacteria = [bact for bact in surviving_bacteria \
                                  if bact.get_resistant()]
            # calculate the population density
        pop_density = len(surviving_bacteria) / self.max_pop
        # create an empty list to contain offspring bacteria
        offspring_bacteria = []
        # loop over surviving bacteria and try to append their offspring 
        # to the list
        for bact in surviving_bacteria:
            try:
                offspring_bacteria.append(bact.reproduce(pop_density))
            # if we get a NoChildException error then do nothing and continue 
            except NoChildException:
                continue
        # reassign the bacteria to the surviving and offspring bacteria
        self.bacteria = surviving_bacteria + offspring_bacteria
        return self.get_total_pop()


##########################
# PROBLEM 5
##########################

def simulation_with_antibiotic(num_bacteria,
                               max_pop,
                               birth_prob,
                               death_prob,
                               resistant,
                               mut_prob,
                               num_trials):
    """
    Runs simulations and plots graphs for problem 4.

    For each of num_trials trials:
        * instantiate a list of ResistantBacteria
        * instantiate a patient
        * run a simulation for 150 timesteps, add the antibiotic, and run the
          simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot. You might find
    the helper function make_two_curve_plot helpful

    Args:
        num_bacteria (int): number of ResistantBacteria to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacteria cells
        num_trials (int): number of simulation runs to execute

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    # all time steps
    timesteps = 400
    # time step at which to start antibiotics
    on_antibiotic_at_step = 149
    # lists to contain all populations and populations of resistant bacteria
    populations_all = []
    populations_resistant = []
    for _ in range(num_trials):
        # initiate a list of resistant bacteria and a patient
        bacteria = [ResistantBacteria(birth_prob, death_prob, resistant,
                                      mut_prob) for _ in range(num_bacteria)]
        patient = TreatedPatient(bacteria, max_pop)
        # create a list of populations, starting with the initial population
        # for all and resistant bacteria
        population_all_at_timestep = [patient.get_total_pop()]
        population_resistant_at_timestep = [patient.get_resist_pop()]
        # loop over the number of time steps and append the number of population
        # - 1 because the first step is the initial population
        for i in range(timesteps - 1):
            # once we reach the time step specified set the patient on antibiotics
            if i == on_antibiotic_at_step:
                patient.set_on_antibiotic()
            # update the populations for both resistant and all
            population_all_at_timestep.append(patient.update())
            population_resistant_at_timestep.append(patient.get_resist_pop())
    # add the populations to the list
    populations_all.append(population_all_at_timestep)
    populations_resistant.append(population_resistant_at_timestep)
    # plot the average populations by timestep
    # the xs are the timesteps from 0 to timesteps
    xs = [i for i in range(timesteps)]
    # ys is the average population across trials at timestep n (all and resistant)
    ys1 = [calc_pop_avg(populations_all, n) for n in range(timesteps)]
    ys2 = [calc_pop_avg(populations_resistant, n) for n in range(timesteps)]
    make_two_curve_plot(xs, ys1, ys2, y_name1='Total', y_name2='Resistant',
                        x_label='Timestep', y_label='Average Population',
                        title='With an Antibiotic')
    return populations_all, populations_resistant


# When you are ready to run the simulations, uncomment the next lines one
# at a time
total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
                                                      max_pop=1000,
                                                      birth_prob=0.3,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)
print('95% CI for all populations at step 299 is:', calc_95_ci(total_pop, 299))
print('95% CI for resistant populations at step 299 is:', calc_95_ci(resistant_pop, 299))

total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
                                                      max_pop=1000,
                                                      birth_prob=0.17,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)
print('95% CI for all populations at step 299 is:', calc_95_ci(total_pop, 299))
print('95% CI for resistant populations at step 299 is:', calc_95_ci(resistant_pop, 299))