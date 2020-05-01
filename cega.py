"""
cega.py
--------------------
Genetic algorithm to find effective information in coarse grained networks
https://arxiv.org/pdf/1907.03902.pdf

author: Julian Amacker
email: julian.amacker at gmail dot com
"""
import networkx as nx
import numpy as np
from ei_net.ei_net import *
from ei_net.ce_net import *
import random
import copy

TPM_example1 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5],
                        [1/3, 0.0, 1/3, 1/3, 0.0],
                        [0.0, 0.5, 0.0, 0.5, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.5, 0.0, 0.0, 0.5, 0.0]])


#An example real world network (cat brain connectivity). Please ensure the path to the file is corrent
catbraiNX = nx.read_edgelist('/Users/julian/Downloads/bn-cat-mixed-species_brain_1/bn/bn-cat-mixed-species_brain_1.edges')
catbrain = nx.to_numpy_array(catbraiNX)


##################
#fitness function
##################
def EI(TPM):
    """
    This function calculates the effect information of a graph.
    :param TPM: transition matrix
    :return: effective information
    """
    l = len(TPM)
    w_in = entropy(np.sum(TPM, axis=0) / l, base=2)
    w_out = sum([entropy(i, base=2) for i in TPM]) / l
    #EI = w_in - w_out
    return w_in - w_out

def fitness(TPM, individual):
    '''
    this function returns the fitness of an individual
    :param TPM: transition matrix
    :param individual: dict, genome of an individual
    :return: int, fitness
    '''
    ind_pheno = phenotyp(TPM, individual)
    #helper function from external module
    macro_network = create_macro(TPM, ind_pheno)
    ind_fitness = EI(macro_network)
    return ind_fitness

##################
# Solution space
##################
def solution_space(TPM):
    '''
    This function creates a list of all allowed connections in a network
    :param TPM: transition matrix
    :return: a list of list containing all possible links for a given node
    '''
    solution_space = []
    for row in TPM:
        #add zero to allow for single nodes with no self connections
        nonzero = np.nonzero(row)[0]
        if np.count_nonzero(nonzero) - len(nonzero) == 0:
            nonzero = np.append(0, nonzero)

        solution_space.append(list(nonzero))
    return solution_space

def random_genome(TPM, sol_space):
    '''
    this function creates a random genome in the solution space
    :param TPM: transition matrix of the network
    :param sol_space: solution space
    :return: a random genome as dictionary
    '''
    nodes = range(len(TPM))
    links = [np.random.choice(sol_space[n], 1)[0] for n in nodes]

    return dict(zip(nodes, links))

def init_population(TPM, sol_space, n):
    '''
    create an initial population of n individuals
    :param TPM: transition matrix
    :param sol_space: solution space
    :param n: number of individuals
    :return: list of dictionaries representing the genomes
    '''
    return [random_genome(TPM, sol_space) for i in range(n)]


def fill_phenos(genome, dict_pheno, group, u):
    '''
    helper function for the function phenotyp. Fills in group cluster recursively
    :param genome: dict, the genome
    :param dict_pheno: dict, the phenotype mapping of the genome
    :param group: current group
    :param u: int, iterator
    :return:
    '''
    dict_pheno[u] = group
    if genome[u] in dict_pheno:
        return
    else:
        fill_phenos(genome, dict_pheno, group, genome[u])



def phenotyp(TPM, genome):
    '''
    this function creates a TPM of the macro network defined by the genome
    :param TPM: tranistion matrix
    :param genome: dict, genome of an individual
    :return: dict, the phenotype mapping from the genome
    '''
    dict_pheno = dict()
    group = 0
    for i in range(len(genome)):
        if i not in dict_pheno and genome[i] not in dict_pheno:
            fill_phenos(genome, dict_pheno, group, i)
            group += 1
        else:
            dict_pheno[i] = dict_pheno[genome[i]]

    #values for the function create_macro start from 1 not 0
    for k, v in dict_pheno.items():
        dict_pheno[k] = v + 1
    return dict_pheno

def cross_over(cross_prob, ind1, ind2):
    '''
    crossover with random binary vector. 100% cross_prob means nearly every bit is flipped
    :param cross_prob: int, crossover probabilty
    :param ind1: dict, genome of indiviual 1
    :param ind2: dict, genome of indiviual 2
    :return: tupple, children
    '''
    for key in ind1.keys():
        if np.random.random() < cross_prob:
            ind1[key], ind2[key] = ind2[key], ind1[key]
    return ind1, ind2

def mutation(mut_prob, ind, sol_space):
    '''
    this function changes every locus with a probability of mut_prob
    :param mut_prob: probabilty of individual mutation
    :param ind: genome of individual
    :param sol_space: possible solutions for the mutation
    :return: mutated individual
    '''
    for key in ind.keys():
        if np.random.random() < mut_prob:
            ind[key] = np.random.choice(sol_space[key], 1)[0]
    return ind

def WOF(pop_sorted, fit_sorted):
    '''
    wheel of fortune. Returns an individual by weighted random sampling
    :param pop_sorted: list of individuals in ascending order according to fitness values
    :param fit_sorted: list of fitness values in ascending order
    :return: selected individual
    '''
    total_fitness = sum(pop_fitness)

    pointer = np.random.random()
    wheel = np.cumsum(fit_sorted)/total_fitness
    for i in range(len(pop_sorted)):
        if pointer < wheel[i]:
            return pop_sorted[i]



####################################
#######  Genetic Algorithm  ########
####################################
#define TPM: choose between catbrain and TPMexample1
TPM = catbrain
#parameters
sol_space = solution_space(TPM)
pop_size = 100
mutation_rate = 0.1
crossover_rate = 0.3
generations = 50
elitism = 0.2
#percentage of population considered for selection
parents = 0.4

np.random.seed(0)

#create an initial population with 2 x size and take the best n as initial population
print('intialise population')
init_pop = init_population(TPM, sol_space, pop_size*5)
pop_fitness = []

#find the best individual
for individual in init_pop:
    ind_fitness = fitness(TPM, individual)
    pop_fitness.append(ind_fitness)

# sort ascending
sort_index = np.argsort(pop_fitness)
pop_sorted = [init_pop[i] for i in sort_index]
fit_sorted = [pop_fitness[i] for i in sort_index]

population = pop_sorted[-pop_size:]

#store an elite
elite = pop_sorted[-1], fit_sorted[-1]


#population = init_population(TPM, sol_space, pop_size)

#run for n generations
for i in range(generations):
    # calculate fitness of individuals
    if i%10 == 0:
        print('generation:', i)

    pop_fitness = []
    #print(population)
    #calculate the fitness values
    pop_fitness = [fitness(TPM, ind) for ind in population]
    sort_index = np.argsort(pop_fitness)
    pop_sorted = [population[i] for i in sort_index]
    fit_sorted = [pop_fitness[i] for i in sort_index]

    #store new elite
    print('highest found effective information is:', fit_sorted[-1], fit_sorted[0])
    if fit_sorted[-1] > elite[1]:
        elite = pop_sorted[-1], fit_sorted[-1]
        print('new elite:', elite[0], elite[1])

    # select elitists
    elites = round(elitism * pop_size)
    new_pop = copy.deepcopy(pop_sorted[-elites:])
    pop_sorted[-elites:]
    descentents = []

    #crossover and mutation
    pop_for_selection = pop_sorted[round((1-parents) * pop_size):]
    for i in range(pop_size - elites):
        # select 2 individuals with the wheel of fortune
        ind1, ind2 = [WOF(pop_sorted, fit_sorted) for i in range(2)]
        children = cross_over(crossover_rate, ind1, ind2)
        #calculate fitness of 2 children; take the better solution
        fitness1 = fitness(TPM, children[0])
        fitness2 = fitness(TPM, children[1])

        sort_ch = np.argsort((fitness1, fitness2))
        children_sorted = [children[i] for i in sort_ch]
        mutant = mutation(mutation_rate, children_sorted[1], sol_space)
        descentents.append(mutant)

    #update population
    population = descentents + new_pop
    print('''''''''''''''''''''''''''''''''''''''''')


print('The macro-scale network with the highest effective information is:', elite[0])
print('The effective information of the found macro-scale is:', elite[1])


