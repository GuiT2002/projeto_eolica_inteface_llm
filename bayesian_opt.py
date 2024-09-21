import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from wind_farm_GA import main, toolbox
from deap import tools, creator
import random

IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo

# Definindo o espaço de busca para os parâmetros
space = [
    Real(0.5, 0.9, name='cxpb'),      # Proporção de cruzamento
    Real(0.2, 0.5, name='mutpb'),     # Proporção de mutação
    Integer(100, 500, name='pop'),    # Tamanho da população
    Integer(2, 6, name='torneio'),    # Tamanho do torneio
    Real(0.4, 0.75, name='alpha'),    # Proporção genética dos pais para os filhos
    Integer(10, 11, name='gen'),   # Número de gerações
    Real(0.15, 0.4, name='indpb'),    # Proporção de genes mutados
    Integer(10, 150, name='sigma')    # Desvio padrão da mutação
]
def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            # Ajusta a turbina para ficar dentro do círculo
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        # Garantir que a turbina permaneça dentro do círculo após mutação
        enforce_circle(individual)
    return creator.Individual(individual.tolist()),

# Função de avaliação para a otimização bayesiana
@use_named_args(space)
def evaluate_ga(cxpb, mutpb, pop, torneio, alpha, gen, indpb, sigma):
    # Atualizando parâmetros no toolbox
    toolbox.register("mate", tools.cxBlend, alpha=alpha)
    toolbox.register("mutate", mutate, mu=0, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=torneio)

    # Rodando o algoritmo genético com os parâmetros fornecidos
    pop, stats, hof = main()

    # Pegando o melhor valor de AEP (fitness) alcançado
    best_fitness = hof[0].fitness.values[0]

    # Queremos maximizar o AEP, mas a otimização bayesiana minimiza por padrão
    # Portanto, retornamos o negativo do fitness
    return -best_fitness

# Rodando a otimização bayesiana
result = gp_minimize(evaluate_ga, space, n_calls=10, random_state=42)

# Exibindo os melhores parâmetros encontrados
print("Melhores parâmetros encontrados:")
print(f"cxpb = {result.x[0]}")
print(f"mutpb = {result.x[1]}")
print(f"pop = {result.x[2]}")
print(f"torneio = {result.x[3]}")
print(f"alpha = {result.x[4]}")
print(f"gen = {result.x[5]}")
print(f"indpb = {result.x[6]}")
print(f"sigma = {result.x[7]}")
