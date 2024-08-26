import numpy as np
from pyswarm import pso
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution, create_animation

# Parâmetros
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
MIN_DISTANCE = 130  # Distância mínima entre turbinas

# Carregando os dados dos arquivos YAML
turb_coords, fname_turb, fname_wr = getTurbLocYAML("iea37-ex16.yaml")
turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")

# Função de avaliação
def evaluate(individual):
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Calculando o AEP
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    return -np.sum(aep)  # Retorna o negativo porque o PSO no pyswarm maximiza a função

# Função para verificar se um ponto está dentro do círculo
def constraint_circle(individual):
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    constraints = [CIRCLE_RADIUS**2 - (x**2 + y**2) for x, y in turb_coords]
    return min(constraints)  # Retorna o valor mínimo

# Função para verificar a distância mínima entre as turbinas
def constraint_distance(individual):
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    constraints = []
    
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            constraints.append(dist - MIN_DISTANCE)
    
    return min(constraints)  # Retorna o valor mínimo

# Limites para as coordenadas (x, y)
lb = [-CIRCLE_RADIUS] * (IND_SIZE * 2)
ub = [CIRCLE_RADIUS] * (IND_SIZE * 2)

# Configurando e rodando o PSO
def optimize_with_pso():
    # Lista para armazenar as soluções intermediárias
    solutions = []

    # Função de avaliação customizada que também armazena as soluções intermediárias
    def evaluate_and_store(individual):
        fitness = evaluate(individual)
        solutions.append(np.array(individual).reshape((IND_SIZE, 2)))
        return fitness

    # Executando o PSO
    best_coords, best_fitness = pso(
        evaluate_and_store, 
        lb, 
        ub, 
        ieqcons=[constraint_circle, constraint_distance],  # Adicionando as restrições
        swarmsize=50, 
        maxiter=100, 
        minstep=1e-8, 
        minfunc=1e-8, 
        debug=True
    )
    
    # Ajustando a forma do array best_coords para bidimensional
    best_coords = np.array(best_coords).reshape((IND_SIZE, 2))
    
    return best_coords, best_fitness, solutions

def main():
    best_coords, best_fitness, solutions = optimize_with_pso()
    
    print("Melhor solução:")
    print("Coordenadas X:", best_coords[:, 0])
    print("Coordenadas Y:", best_coords[:, 1])


    # Cria uma animação com as soluções intermediárias
    create_animation(solutions, IND_SIZE, CIRCLE_RADIUS)

if __name__ == "__main__":
    main()
