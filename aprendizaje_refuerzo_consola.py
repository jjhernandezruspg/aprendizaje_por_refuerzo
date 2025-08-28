# Librerías a utilizar
import numpy as np
import random
import json
import os
from collections import deque
import time

# Parámetros principales del laberinto
ROWS = 5
COLS = 5
START = (0, 0)
GOAL = (4, 4)

# Acciones permitidas
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Parametrización de Q-learning
ALPHA = 0.1  # Definición de la tasa de aprendizaje
GAMMA = 0.9  # Definición del factor de descuento
EPSILON = 1.0  # Exploración inicial
EPSILON_DECAY = 0.999  # Decaimiento lento para mejor exploración
MIN_EPSILON = 0.01
NUM_EPISODES = 1000
MAX_STEPS = 100  # Límite de pasos

# Laberintos prediseñados 0 > camino, 1 > pared
PREDEFINED_MAZES = [
    [  # Laberinto 1
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ],
    [  # Laberinto 2
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ],
    [  # Laberinto 3
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
]

# Generar laberinto aleatorio usando DFS
def generate_random_maze(rows, cols, start, goal):
    maze = [[1 for _ in range(cols)] for _ in range(rows)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def dfs(x, y):
        maze[x][y] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                maze[x + dx][y + dy] = 0
                dfs(nx, ny)
    
    dfs(start[0], start[1])
    maze[goal[0]][goal[1]] = 0
    return maze

# Verificar si hay un camino desde (S) > inicio hasta (G) > meta usando BFS
def has_path(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    visited = set()
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny))
    return False

# Generar laberinto con camino válido
def generate_valid_maze(rows, cols, start, goal):
    while True:
        maze = generate_random_maze(rows, cols, start, goal)
        if has_path(maze, start, goal):
            return maze

# Menú y elección de laberinto
def choose_maze():
    print("Elige un laberinto:")
    print("1. Laberinto prediseñado 1")
    print("2. Laberinto prediseñado 2")
    print("3. Laberinto prediseñado 3")
    print("4. Generar laberinto aleatorio")
    choice = input("Ingresa el número (1-4): ")
    if choice == '1':
        return PREDEFINED_MAZES[0]
    elif choice == '2':
        return PREDEFINED_MAZES[1]
    elif choice == '3':
        return PREDEFINED_MAZES[2]
    elif choice == '4':
        return generate_valid_maze(ROWS, COLS, START, GOAL)
    else:
        print("Opción inválida. Usando laberinto aleatorio por defecto.")
        return generate_valid_maze(ROWS, COLS, START, GOAL)

# Mostrar laberinto
def print_maze(maze, agent_pos):
    os.system('cls' if os.name == 'nt' else 'clear') # Primero limpiamos la consola
    for i in range(len(maze)):
        row = ''
        for j in range(len(maze[0])):
            if (i, j) == agent_pos:
                row += 'A'  # Agente
            elif (i, j) == START:
                row += 'S'  # Inicio
            elif (i, j) == GOAL:
                row += 'G'  # Meta
            elif maze[i][j] == 1:
                row += '#'  # Pared
            else:
                row += '.'  # Camino
        print(row)
    print()

# Entrenamiento del agente
def train_agent(maze, start, goal):
    q_table = np.zeros((ROWS, COLS, 4))
    epsilon = EPSILON
    episode_rewards = []
    for episode in range(NUM_EPISODES):
        state = start
        total_reward = 0
        steps = 0
        reached_goal = False
        while steps < MAX_STEPS:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[state[0], state[1]])
            dx, dy = ACTIONS[action]
            next_row = state[0] + dx
            next_col = state[1] + dy
            valid = False
            if 0 <= next_row < ROWS and 0 <= next_col < COLS and maze[next_row][next_col] == 0:
                next_state = (next_row, next_col)
                valid = True
            else:
                next_state = state
            if next_state == goal:
                reward = 100
                reached_goal = True
            elif valid:
                reward = 1 - 0.1
            else:
                reward = -10
            total_reward += reward
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            td_target = reward + GAMMA * q_table[next_state[0], next_state[1]][best_next_action]
            td_error = td_target - q_table[state[0], state[1]][action]
            q_table[state[0], state[1]][action] += ALPHA * td_error
            state = next_state
            steps += 1
            if state == goal:
                break
        episode_rewards.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "steps": steps,
            "reached_goal": reached_goal
        })
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    
    # Guardar log en JSON
    log_file = 'rewards_log.json'
    with open(log_file, 'w') as f:
        json.dump({"episodes": episode_rewards}, f, indent=4)
    print(f"Log de recompensas guardado en {log_file}")
    
    return q_table

# Simulación
def simulate(q_table, maze, start, goal):
    state = start
    path = [state]
    done = False
    print("Simulando el movimiento del agente en la consola...")
    print_maze(maze, state)
    time.sleep(0.5)
    while not done:
        action = np.argmax(q_table[state[0], state[1]])
        dx, dy = ACTIONS[action]
        next_row = state[0] + dx
        next_col = state[1] + dy
        if 0 <= next_row < ROWS and 0 <= next_col < COLS and maze[next_row][next_col] == 0:
            next_state = (next_row, next_col)
            if next_state in path[-2:]:
                q_values = q_table[state[0], state[1]].copy()
                q_values[action] = -float('inf')
                action = np.argmax(q_values)
                dx, dy = ACTIONS[action]
                next_row = state[0] + dx
                next_col = state[1] + dy
                if 0 <= next_row < ROWS and 0 <= next_col < COLS and maze[next_row][next_col] == 0:
                    state = (next_row, next_col)
                else:
                    state = state
            else:
                state = next_state
            path.append(state)
        print_maze(maze, state)
        time.sleep(0.5)
        if state == goal or len(path) > MAX_STEPS:
            done = True
    time.sleep(3)

if __name__ == "__main__":
    while True:
        maze = choose_maze()
        print("Entrenando al agente...")
        q_table = train_agent(maze, START, GOAL)
        print("Entrenamiento completado. Simulando...")
        simulate(q_table, maze, START, GOAL)
        restart = input("¿Iniciar nuevamente? (S/N): ").upper()
        if restart != 'S':
            break
