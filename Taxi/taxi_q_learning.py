import gymnasium as gym
import numpy as np
from taxi_env_extended import TaxiEnvExtended  # Asegúrate de que este archivo esté en el mismo directorio

# Configuración del entorno
env = TaxiEnvExtended()

# Inicialización de la tabla Q
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Configuración de hiperparámetros
alpha = 0.1
gamma = 0.6
epsilon = 0.1
num_episodes = 1000

# Entrenamiento con Q-Learning
for i in range(1, num_episodes + 1):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(Q[state])  # Explotación
        
        next_state, reward, done, truncated, _ = env.step(action)
        
        old_value = Q[state, action]
        next_max = np.max(Q[next_state])
        
        # Actualización de la tabla Q
        Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1

# Imprimir la tabla Q final después de que el entrenamiento haya terminado
print("Tabla Q final después del entrenamiento:")
print(Q)

# Evaluación del modelo entrenado
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, truncated, _ = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Resultados después de {episodes} episodios:")
print(f"Promedio de pasos por episodio: {total_epochs / episodes}")
print(f"Promedio de penalizaciones por episodio: {total_penalties / episodes}")
