# time_callback.py

import time

class TimeCallback:
    def __init__(self, total_calls):
        self.total_calls = total_calls
        self.start_time = None
        self.iteration = 0

    def __call__(self, res):
        # Inicializa o tempo de início na primeira chamada
        if self.iteration == 0:
            self.start_time = time.time()
        
        # Incrementa o número de iterações
        self.iteration += 1

        # Calcula o tempo decorrido
        elapsed_time = time.time() - self.start_time
        avg_time_per_iter = elapsed_time / self.iteration

        # Calcula o tempo restante com base nas iterações restantes
        remaining_iters = self.total_calls - self.iteration
        remaining_time = remaining_iters * avg_time_per_iter

        # Exibe o tempo restante
        print(f"Epoch {self.iteration}/{self.total_calls} - Remaining time: {remaining_time // 60:.0f} minutes aprox.")
