import copy
import random
import matplotlib.pyplot as plt
class Machine:
    def __init__(self, id):
        self.id = id
        self.tasks = []
        self.finished_time = None
        self.current_task = None

    def copy(self):
        return copy.deepcopy(self)

    def set_tasks(self, tasks):
        for task in tasks:
            self. add_task(task)
        self.execute_tasks()

    def add_task(self, task):
        self.tasks.append(task)

    def update_finished_time(self, time):
        self.finished_time = time

    def execute_tasks(self):
        current_time = 0
        for task in self.tasks:
            task.start_times[self.id] = current_time
            task.finished_times[self.id] = task.start_times[self.id] + task.durations[self.id]
            print(f"Machine {self.id} executing Task {task.id} from time {task.start_times[self.id]} to {task.finished_times[self.id]}")
            current_time = task.finished_times[self.id]
        self.update_finished_time(current_time)


class Task:
    def __init__(self, id, durations, deadline):
        self.id = id
        self.durations = durations
        self.deadline = deadline
        self.finished_times = [None, None, None]
        self.start_times = [None, None, None]


class Solution:
    def __init__(self, machines):
        self.machines = machines
        self.execute_tasks()

    def execute_tasks(self):
        for machine in self.machines:
            machine.execute_tasks()

    def calculate_makespan(self):
        return max(max(task.finished_times) for task in self.machines[-1].tasks)

    def calculate_total_flowtime(self):
        return sum(sum(task.finished_times) for task in self.machines[-1].tasks)

    def is_dominating(self, b):
        makespan_a = self.calculate_makespan()
        makespan_b = b.calculate_makespan()
        total_flowtime_a = self.calculate_total_flowtime()
        total_flowtime_b = b.calculate_total_flowtime()

        if makespan_a <= makespan_b and total_flowtime_a <= total_flowtime_b:
            return True
        return False

    def check_constraints(self, num_tasks):
        executed_tasks = [False] * num_tasks
        machine_schedule = {}  # Słownik śledzący zajęte maszyny w danym czasie
        for machine in self.machines:
            current_time = 0
            for task in machine.tasks:
                task_id = task.id
                if executed_tasks[task_id]:
                    return False
                if current_time < task.start_times[machine.id]:
                    return False
                if task.start_times[machine.id] <= current_time < task.finished_times[machine.id]:
                    return False
                # Sprawdzanie zajętości maszyn w tym samym czasie
                for time in range(task.start_times[machine.id], task.finished_times[machine.id]):
                    if time in machine_schedule:
                        if machine.id in machine_schedule[time]:
                            return False
                    else:
                        machine_schedule[time] = [machine.id]
                current_time = task.finished_times[machine.id]
                if machine.id > 0:
                    prev_machine_task_id = self.machines[machine.id - 1].tasks[-1].id
                    if not executed_tasks[prev_machine_task_id]:
                        return False
                executed_tasks[task_id] = True
        return True


class Scheduler:
    def generate_neighbour_solution(self, solution):
        machines = [machine.copy() for machine in solution.machines]
        machine_index_1 = random.randint(0, len(machines) - 1)
        machine_index_2 = random.randint(0, len(machines) - 1)

        while machine_index_1 == machine_index_2:
            machine_index_2 = random.randint(0, len(machines) - 1)

        tasks_1 = machines[machine_index_1].tasks
        tasks_2 = machines[machine_index_2].tasks

        task_indices_1 = list(range(len(tasks_1)))
        random.shuffle(task_indices_1)

        task_indices_2 = list(range(len(tasks_2)))
        random.shuffle(task_indices_2)

        machines[machine_index_1].tasks = [tasks_1[index] for index in task_indices_1]
        machines[machine_index_2].tasks = [tasks_2[index] for index in task_indices_2]

        return Solution(machines)

    def simulated_annealing(self, max_iter, p_func, machines, tasks):
        P = []
        i = 0
        x = self.generate_initial_solution(machines, tasks)
        P.append(x)
        it = 0

        while it < max_iter:
            x0 = self.generate_neighbour_solution(x)
            if x0.check_constraints(len(tasks)) and x0.is_dominating(x):
                x = x0
                P.append(x0)
            else:
                p = p_func(it)
                if random.random() < p:
                    x = x0
                    P.append(x0)
            it += 1

        F = self.generate_pareto_front(P)
        return F, P

    def generate_initial_solution(self, machines, tasks):
        for machine in machines:
            shuffled_tasks = tasks.copy()
            random.shuffle(shuffled_tasks)  # Shuffle the tasks for the current machine
            machine.set_tasks(shuffled_tasks)  # Assign the shuffled tasks to the machine

        print("Init solution:")
        for i, machine in enumerate(machines):
            print("Machine {}\n".format(i))
            print(machine.tasks)

        return Solution(machines)

    def generate_pareto_front(self, solutions):
        F = solutions.copy()
        non_dominated_solutions = []

        for a in F:
            is_dominated = False
            for b in F:
                if a != b and a.is_dominating(b):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_solutions.append(a)

        return non_dominated_solutions


def p_func(iteration):
    return 0.995 ** iteration


tasks = [
    Task(0, [3, 5, 2], 7),
    Task(1, [2, 4, 3], 6),
    Task(2, [4, 3, 2], 5),
    Task(3, [2, 4, 7], 6),
    Task(4, [4, 6, 2], 5),
    Task(5, [2, 4, 3], 6),
    Task(6, [1, 3, 2], 5),
    Task(7, [4, 4, 3], 6),
    Task(8, [8, 3, 2], 5),
    Task(9, [2, 5, 3], 6),
]


machines = [Machine(0), Machine(1), Machine(2)]
scheduler = Scheduler()
front, solutions = scheduler.simulated_annealing(100, p_func, machines, tasks)



print("Pareto Front:")
for solution in front:
    print(solution)

print("\nPareto Set:")
for solution in solutions:
    print(solution)

def plot_gantt_chart(solution):
    tasks = solution.machines[-1].tasks
    num_machines = len(solution.machines)
    num_tasks = len(tasks)

    # Tworzenie list z wartościami czasu rozpoczęcia i zakończenia każdego zadania na każdej maszynie
    start_times = [[task.start_times[machine_id] for machine_id in range(num_machines)] for task in tasks]
    end_times = [[task.finished_times[machine_id] for machine_id in range(num_machines)] for task in tasks]

    # Tworzenie listy kolorów dla każdej maszyny
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Tworzenie wykresu Gantta
    fig, ax = plt.subplots()

    # Ustawianie tytułu i etykiet osi
    ax.set_title('Gantt Chart')
    ax.set_xlabel('Time')
    ax.set_ylabel('Task')

    # Ustawianie zakresu osi X
    max_time = max(max(end_times))
    ax.set_xlim(0, max_time)

    # Dodawanie pasków dla każdego zadania na każdej maszynie
    for task_id in range(num_tasks):
        for machine_id in range(num_machines):
            start = start_times[task_id][machine_id]
            end = end_times[task_id][machine_id]
            duration = end - start

            ax.barh(task_id, duration, left=start, height=0.8, color=colors[machine_id], alpha=0.8)

    # Ustawianie etykiet dla osi Y
    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels([f'Task {task.id}' for task in tasks])

    # Dodawanie legendy
    ax.legend([f'Machine {machine.id}' for machine in solution.machines])

    # Wyświetlanie wykresu Gantta
    plt.show()


def plot_pareto_front(solutions):
    makespans = [solution.calculate_makespan() for solution in solutions]
    flowtimes = [solution.calculate_total_flowtime() for solution in solutions]

    # Tworzenie wykresu dla Pareto Set i Pareto Front
    fig, ax = plt.subplots()
    print("Flowtimes:", flowtimes)
    print("Makespans:", makespans)

    print(f"len flowtimes: {len(flowtimes)}")
    print(f"len makespans: {len(makespans)}")
    ax.scatter(flowtimes, makespans, color='blue', label='Pareto Set')

    # Wyszukiwanie dominujących rozwiązań w celu oznaczenia frontu Pareto
    non_dominated_solutions = []
    for i in range(len(solutions)):
        is_dominated = False
        for j in range(len(solutions)):
            if i != j and solutions[i].is_dominating(solutions[j]):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated_solutions.append(solutions[i])

    # Oznaczanie rozwiązań frontu Pareto innym stylem i kolorem
    flowtimes_front = [solution.calculate_total_flowtime() for solution in non_dominated_solutions]
    makespans_front = [solution.calculate_makespan() for solution in non_dominated_solutions]
    print(f"len flowtimes: {len(flowtimes)}")
    print(f"len makespans: {len(makespans)}")
    ax.scatter(flowtimes_front, makespans_front, color='red', marker='s', label='Pareto Front')

    ax.set_xlabel('Total Flowtime')
    ax.set_ylabel('Makespan')
    ax.set_title('Pareto Front and Pareto Set')
    ax.legend()

    # Wyświetlanie wykresu
    plt.show()



plot_gantt_chart(front[0])
plot_pareto_front(solutions)