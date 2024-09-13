import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from platypus import Real, NSGAII, Problem
import random
from deap import base, creator, tools, algorithms
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图
import time
import math
import logging
import json
import os
from datetime import datetime


class IEEE69BusSystem:
    def __init__(self, max_iterations_pf, population_size, max_iterations):
        self.num_buses = 69
        self.num_branches = 68
        self.base_voltage = 12.66  # kV
        self.base_power = 100  # MVA
        self.max_iterations_pf = max_iterations_pf
        self.population_size = population_size
        self.max_iterations = max_iterations

        # 线路数据：[起始节点, 终止节点, R (Ω), X (Ω), P_load (kW), Q_load (kVar)]
        self.line_data = np.array(
            [
                [1, 2, 0.0005, 0.0012, 0, 0],
                [2, 3, 0.0005, 0.0012, 0, 0],
                [3, 4, 0.0015, 0.0036, 0, 0],
                [4, 5, 0.0251, 0.0294, 0, 0],
                [5, 6, 0.366, 0.1864, 2.6, 2.2],
                [6, 7, 0.3811, 0.1941, 40.4, 30],
                [7, 8, 0.0922, 0.047, 75, 54],
                [8, 9, 0.0493, 0.0251, 30, 22],
                [9, 10, 0.819, 0.2707, 28, 19],
                [10, 11, 0.1872, 0.0619, 145, 104],
                [11, 12, 0.7114, 0.2351, 145, 104],
                [12, 13, 1.03, 0.34, 8, 5.5],
                [13, 14, 1.044, 0.345, 8, 5.5],
                [14, 15, 1.058, 0.3496, 0, 0],
                [15, 16, 0.1966, 0.065, 45.5, 30],
                [16, 17, 0.3744, 0.1238, 60, 35],
                [17, 18, 0.0047, 0.0016, 60, 35],
                [18, 19, 0.3276, 0.1083, 0, 0],
                [19, 20, 0.2106, 0.0696, 1, 0.6],
                [20, 21, 0.3416, 0.1129, 114, 81],
                [21, 22, 0.014, 0.0046, 5.3, 3.5],
                [22, 23, 0.1591, 0.0526, 0, 0],
                [23, 24, 0.3463, 0.1145, 28, 20],
                [24, 25, 0.7488, 0.2475, 0, 0],
                [25, 26, 0.3089, 0.1021, 14, 10],
                [26, 27, 0.1732, 0.0572, 14, 10],
                [3, 28, 0.0044, 0.0108, 26, 18.6],
                [28, 29, 0.064, 0.1565, 26, 18.6],
                [29, 30, 0.3978, 0.1315, 0, 0],
                [30, 31, 0.0702, 0.0232, 0, 0],
                [31, 32, 0.351, 0.116, 0, 0],
                [32, 33, 0.839, 0.2816, 14, 10],
                [33, 34, 1.708, 0.5646, 19.5, 14],
                [34, 35, 1.474, 0.4873, 6, 4],
                [3, 36, 0.0044, 0.0108, 26, 18.55],
                [36, 37, 0.064, 0.1565, 26, 18.55],
                [37, 38, 0.1053, 0.123, 0, 0],
                [38, 39, 0.0304, 0.0355, 24, 17],
                [39, 40, 0.0018, 0.0021, 24, 17],
                [40, 41, 0.7283, 0.8509, 1.2, 1],
                [41, 42, 0.31, 0.3623, 0, 0],
                [42, 43, 0.041, 0.0478, 6, 4.3],
                [43, 44, 0.0092, 0.0116, 0, 0],
                [44, 45, 0.1089, 0.1373, 39.22, 26.3],
                [45, 46, 0.0009, 0.0012, 39.22, 26.3],
                [4, 47, 0.0034, 0.0084, 0, 0],
                [47, 48, 0.0851, 0.2083, 79, 56.4],
                [48, 49, 0.2898, 0.7091, 384.7, 274.5],
                [49, 50, 0.0822, 0.2011, 384.7, 274.5],
                [8, 51, 0.0928, 0.0473, 40.5, 28.3],
                [51, 52, 0.3319, 0.1114, 3.6, 2.7],
                [9, 53, 0.174, 0.0886, 4.35, 3.5],
                [53, 54, 0.203, 0.1034, 26.4, 19],
                [54, 55, 0.2842, 0.1447, 24, 17.2],
                [55, 56, 0.2813, 0.1433, 0, 0],
                [56, 57, 1.59, 0.5337, 0, 0],
                [57, 58, 0.7837, 0.263, 0, 0],
                [58, 59, 0.3042, 0.1006, 100, 72],
                [59, 60, 0.3861, 0.1172, 0, 0],
                [60, 61, 0.5075, 0.2585, 1244, 888],
                [61, 62, 0.0974, 0.0496, 32, 23],
                [62, 63, 0.145, 0.0738, 0, 0],
                [63, 64, 0.7105, 0.3619, 227, 162],
                [64, 65, 1.041, 0.5302, 59, 42],
                [11, 66, 0.2012, 0.0611, 18, 13],
                [66, 67, 0.0047, 0.0014, 18, 13],
                [12, 68, 0.7394, 0.2444, 28, 20],
                [68, 69, 0.0047, 0.0016, 28, 20],
            ]
        )

        # DG 数据：[节点, 容量 (kW), 功率因数]
        self.dg_data = np.array([[11, 490, 0.9], [18, 390, 0.9], [61, 1690, 0.9]])

        # RPS 数据：[节点, 最大容量 (kVar)]
        self.rps_data = np.array([[21, 500], [61, 500], [64, 500]])

        # RPS 数据：[节点, 最大容量 (kVar)]
        self.rps_data = np.array(
            [[21, 500], [61, 500], [64, 500], [15, 500]]  # 添加第三个RPS
        )

        # 电压限制
        self.voltage_min = 0.95
        self.voltage_max = 1.05

        # 线路容量限制 (假设所有线路容量相同，单位：MVA)
        self.line_capacity = 10

        # 初始化节点电压
        self.voltages = np.ones(self.num_buses)

        # 添加决策变量限制
        self.Q_DG_limits = {
            "Q_DG11": (0, 800),  # kVAR
            "Q_DG15": (0, 800),  # kVAR
            "Q_DG61": (0, 1800),  # kVAR
            "Q_DG65": (0, 450),  # kVAR
            "Q_RPS61": (0, 1200),  # kVAR
        }

        # 添加控制变量
        self.control_variables = {
            "Q_DG11": 0,
            "Q_DG15": 0,
            "Q_DG61": 0,
            "Q_DG65": 0,
            "Q_RPS21": 0,
            "Q_RPS61": 0,
            "P_L": 0,
            "TV": 0,
            "TC_RPS": 0,
        }

    def get_line_parameters(self):
        return self.line_data[:, 2:4]

    def get_load_data(self):
        return self.line_data[:, 4:6]

    def get_dg_locations(self):
        return self.dg_data[:, 0].astype(int)

    def get_dg_capacities(self):
        return self.dg_data[:, 1]

    def get_rps_locations(self):
        return self.rps_data[:, 0].astype(int)

    def get_rps_capacities(self):
        return self.rps_data[:, 1]

    def save_pareto_front(self, algorithm_name, case, pareto_front):
        # 创建output文件夹（如果不存在）
        if not os.path.exists("output"):
            os.makedirs("output")

        # 创建今天日期的文件夹
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = os.path.join("output", today)
        if not os.path.exists(today_dir):
            os.makedirs(today_dir)

        # 生成文件名
        current_time = datetime.now().strftime("%H-%M-%S")
        filename = f"{algorithm_name}_{case}_{current_time}.json"
        filepath = os.path.join(today_dir, filename)

        # 将pareto_front转换为可序列化的格式
        serializable_pareto_front = []
        for solution in pareto_front:
            serializable_solution = {
                "dg_outputs": (
                    solution["dg_outputs"].tolist()
                    if isinstance(solution["dg_outputs"], np.ndarray)
                    else solution["dg_outputs"]
                ),
                "rps_outputs": (
                    solution["rps_outputs"].tolist()
                    if isinstance(solution["rps_outputs"], np.ndarray)
                    else solution["rps_outputs"]
                ),
                "objectives": solution["objectives"],
            }
            serializable_pareto_front.append(serializable_solution)

        # 保存为JSON文件
        with open(filepath, "w") as f:
            json.dump(serializable_pareto_front, f, indent=4, default=json_serialize)

        print(f"Pareto front saved to {filepath}")

    def generate_random_weights(self, num_objectives):
        weights = np.random.rand(num_objectives)
        return weights / np.sum(weights)  # 归一化权重

    def multi_objective_optimization(self, objective_function):
        def weighted_sum_objective(individual):
            objectives = objective_function(individual)
            weights = self.generate_random_weights(len(objectives))
            return (np.dot(objectives, weights),)

    def calculate_line_flows(self, voltage_magnitudes, voltage_angles):
        line_flows = []
        for line in self.line_data:
            from_bus, to_bus = int(line[0]) - 1, int(line[1]) - 1
            v_from = voltage_magnitudes[from_bus]
            v_to = voltage_magnitudes[to_bus]
            angle_diff = voltage_angles[from_bus] - voltage_angles[to_bus]
            y = 1 / (line[2] + 1j * line[3])
            s = v_from * np.conj(y * (v_from - v_to * np.exp(1j * angle_diff)))
            line_flows.append(np.abs(s))
        return line_flows

    # 调整惩罚权重
    # voltage_penalty_weight = 1000
    # line_flow_penalty_weight = 1000
    # dg_output_penalty_weight = 1000
    # rps_output_penalty_weight = 1000
    def calculate_penalty(
        self, voltage_magnitudes, voltage_angles, line_flows, dg_outputs, rps_outputs
    ):
        penalty = 0

        # 电压约束惩罚
        for v in voltage_magnitudes:
            if v < self.voltage_min:
                penalty += (self.voltage_min - v) ** 2
            elif v > self.voltage_max:
                penalty += (v - self.voltage_max) ** 2

        # 线路容量约束惩罚
        for flow in line_flows:
            if flow > self.line_capacity:
                penalty += (flow - self.line_capacity) ** 2

        # DG输出约束惩罚
        for dg, output in zip(["Q_DG11", "Q_DG15", "Q_DG61", "Q_DG65"], dg_outputs):
            if output < self.Q_DG_limits[dg][0]:
                penalty += (self.Q_DG_limits[dg][0] - output) ** 2
            elif output > self.Q_DG_limits[dg][1]:
                penalty += (output - self.Q_DG_limits[dg][1]) ** 2

        # RPS输出约束惩罚
        if rps_outputs[1] < self.Q_DG_limits["Q_RPS61"][0]:
            penalty += (self.Q_DG_limits["Q_RPS61"][0] - rps_outputs[1]) ** 2
        elif rps_outputs[1] > self.Q_DG_limits["Q_RPS61"][1]:
            penalty += (rps_outputs[1] - self.Q_DG_limits["Q_RPS61"][1]) ** 2

        return penalty * 1e2  # 使用更小的惩罚系数

    def fuzzy_logic_selection(self, objectives, algorithm):
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        logger.debug(f"Algorithm: {algorithm}")
        logger.debug(f"Type of objectives: {type(objectives)}")
        logger.debug(f"Number of objectives: {len(objectives)}")
        logger.debug(f"Sample objective: {objectives[0] if objectives else None}")

        for i, obj in enumerate(objectives[:5]):  # 只打印前5个目标，避免日志过长
            logger.debug(f"Objective {i}: {obj}")

        def membership_function(x, min_val, max_val):
            if x <= min_val:
                return 1
            elif x >= max_val:
                return 0
            else:
                return (max_val - x) / (max_val - min_val)

        objective_keys = ["power_loss", "voltage_deviation", "rps_investment"]

        # 预处理步骤：统一 Pareto 前沿格式
        try:
            unified_objectives = []
            for sol in objectives:
                if isinstance(sol, dict) and "objectives" in sol:
                    unified_objectives.append(sol)
                elif isinstance(sol, (list, tuple)) and len(sol) >= 2:
                    unified_objectives.append(
                        {
                            "dg_outputs": sol[0][:4] if len(sol[0]) >= 4 else sol[0],
                            "rps_outputs": sol[0][4:] if len(sol[0]) >= 4 else [],
                            "objectives": dict(
                                zip(objective_keys[: len(sol[1])], sol[1])
                            ),
                        }
                    )
                else:
                    logger.warning(f"Unexpected solution format: {sol}")

            membership_values = []
            for solution in unified_objectives:
                memberships = []
                for key in objective_keys:
                    if key in solution["objectives"]:
                        values = [
                            sol["objectives"][key]
                            for sol in unified_objectives
                            if key in sol["objectives"]
                        ]
                        if values:
                            min_val = min(values)
                            max_val = max(values)
                            value = solution["objectives"][key]
                            memberships.append(
                                membership_function(value, min_val, max_val)
                            )

                if memberships:
                    membership_values.append(min(memberships))

            if membership_values:
                best_index = np.argmax(membership_values)
                return unified_objectives[best_index]
            else:
                logger.warning("No valid solutions found")
                return None

        except Exception as e:
            logger.error(f"Error in fuzzy_logic_selection: {str(e)}")
            return None

    # 多目标优化
    # 首先,我们需要在IEEE69BusSystem类中添加三个新的方法,分别对应Case A、B和C:
    def optimize_case_a(self, case="A"):
        # Case A: 最小化PL和TVV
        def objective_function(x):
            dg_outputs = x[:4]
            rps_outputs = x[4:]
            power_loss, voltage_deviation, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )
            if case == "A":
                return power_loss, voltage_deviation
            elif case == "B":
                return power_loss, rps_investment
            elif case == "C":
                return power_loss, voltage_deviation, rps_investment

        return self.multi_objective_optimization(objective_function)

    def optimize_case_b(self):
        # Case B: 最小化PL和TCRPS
        def objective_function(x):
            dg_outputs = x[:4]
            rps_outputs = x[4:]
            power_loss, _, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )
            if constraints_violated:
                return float("inf"), float("inf")
            return power_loss, rps_investment

        return self.multi_objective_optimization(objective_function)

    def optimize_case_c(self):
        # Case C: 最小化PL、TVV和TCRPS
        def objective_function(x):
            dg_outputs = x[:4]
            rps_outputs = x[4:]
            power_loss, voltage_deviation, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )
            if constraints_violated:
                return float("inf"), float("inf"), float("inf")
            return power_loss, voltage_deviation, rps_investment

        return self.multi_objective_optimization(objective_function)

    # def multi_objective_optimization(self, objective_function):
    #     def weighted_sum_objective(individual):
    #         objectives = objective_function(individual)
    #         weights = self.generate_random_weights(len(objectives))
    #         return (np.dot(objectives, weights),)

    #     def normalized_weighted_sum_objective(individual):
    #         objectives = objective_function(individual)
    #         normalized_objectives = self.normalize_objectives(objectives)
    #         weights = self.assign_weights(len(objectives))
    #         return (np.dot(normalized_objectives, weights),)

    #     creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
    #     creator.create("Individual", list, fitness=creator.FitnessMulti)

    #     toolbox = base.Toolbox()
    #     toolbox.register("attr_float", random.uniform, 0, 1000)
    #     toolbox.register(
    #         "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=7
    #     )
    #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #     toolbox.register("evaluate", normalized_weighted_sum_objective)
    #     toolbox.register(
    #         "mate", tools.cxSimulatedBinaryBounded, low=0, up=1000, eta=20.0
    #     )
    #     toolbox.register(
    #         "mutate",
    #         tools.mutPolynomialBounded,
    #         low=0,
    #         up=1000,
    #         eta=20.0,
    #         indpb=1.0 / 7,
    #     )
    #     toolbox.register("select", tools.selTournament, tournsize=3)

    #     population = toolbox.population(n=100)
    #     NGEN = 50

    #     for gen in range(NGEN):
    #         offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
    #         fits = toolbox.map(toolbox.evaluate, offspring)
    #         for fit, ind in zip(fits, offspring):
    #             ind.fitness.values = fit
    #         population = toolbox.select(offspring, k=len(population))

    #     best_ind = tools.selBest(population, 1)[0]
    #     return best_ind, objective_function(best_ind)

    # max_iterations=100
    def run_power_flow(self, dg_outputs, rps_outputs, tolerance=1e-5):
        max_iterations = self.max_iterations_pf
        # 初始化
        print("run_power_flow: ", "1")
        n = self.num_buses
        # Initialize voltage magnitudes to 1.0 p.u. and angles to 0
        V = np.ones(self.num_buses, dtype=complex)
        # print("run_power_flow: ", "3")
        Y = self.build_admittance_matrix()
        print("run_power_flow: ", "4")
        S = self.calculate_power_injection(dg_outputs, rps_outputs)

        for iteration in range(max_iterations):
            # 打印迭代次数
            print("run_power_flow: ", "iteration", iteration + 1)
            # print("run_power_flow: ", "for","1")
            # 计算功率不平衡
            V_diag = csr_matrix((V, (range(n), range(n))), shape=(n, n))
            I = Y.dot(V)
            S_calc = V_diag.dot(np.conj(I))
            dS = S - S_calc

            # 检查收敛性
            if np.all(np.abs(dS) < tolerance):
                break

            # 构建雅可比矩阵
            J = self.build_jacobian(V, Y, S_calc)

            # 求解线性方程组
            dV = spsolve(J, np.concatenate([dS.real[1:], dS.imag[1:]]))

            # 更新电压
            dV_complex = dV[: n - 1] + 1j * dV[n - 1 :]
            V[1:] += dV_complex

        return np.abs(V), np.angle(V)

    def build_admittance_matrix(self):
        n = self.num_buses
        Y = np.zeros((n, n), dtype=complex)

        for line in self.line_data:
            from_bus, to_bus = int(line[0]) - 1, int(line[1]) - 1  # 转换为0-based索引
            r, x = line[2], line[3]
            y = 1 / (r + 1j * x)
            Y[from_bus, from_bus] += y
            Y[to_bus, to_bus] += y
            Y[from_bus, to_bus] -= y
            Y[to_bus, from_bus] -= y

        return csr_matrix(Y)

    def calculate_power_injection(self, dg_outputs, rps_outputs):
        S = np.zeros(self.num_buses, dtype=complex)

        # 添加负载
        for line in self.line_data:
            bus = int(line[1]) - 1  # 转换为0-based索引
            S[bus] -= (line[4] + 1j * line[5]) / 1000  # 转换为MW和MVar

        # 添加DG输出
        for dg_loc, dg_output in zip(self.get_dg_locations(), dg_outputs):
            bus = dg_loc - 1  # 转换为0-based索引
            S[bus] += dg_output / 1000  # 转换为MW

        # 添加RPS输出
        for rps_loc, rps_output in zip(self.get_rps_locations(), rps_outputs):
            bus = rps_loc - 1  # 转换为0-based索引
            S[bus] += 1j * rps_output / 1000  # 转换为MVar

        return S

    def build_jacobian(self, V, Y, S):
        n = self.num_buses
        J = np.zeros((2 * (n - 1), 2 * (n - 1)))

        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    J[i - 1, j - 1] = -S[i].imag - Y[i, i].imag * np.abs(V[i]) ** 2
                    J[i - 1, n - 1 + j - 1] = (
                        S[i].real - Y[i, i].real * np.abs(V[i]) ** 2
                    )
                    J[n - 1 + i - 1, j - 1] = (
                        S[i].real - Y[i, i].real * np.abs(V[i]) ** 2
                    )
                    J[n - 1 + i - 1, n - 1 + j - 1] = (
                        -S[i].imag - Y[i, i].imag * np.abs(V[i]) ** 2
                    )
                else:
                    J[i - 1, j - 1] = (
                        -np.abs(V[i])
                        * np.abs(V[j])
                        * (
                            Y[i, j].real * np.sin(np.angle(V[i]) - np.angle(V[j]))
                            - Y[i, j].imag * np.cos(np.angle(V[i]) - np.angle(V[j]))
                        )
                    )
                    J[i - 1, n - 1 + j - 1] = (
                        np.abs(V[i])
                        * np.abs(V[j])
                        * (
                            Y[i, j].real * np.cos(np.angle(V[i]) - np.angle(V[j]))
                            + Y[i, j].imag * np.sin(np.angle(V[i]) - np.angle(V[j]))
                        )
                    )
                    J[n - 1 + i - 1, j - 1] = (
                        np.abs(V[i])
                        * np.abs(V[j])
                        * (
                            Y[i, j].real * np.cos(np.angle(V[i]) - np.angle(V[j]))
                            + Y[i, j].imag * np.sin(np.angle(V[i]) - np.angle(V[j]))
                        )
                    )
                    J[n - 1 + i - 1, n - 1 + j - 1] = (
                        -np.abs(V[i])
                        * np.abs(V[j])
                        * (
                            Y[i, j].real * np.sin(np.angle(V[i]) - np.angle(V[j]))
                            - Y[i, j].imag * np.cos(np.angle(V[i]) - np.angle(V[j]))
                        )
                    )
        return csr_matrix(J)

    def calculate_power_loss(self, voltage_magnitudes, voltage_angles):
        logging.basicConfig(
            filename="calculate_power_loss.log",
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            encoding="utf-8",
        )
        logging.info("Starting power loss calculation")
        logging.info(f"Voltage magnitudes: {voltage_magnitudes}")
        logging.info(f"Voltage angles: {voltage_angles}")

        power_loss = 0
        penalty = 0
        for line in self.line_data:
            from_bus, to_bus = int(line[0]) - 1, int(line[1]) - 1
            r = line[2]  # 线路电阻
            x = line[3]  # 线路电抗
            if r == 0 and x == 0:  # 线路为纯虚线，不考虑
                logging.info(
                    f"Skipping line {from_bus+1}-{to_bus+1}: pure imaginary line"
                )
                continue
            v_from = voltage_magnitudes[from_bus]
            v_to = voltage_magnitudes[to_bus]
            angle_diff = voltage_angles[from_bus] - voltage_angles[to_bus]

            # 避免除以零
            z_squared = r**2 + x**2
            if z_squared < 1e-10:  # 设置一个小的阈值
                logging.warning(
                    f"Near-zero impedance for line {from_bus+1}-{to_bus+1}. Adding penalty."
                )
                penalty += 1e6  # 添加一个大的惩罚值
                continue

            # 计算电流
            i_real = (v_from * np.cos(angle_diff) - v_to) / z_squared
            i_imag = v_from * np.sin(angle_diff) / z_squared

            # 计算功率损失
            loss = r * (i_real**2 + i_imag**2)

            # 检查数值溢出
            if np.isfinite(loss):
                power_loss += loss
                logging.info(f"Line {from_bus+1}-{to_bus+1}: loss = {loss}")
            else:
                logging.warning(
                    f"Infinite or NaN loss for line {from_bus+1}-{to_bus+1}. Adding penalty."
                )
                penalty += 1e6  # 添加一个大的惩罚值

        total_loss = power_loss + penalty
        logging.info(f"Total power loss: {power_loss}")
        logging.info(f"Total penalty: {penalty}")
        logging.info(f"Final result (loss + penalty): {total_loss}")
        return total_loss

    def calculate_voltage_deviation(self, voltage_magnitudes):
        total_deviation = 0
        max_deviation = 1.0  # 设置最大偏差值
        for v in voltage_magnitudes:
            if abs(v) < 1e-6:
                deviation = max_deviation  # 避免除以零
            else:
                deviation = min(
                    abs(v - 1.0) / 1.0, max_deviation
                )  # 使用相对偏差并限制最大值
            total_deviation += deviation
        return total_deviation

    def calculate_rps_investment(self, rps_outputs):
        total_investment = 0
        for output in rps_outputs:
            if output < 0:
                investment = 0
            else:
                investment = output * 100  # 假设每kVar的RPS投资成本为100美元
            total_investment += investment
        return total_investment

    def check_voltage_constraints(self, voltage_magnitudes):
        return np.all(
            (voltage_magnitudes >= self.voltage_min)
            & (voltage_magnitudes <= self.voltage_max)
        )

    def check_line_capacity_constraints(self, voltage_magnitudes, voltage_angles):
        for line in self.line_data:
            from_bus, to_bus = int(line[0]) - 1, int(line[1]) - 1
            v_from = voltage_magnitudes[from_bus]
            v_to = voltage_magnitudes[to_bus]
            angle_diff = voltage_angles[from_bus] - voltage_angles[to_bus]
            y = 1 / (line[2] + 1j * line[3])
            s = v_from * np.conj(y * (v_from - v_to * np.exp(1j * angle_diff)))
            if np.abs(s) > self.line_capacity:
                return False
        return True

    # 更新评估函数以包含新的控制变量
    # def evaluate_solution(self, dg_outputs, rps_outputs):
    #     voltage_magnitudes, voltage_angles = self.run_power_flow(dg_outputs, rps_outputs)
    #     power_loss = self.calculate_power_loss(voltage_magnitudes, voltage_angles)
    #     voltage_deviation = self.calculate_voltage_deviation(voltage_magnitudes)
    #     rps_investment = self.calculate_rps_investment(rps_outputs)
    #
    #     constraints_violated = not (self.check_voltage_constraints(voltage_magnitudes) and
    #                                 self.check_line_capacity_constraints(voltage_magnitudes, voltage_angles))
    #
    #     # 更新控制变量
    #     self.control_variables['P_L'] = power_loss
    #     self.control_variables['TV'] = voltage_deviation
    #     self.control_variables['TC_RPS'] = rps_investment
    #
    #     return power_loss, voltage_deviation, rps_investment, constraints_violated

    def evaluate_solution(self, dg_outputs, rps_outputs):
        logging.basicConfig(
            filename="evaluation.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )

        logging.info("Starting evaluation of solution")
        logging.debug(f"nsga2调查 Evaluating solution: dg_outputs={dg_outputs}, rps_outputs={rps_outputs}")

        if not self.check_variable_constraints(dg_outputs, rps_outputs):
            logging.warning("Variable constraints violated")
            return float("inf"), float("inf"), float("inf"), True

        if not self.check_decision_variable_limits(dg_outputs, rps_outputs):
            logging.warning("Decision variable limits violated")
            return float("inf"), float("inf"), float("inf"), True

        voltage_magnitudes, voltage_angles = self.run_power_flow(
            dg_outputs, rps_outputs
        )
        logging.debug(f"Voltage magnitudes: {voltage_magnitudes}")
        logging.debug(f"Voltage angles: {voltage_angles}")

        power_loss = self.calculate_power_loss(voltage_magnitudes, voltage_angles)
        voltage_deviation = self.calculate_voltage_deviation(voltage_magnitudes)
        rps_investment = self.calculate_rps_investment(rps_outputs)

        logging.debug(f"Power loss before penalty: {power_loss}")
        logging.debug(f"Voltage deviation before penalty: {voltage_deviation}")
        logging.debug(f"RPS investment before penalty: {rps_investment}")

        line_flows = self.calculate_line_flows(voltage_magnitudes, voltage_angles)
        logging.debug(f"Line flows: {line_flows}")
        penalty = self.calculate_penalty(
            voltage_magnitudes, voltage_angles, line_flows, dg_outputs, rps_outputs
        )
        logging.debug(f"Calculated penalty: {penalty}")

        # logging.debug(f"Power loss before penalty: {power_loss}")
        # logging.debug(f"Voltage deviation before penalty: {voltage_deviation}")
        # logging.debug(f"RPS investment before penalty: {rps_investment}")
        # logging.debug(f"Calculated penalty: {penalty}")

        power_loss += penalty
        voltage_deviation += penalty
        rps_investment += penalty

        logging.debug(f"Power loss after penalty: {power_loss}")
        logging.debug(f"Voltage deviation after penalty: {voltage_deviation}")
        logging.debug(f"RPS investment after penalty: {rps_investment}")

        constraints_violated = penalty > 0
        logging.debug(f"Constraints violated: {constraints_violated}")

        self.control_variables["P_L"] = power_loss
        self.control_variables["TV"] = voltage_deviation
        self.control_variables["TC_RPS"] = rps_investment

        logging.debug(f"nsga2调查 Evaluation results: power_loss={power_loss}, voltage_deviation={voltage_deviation}, rps_investment={rps_investment}, constraints_violated={constraints_violated}")
        return power_loss, voltage_deviation, rps_investment, constraints_violated

    def check_variable_constraints(self, dg_outputs, rps_outputs):
        # 检查决策变量是否在允许范围内
        for output in dg_outputs + rps_outputs:
            if output < 0 or output > 1:
                return False
        return True

    # 添加一个新方法来检查决策变量是否在限制范围内
    def check_decision_variable_limits(self, dg_outputs, rps_outputs):
        for dg, output in zip(["Q_DG11", "Q_DG15", "Q_DG61", "Q_DG65"], dg_outputs):
            if output < self.Q_DG_limits[dg][0] or output > self.Q_DG_limits[dg][1]:
                return False
        if (
            rps_outputs[1] < self.Q_DG_limits["Q_RPS61"][0]
            or rps_outputs[1] > self.Q_DG_limits["Q_RPS61"][1]
        ):
            return False
        return True

    def optimize(self, algorithm, case):
        population_size = self.population_size
        max_iterations = self.max_iterations
        print("运行optimize: ", "algorithm: ", algorithm, "case: ", case)
        if algorithm == "POA":
            print("optimize;", "POA;")
            best_solution, pareto_front, iteration_data = (
                self.fuzzy_tuned_pelican_optimization_algorithm(
                    case, population_size, max_iterations, algorithm
                )
            )

            self.save_pareto_front("POA", case, pareto_front)
            best_solution = self.fuzzy_logic_selection(pareto_front, "FTPOA")
        elif algorithm == "FTPOA":
            print("optimize;", "运行FTPOA;")
            best_solution, pareto_front, iteration_data = (
                self.fuzzy_tuned_pelican_optimization_algorithm(
                    case, population_size, max_iterations, algorithm
                )
            )
            self.save_pareto_front("FTPOA", case, pareto_front)
            best_solution = self.fuzzy_logic_selection(pareto_front, "FTPOA")
        elif algorithm == "NSGA-II":
            print("optimize;", "运行NSGA-II;")
            num_objectives = 2 if case in ["A", "B"] else 3
            problem = Problem(7, num_objectives)
            problem.types[:] = [Real(0, 1)] * 7

            def objective_function(vars):
                dg_outputs = vars[:4]
                rps_outputs = vars[4:]
                power_loss, voltage_deviation, rps_investment, constraints_violated = (
                    self.evaluate_solution(dg_outputs, rps_outputs)
                )

                if constraints_violated:
                    return [float("inf")] * num_objectives

                if case == "A":
                    return [power_loss, voltage_deviation]
                elif case == "B":
                    return [power_loss, rps_investment]
                else:  # case C
                    return [power_loss, voltage_deviation, rps_investment]

            problem.function = objective_function
            algorithm = NSGAII(problem, population_size=self.population_size)
            algorithm.run(self.max_iterations)

            solutions = algorithm.result
            objective_keys = ["power_loss", "voltage_deviation", "rps_investment"][
                :num_objectives
            ]
            pareto_front = [
                {
                    "dg_outputs": s.variables[:4],
                    "rps_outputs": s.variables[4:],
                    "objectives": dict(zip(objective_keys, s.objectives)),
                }
                for s in solutions
            ]  # 转换为 (variables, objectives) 格式
            self.save_pareto_front("NSGA-II", case, pareto_front)
            best_solution = self.fuzzy_logic_selection(pareto_front, "NSGA-II")
        elif algorithm == "MOPSO":
            # pareto_front = self.mopso(case, population_size, max_iterations)
            # self.save_pareto_front("MOPSO", case, pareto_front)
            # best_solution = self.fuzzy_logic_selection(pareto_front, "MOPSO")
            swarm = self.mopso(case, population_size, max_iterations)
            objective_keys = ["power_loss", "voltage_deviation", "rps_investment"][
                : len(swarm[0].fitness)
            ]
            pareto_front = [
                {
                    "dg_outputs": particle.position[:4],
                    "rps_outputs": particle.position[4:],
                    "objectives": dict(zip(objective_keys, particle.fitness)),
                }
                for particle in swarm
            ]
            self.save_pareto_front("MOPSO", case, pareto_front)
            best_solution = self.fuzzy_logic_selection(pareto_front, "MOPSO")
        else:
            raise ValueError("Unsupported algorithm")

        # best_solution = self.fuzzy_logic_selection(pareto_front, num_objectives)  # 使用 objectives 作为参数
        return best_solution, pareto_front

    def nsga(self, case, population_size, max_iterations):
        # 打印nsga执行时间
        start_time = time.time()
        print("NSGA-II started at: ", start_time)

        def objective_function(vars):
            dg_outputs = vars[:4]
            rps_outputs = vars[4:]
            power_loss, voltage_deviation, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )

            if case == "A":
                return power_loss, voltage_deviation
            elif case == "B":
                return power_loss, rps_investment
            else:  # case C
                return power_loss, voltage_deviation, rps_investment

        problem = Problem(7, 2 if case in ["A", "B"] else 3)
        problem.types[:] = [Real(0, 1)] * 7
        problem.function = objective_function

        algorithm = NSGAII(problem, population_size=population_size)
        algorithm.run(max_iterations)

        solutions = algorithm.result
        pareto_front = [
            [s.objectives[i] for i in range(len(s.objectives))] for s in solutions
        ]

        # 找到最佳解
        best_solution = min(solutions, key=lambda s: sum(s.objectives)).variables

        print("NSGA-II finished at: ", time.time())
        print("Time taken: ", time.time() - start_time)
        return best_solution, pareto_front

    def visualize_network(self, pareto_front, case):
        # Pareto前沿可视化
        if case == "A":
            plt.figure(figsize=(10, 6))
            plt.scatter(
                [sol[0] for sol in pareto_front],
                [sol[1] for sol in pareto_front],
                c="b",
                marker="o",
            )
            plt.xlabel("功率损耗 (PL)")
            plt.ylabel("总电压变化 (TVV)")
            plt.title("Case A: Pareto前沿")
            plt.grid(True)
            plt.show()

        elif case == "B":
            plt.figure(figsize=(10, 6))
            plt.scatter(
                [sol[0] for sol in pareto_front],
                [sol[1] for sol in pareto_front],
                c="r",
                marker="o",
            )
            plt.xlabel("功率损耗 (PL)")
            plt.ylabel("RPS机组总投资 (TCRPS)")
            plt.title("Case B: Pareto前沿")
            plt.grid(True)
            plt.show()

        elif case == "C":
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                [sol[0] for sol in pareto_front],
                [sol[1] for sol in pareto_front],
                [sol[2] for sol in pareto_front],
                c="g",
                marker="o",
            )
            ax.set_xlabel("功率损耗 (PL)")
            ax.set_ylabel("总电压变化 (TVV)")
            ax.set_zlabel("RPS机组总投资 (TCRPS)")
            ax.set_title("Case C: Pareto前沿")
            plt.show()

        # 帕累托图
        objectives = ["PL", "TVV", "TCRPS"][: len(pareto_front[0])]
        values = np.array(pareto_front).T

        plt.figure(figsize=(12, 6))
        for i, obj in enumerate(objectives):
            plt.bar(i, np.mean(values[i]), label=obj)

        plt.xlabel("目标函数")
        plt.ylabel("平均值")
        plt.title(f"Case {case}: 帕累托图")
        plt.legend()
        plt.show()

    def generate_report(self, dg_outputs, rps_outputs):
        voltage_magnitudes, voltage_angles = self.run_power_flow(
            dg_outputs, rps_outputs
        )
        power_loss, voltage_deviation, rps_investment, constraints_violated = (
            self.evaluate_solution(dg_outputs, rps_outputs)
        )

        report = "优化结果报告\n"
        report += "=" * 20 + "\n\n"

        report += "概述：\n"
        report += f"总功率损耗 (PL): {power_loss:.4f} MW\n"
        report += f"总电压偏差 (TVV): {voltage_deviation:.4f} p.u.\n"
        report += f"RPS机组总投资 (TCRPS): {rps_investment:.2f} 美元\n"
        report += f"约束是否违反: {'是' if constraints_violated else '否'}\n\n"

        report += "DG输出：\n"
        for i, output in enumerate(dg_outputs):
            report += f"DG {i+1}: {output:.2f} kVAR\n"
        report += "\n"

        report += "RPS输出：\n"
        for i, output in enumerate(rps_outputs):
            report += f"RPS {i+1}: {output:.2f} kVAR\n"
        report += "\n"

        report += "母线电压状态：\n"
        for i, (mag, ang) in enumerate(zip(voltage_magnitudes, voltage_angles)):
            report += f"母线 {i+1}: 幅值 = {mag:.4f} p.u., 角度 = {ang:.4f} rad\n"
        report += "\n"

        report += "线路负载情况：\n"
        for line in self.line_data:
            from_bus, to_bus = int(line[0]) - 1, int(line[1]) - 1
            v_from = voltage_magnitudes[from_bus]
            v_to = voltage_magnitudes[to_bus]
            angle_diff = voltage_angles[from_bus] - voltage_angles[to_bus]
            y = 1 / (line[2] + 1j * line[3])
            s = v_from * np.conj(y * (v_from - v_to * np.exp(1j * angle_diff)))
            report += f"线路 {from_bus+1}-{to_bus+1}: 负载 = {abs(s):.4f} MVA, 容量利用率 = {abs(s)/self.line_capacity*100:.2f}%\n"

        report += "\n约束检查：\n"
        report += f"电压约束满足: {'是' if self.check_voltage_constraints(voltage_magnitudes) else '否'}\n"
        report += f"线路容量约束满足: {'是' if self.check_line_capacity_constraints(voltage_magnitudes, voltage_angles) else '否'}\n"

        return report

    # 鹈鹕优化算法（POA）：50,100
    def pelican_optimization_algorithm(self, case, population_size, max_iterations):
        population_size = 5
        max_iterations = 10
        a = 2
        b = 2
        symbiosis_rate = 0.3
        elite_size = 3
        start_time = time.time()
        print("pelican_optimization_algorithm: ", "开始优化算法")
        bounds = [(0, 1)] * 7  # 假设有7个决策变量，每个的范围都是[0, 1]

        def objective_function(individual):
            dg_outputs = individual[:4]
            rps_outputs = individual[4:]
            power_loss, voltage_deviation, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )
            if constraints_violated:
                return float("inf")
            return power_loss + voltage_deviation + rps_investment

        population = np.zeros((population_size, len(bounds)))
        for i in range(population_size):
            for j in range(len(bounds)):
                population[i, j] = random.uniform(bounds[j][0], bounds[j][1])

        fitness = np.array(
            [objective_function(individual) for individual in population]
        )
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        pareto_front = []
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        for iteration in range(max_iterations):
            w1 = 0.5 - (iteration / max_iterations) * 0.5
            w2 = 0.5 + (iteration / max_iterations) * 0.5
            R = 0.2 * (1 - iteration / max_iterations)

            for i in range(population_size):
                if random.random() < 0.5:
                    I = random.choice([1, 2])
                    Xp = np.array(
                        [
                            random.uniform(bounds[j][0], bounds[j][1])
                            for j in range(len(bounds))
                        ]
                    )
                    F_Xp = objective_function(Xp)
                    new_position = (
                        population[i]
                        - I * random.random() * (population[i] - Xp) * F_Xp
                    )
                else:
                    new_position = population[i] + R * random.random() * (
                        best_solution - population[i]
                    )

                if random.random() < 0.3:
                    u = np.random.randn(len(bounds)) * sigma
                    v = np.random.randn(len(bounds))
                    step = u / abs(v) ** (1 / beta)
                    levy_flight = 0.01 * step * (population[i] - best_solution)
                    new_position = new_position + levy_flight

                for j in range(len(bounds)):
                    new_position[j] = np.clip(
                        new_position[j], bounds[j][0], bounds[j][1]
                    )

                new_fitness = objective_function(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

            # Elite strategy
            elite_indices = np.argsort(fitness)[:elite_size]
            elite = population[elite_indices]
            worst_indices = np.argsort(fitness)[-elite_size:]
            population[worst_indices] = elite
            fitness[worst_indices] = np.array(
                [objective_function(individual) for individual in elite]
            )

            current_best = np.min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = population[np.argmin(fitness)]

            # Update Pareto front
            # 在pelican_optimization_algorithm方法中
            current_solutions = []
            for ind in population:
                objectives = self.evaluate_solution(ind[:4], ind[4:])
                current_solutions.append((ind, objectives))  # 存储所有目标值
            pareto_front = self.update_pareto_front(pareto_front + current_solutions)

        print("pelican_optimization_algorithm: ", "优化完成")
        return best_solution, best_fitness, pareto_front

    def update_pareto_front(self, solutions):
        pareto_front = []
        for solution in solutions:
            if all(
                self.dominates(solution, other)
                for other in solutions
                if other != solution
            ):
                pareto_front.append(solution)
        return pareto_front

    def dominates(self, solution1, solution2):
        return all(f1 <= f2 for f1, f2 in zip(solution1[1], solution2[1])) and any(
            f1 < f2 for f1, f2 in zip(solution1[1], solution2[1])
        )

    # 添加一个方法来修正超出限制的解
    def repair_solution(self, solution):
        dg_outputs = solution[:4]
        rps_outputs = solution[4:]

        for i, dg in enumerate(["Q_DG11", "Q_DG15", "Q_DG61", "Q_DG65"]):
            dg_outputs[i] = np.clip(
                dg_outputs[i], self.Q_DG_limits[dg][0], self.Q_DG_limits[dg][1]
            )

        rps_outputs[1] = np.clip(
            rps_outputs[1],
            self.Q_DG_limits["Q_RPS61"][0],
            self.Q_DG_limits["Q_RPS61"][1],
        )

        return np.concatenate([dg_outputs, rps_outputs])

    # max_iterations=100
    # population_size = 50
    def fuzzy_tuned_pelican_optimization_algorithm(
        self, case, population_size, max_iterations, algorithm
    ):
        # 配置日志记录
        logging.basicConfig(
            filename="ftpoa.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )

        # Initialize population
        start_time = time.time()
        print("fuzzy_tuned_pelican_optimization_algorithm: ", "开始优化算法")
        bounds = [(0, 1)] * 7  # 假设有7个决策变量，每个的范围都是[0, 1]
        population = np.zeros((population_size, len(bounds)))
        for i in range(population_size):
            for j in range(len(bounds)):
                population[i, j] = random.uniform(bounds[j][0], bounds[j][1])

        # Initialize best solution and Pareto front
        best_solution = None
        pareto_front = []

        # Initialize iteration data
        iteration_data = []

        for iteration in range(max_iterations):
            iteration_start_time = time.time()
            logging.info(f"开始第 {iteration + 1} 次迭代...")
            print(f"正在执行第 {iteration + 1} 次迭代...")

            # Evaluate all solutions
            # objectives = [self.evaluate_solution(sol[:4], sol[4:]) for sol in population]
            objectives = []
            for i, sol in enumerate(population):
                objectives.append(self.evaluate_solution(sol[:4], sol[4:]))
                print(f"正在评估第 {i + 1}/{len(population)} 个解决方案")
                logging.debug(
                    f"评估第 {i + 1}/{len(population)} 个解: {sol}, 目标值: {objectives[-1]}"
                )

            # 更新帕累托前沿
            logging.debug(f"迭代 {iteration + 1} 前的帕累托前沿: {pareto_front}")
            pareto_front = self.update_pareto_front_1(
                population, objectives, pareto_front, algorithm
            )
            logging.debug(f"迭代 {iteration + 1} 后的帕累托前沿: {pareto_front}")

            # 格式统一化
            unified_pareto_front = self.unify_pareto_front_format(
                pareto_front, algorithm
            )
            logging.debug(f"统一化后的帕累托前沿: {unified_pareto_front}")

            # 检查帕累托前沿是否为空
            if not unified_pareto_front:
                logging.warning(f"迭代 {iteration + 1} 的帕累托前沿为空！")

            # 使用模糊逻辑调整权重
            for i, sol in enumerate(population):
                weights = self.apply_fuzzy_logic_for_weights(
                    objectives[i], unified_pareto_front
                )
                logging.debug(f"解 {sol} 的模糊权重: {weights}")

                # Sort population based on weighted sum of objectives
                population = self.sort_population(population, objectives, weights)

            # Select elite solutions
            elite_size = int(population_size * 0.1)
            elite_solutions = population[:elite_size]

            # Generate new solutions
            new_population = []
            for i in range(population_size):
                if i < elite_size:
                    new_solution = elite_solutions[i]
                else:
                    # Apply mutual symbiosis strategy
                    partner = random.choice(population)
                    new_solution = self.mutual_symbiosis(population[i], partner)

                # Apply local search to improve solution
                new_solution = self.local_search(new_solution)
                new_population.append(new_solution)

            population = new_population

            # Update best solution
            current_best = min(
                population, key=lambda x: sum(self.evaluate_solution(x[:4], x[4:])[:3])
            )
            if best_solution is None or sum(
                self.evaluate_solution(current_best[:4], current_best[4:])[:3]
            ) < sum(self.evaluate_solution(best_solution[:4], best_solution[4:])[:3]):
                best_solution = current_best

            logging.debug(f"迭代 {iteration + 1} 的最佳解: {best_solution}")

            for i, solution in enumerate(pareto_front):
                if isinstance(solution, dict):
                    # 如果 solution 是字典，保持原样
                    pareto_front[i] = solution
                elif isinstance(solution, (list, tuple)) and len(solution) >= 3:
                    # 如果 solution 是列表或元组，按原来的方式处理
                    pareto_front[i] = (
                        (
                            solution[0].tolist()
                            if hasattr(solution[0], "tolist")
                            else solution[0]
                        ),
                        (
                            solution[1].tolist()
                            if hasattr(solution[1], "tolist")
                            else solution[1]
                        ),
                        solution[2],
                    )
                else:
                    # 如果是其他格式，记录日志并跳过
                    logging.warning(
                        f"Unexpected solution format at index {i}: {solution}"
                    )

            # Store iteration data
            iteration_data.append(
                {
                    "iteration": iteration,
                    "best_solution": best_solution,
                    "pareto_front": pareto_front.copy(),
                }
            )

            # 返回包含字典的帕累托前沿
            # 返回包含字典的帕累托前沿
            pareto_front = [
                {
                    "dg_outputs": (
                        sol["variables"][:4] if isinstance(sol, dict) else sol[:4]
                    ),
                    "rps_outputs": (
                        sol["variables"][4:] if isinstance(sol, dict) else sol[4:]
                    ),
                    "objectives": {
                        "power_loss": (
                            sol["objectives"][0] if isinstance(sol, dict) else obj[0]
                        ),
                        "voltage_deviation": (
                            sol["objectives"][1] if isinstance(sol, dict) else obj[1]
                        ),
                        "rps_investment": (
                            sol["objectives"][2]
                            if isinstance(sol, dict) and len(sol["objectives"]) > 2
                            else (obj[2] if len(obj) > 2 else None)
                        ),
                    },
                }
                for sol, obj in zip(pareto_front, objectives)
            ]

            iteration_end_time = time.time()
            logging.info(
                f"第 {iteration + 1} 次迭代完成，耗时: {iteration_end_time - iteration_start_time:.2f} 秒"
            )
            # return best_solution, pareto_front, iteration_data
            return best_solution, pareto_front, iteration_data

    def unify_pareto_front_format(self, pareto_front, algorithm):
        unified_front = []

        for solution in pareto_front:
            unified_solution = {}

            if algorithm in ["FTPOA", "POA"]:
                unified_solution = (
                    solution.copy()
                    if isinstance(solution, dict)
                    else {
                        "dg_outputs": solution[0],
                        "rps_outputs": solution[1],
                        "objectives": solution[2],
                    }
                )
            elif algorithm == "NSGA-II":
                if isinstance(solution, dict):
                    unified_solution = solution.copy()
                elif isinstance(solution, tuple):
                    unified_solution = {
                        "dg_outputs": list(solution[0][:4]),
                        "rps_outputs": list(solution[0][4:]),
                        "objectives": {
                            "power_loss": solution[1][0],
                            "voltage_deviation": solution[1][1],
                            "rps_investment": (
                                solution[1][2] if len(solution[1]) > 2 else None
                            ),
                        },
                    }
                elif hasattr(solution, "fitness"):
                    unified_solution = {
                        "dg_outputs": list(solution[:4]),
                        "rps_outputs": list(solution[4:]),
                        "objectives": {
                            "power_loss": solution.fitness.values[0],
                            "voltage_deviation": solution.fitness.values[1],
                            "rps_investment": (
                                solution.fitness.values[2]
                                if len(solution.fitness.values) > 2
                                else None
                            ),
                        },
                    }
            elif algorithm == "MOPSO":
                if isinstance(solution, dict):
                    unified_solution = solution.copy()
                elif isinstance(solution, tuple):
                    unified_solution = {
                        "dg_outputs": list(solution[0][:4]),
                        "rps_outputs": list(solution[0][4:]),
                        "objectives": {
                            "power_loss": solution[1][0],
                            "voltage_deviation": solution[1][1],
                            "rps_investment": (
                                solution[1][2] if len(solution[1]) > 2 else None
                            ),
                        },
                    }
                elif hasattr(solution, "position") and hasattr(solution, "fitness"):
                    unified_solution = {
                        "dg_outputs": list(solution.position[:4]),
                        "rps_outputs": list(solution.position[4:]),
                        "objectives": {
                            "power_loss": solution.fitness[0],
                            "voltage_deviation": solution.fitness[1],
                            "rps_investment": (
                                solution.fitness[2]
                                if len(solution.fitness) > 2
                                else None
                            ),
                        },
                    }

            if unified_solution:
                unified_front.append(unified_solution)
            else:
                print(f"Unexpected solution format for {algorithm}: {solution}")

        return unified_front

    def fuzzy_weight_adjustment(self, current_iteration, max_iterations):
        # Implement fuzzy logic to adjust weights based on iteration progress
        progress = current_iteration / max_iterations
        if progress < 0.3:
            return [0.6, 0.3, 0.1]  # Focus more on exploration
        elif progress < 0.7:
            return [0.4, 0.4, 0.2]  # Balance exploration and exploitation
        else:
            return [0.2, 0.3, 0.5]  # Focus more on exploitation

    # def apply_fuzzy_logic_for_weights(self, objectives, pareto_front):
    #     """
    #     使用模糊逻辑调整权重因子。
    #     :param objectives: 当前解的目标函数值
    #     :param pareto_front: 当前的帕累托前沿
    #     :return: 调整后的权重因子
    #     """
    #     if not pareto_front:
    #         return [1.0 / len(objectives)] * len(objectives)

    #     def triangular_membership_function(value, min_val, peak_val, max_val):
    #         # 三角��属函数
    #         return np.where(
    #             value <= min_val,
    #             0.0,
    #             np.where(
    #                 value >= max_val,
    #                 0.0,
    #                 (value - min_val) / (peak_val - min_val)
    #                 - (value - peak_val) / (max_val - peak_val),
    #             ),
    #         )

    #     # 归一化隶属度以获得权重因子
    #     num_objectives = len(objectives)
    #     memberships = []

    #     for i in range(num_objectives):
    #         min_val = min(sol[1][i] for sol in pareto_front)
    #         max_val = max(sol[1][i] for sol in pareto_front)
    #         memberships.append(
    #             self.membership_function(objectives[i], min_val, max_val)
    #         )

    #     total_membership = sum(memberships)
    #     weights = [m / total_membership for m in memberships]

    #     return weights
    def apply_fuzzy_logic_for_weights(self, objectives, pareto_front):
        if not pareto_front:
            return [1.0 / len(objectives)] * len(objectives)

        num_objectives = len(objectives)
        memberships = []

        for i in range(num_objectives):
            values = []
            for sol in pareto_front:
                if isinstance(sol, dict) and "objectives" in sol:
                    if isinstance(sol["objectives"], dict):
                        values.append(list(sol["objectives"].values())[i])
                    elif isinstance(sol["objectives"], (list, tuple)):
                        values.append(sol["objectives"][i])
                elif isinstance(sol, tuple):
                    # 假设元组的第二个元素是目标函数值
                    values.append(sol[1][i])
                else:
                    print(f"Unexpected solution format: {sol}")

            if values:
                min_val = min(values)
                max_val = max(values)
                memberships.append(
                    self.membership_function(objectives[i], min_val, max_val)
                )

        total_membership = sum(memberships)
        if total_membership == 0:
            return [1.0 / len(objectives)] * len(objectives)

        weights = [m / total_membership for m in memberships]
        return weights

    def membership_function(self, x, min_val, max_val):
        x = np.array(x, dtype=float)
        min_val = np.array(min_val, dtype=float)
        max_val = np.array(max_val, dtype=float)

        # 避免除以零
        denominator = np.maximum(max_val - min_val, 1e-10)

        result = np.where(
            np.less_equal(x, min_val),
            1.0,
            np.where(np.greater_equal(x, max_val), 0.0, (max_val - x) / denominator),
        )

        return np.mean(result)  # 使用平均值处理多维输入

    # def membership_function(self, x, min_val, max_val):
    #     if x <= min_val:
    #         return 1
    #     elif x >= max_val:
    #         return 0
    #     else:
    #         return (max_val - x) / (max_val - min_val)

    def mutual_symbiosis(self, solution1, solution2):
        # Implement mutual symbiosis strategy
        alpha = random.uniform(0, 1)
        new_solution = [
            alpha * s1 + (1 - alpha) * s2 for s1, s2 in zip(solution1, solution2)
        ]
        return new_solution

    def local_search(self, solution):
        # Implement a simple local search to improve the solution
        improved_solution = solution.copy()
        for i in range(len(solution)):
            delta = random.uniform(-0.1, 0.1)
            improved_solution[i] = max(0, min(1, improved_solution[i] + delta))
        return improved_solution

    # def update_pareto_front_1(self, population, objectives, pareto_front, algorithm):
    #     new_pareto_front = []
    #     for sol, obj in zip(population, objectives):
    #         is_dominated = False
    #         for p_sol in pareto_front:
    #             if algorithm in ["FTPOA", "POA"]:
    #                 p_obj = p_sol["objectives"]
    #             elif algorithm == "NSGA-II":
    #                 p_obj = (
    #                     p_sol.fitness.values
    #                     if hasattr(p_sol, "fitness")
    #                     else p_sol["objectives"]
    #                 )
    #             elif algorithm == "MOPSO":
    #                 p_obj = (
    #                     p_sol.fitness
    #                     if hasattr(p_sol, "fitness")
    #                     else p_sol["objectives"]
    #                 )
    #             else:
    #                 raise ValueError(f"Unknown algorithm: {algorithm}")

    #             if self.dominates(p_obj, obj):
    #                 is_dominated = True
    #                 break
    #             elif not self.dominates_1(obj, p_obj):
    #                 new_pareto_front.append(p_sol)

    #         if not is_dominated:
    #             if algorithm in ["FTPOA", "POA"]:
    #                 new_pareto_front.append({"variables": sol, "objectives": obj})
    #             elif algorithm == "NSGA-II":
    #                 new_sol = creator.Individual(sol)
    #                 new_sol.fitness.values = obj
    #                 new_pareto_front.append(new_sol)
    #             elif algorithm == "MOPSO":
    #                 new_sol = Particle(len(sol))
    #                 new_sol.position = sol
    #                 new_sol.fitness = obj
    #                 new_pareto_front.append(new_sol)

    #     # return new_pareto_front
    #     return [[sol[:4], sol[4:], obj] for sol, obj in pareto_front]
    def update_pareto_front_1(self, population, objectives, pareto_front, algorithm):
        new_pareto_front = []
        for sol, obj in zip(population, objectives):
            is_dominated = False
            for p_sol in pareto_front:
                p_obj = self.get_objectives(p_sol, algorithm)
                if self.dominates(p_obj, obj):
                    is_dominated = True
                    break

            if not is_dominated:
                new_solution = self.create_solution(sol, obj, algorithm)
                new_pareto_front.append(new_solution)

        # 移除被新解支配的旧解
        final_pareto_front = []
        for p_sol in pareto_front:
            p_obj = self.get_objectives(p_sol, algorithm)
            if not any(
                self.dominates(self.get_objectives(new_sol, algorithm), p_obj)
                for new_sol in new_pareto_front
            ):
                final_pareto_front.append(p_sol)

        final_pareto_front.extend(new_pareto_front)
        return final_pareto_front

    def get_objectives(self, solution, algorithm):
        if algorithm in ["FTPOA", "POA"]:
            return solution["objectives"]
        elif algorithm in ["NSGA-II", "MOPSO"]:
            return (
                solution.fitness.values
                if hasattr(solution, "fitness")
                else solution["objectives"]
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def create_solution(self, sol, obj, algorithm):
        if algorithm in ["FTPOA", "POA"]:
            return {"variables": sol, "objectives": obj}
        elif algorithm == "NSGA-II":
            new_sol = creator.Individual(sol)
            new_sol.fitness.values = obj
            return {"variables": new_sol, "objectives": obj}
        elif algorithm == "MOPSO":
            new_sol = Particle(len(sol))
            new_sol.position = sol
            new_sol.fitness = obj
            return {"variables": new_sol.position, "objectives": new_sol.fitness}
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # def dominates(self, obj1, obj2):
    #     return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    def dominates_1(self, obj1, obj2):
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(
            o1 < o2 for o1, o2 in zip(obj1, obj2)
        )

    def sort_population(self, population, objectives, weights):
        weighted_objectives = [
            sum(w * o for w, o in zip(weights, obj)) for obj in objectives
        ]
        return [
            x
            for _, x in sorted(
                zip(weighted_objectives, population), key=lambda item: item[0]
            )
        ]  # Use item[0] directly

    # NSGA-II算法：
    #     def nsga_ii(self, population_size=100, num_generations=50):
    def nsga_ii(self, case, population_size, num_generations):
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        def evaluate(individual):
            dg_outputs = individual[:4]
            rps_outputs = individual[4:]
            power_loss, voltage_deviation, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )
            return power_loss, voltage_deviation, rps_investment

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=7
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
        toolbox.register(
            "mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0 / 7
        )
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=population_size)
        algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=0.7,
            mutpb=0.2,
            ngen=num_generations,
            stats=None,
            halloffame=None,
            verbose=False,
        )

        return population

    # MOPSO算法
    def mopso(self, case, swarm_size, num_iterations):
        # swarm_size = 10
        # num_iterations = 5

        # 打印mopso执行时间
        start_time = time.time()
        print("MOPSO started at: ", start_time)

        class Particle:
            def __init__(self, dimensions):
                self.position = np.random.rand(dimensions)
                self.velocity = np.zeros(dimensions)
                self.best_position = self.position.copy()
                self.fitness = None
                self.best_fitness = None

        def evaluate(position):
            dg_outputs = position[:4]
            rps_outputs = position[4:]
            power_loss, voltage_deviation, rps_investment, constraints_violated = (
                self.evaluate_solution(dg_outputs, rps_outputs)
            )
            if constraints_violated:
                return (float("inf"),) * 3
            if case == "A":
                return power_loss, voltage_deviation
            elif case == "B":
                return power_loss, rps_investment
            elif case == "C":
                return power_loss, voltage_deviation, rps_investment

        swarm = [Particle(7) for _ in range(swarm_size)]
        global_best = None
        global_best_fitness = float("inf")

        for _ in range(num_iterations):
            for particle in swarm:
                particle.fitness = evaluate(particle.position)

                if particle.best_fitness is None or all(
                    f1 <= f2 for f1, f2 in zip(particle.fitness, particle.best_fitness)
                ):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness

                if global_best is None or all(
                    f1 <= f2 for f1, f2 in zip(particle.fitness, global_best_fitness)
                ):
                    global_best = particle.position.copy()
                    global_best_fitness = particle.fitness

            for particle in swarm:
                w = 0.5
                c1 = 2.0
                c2 = 2.0
                r1, r2 = np.random.rand(2)

                particle.velocity = (
                    w * particle.velocity
                    + c1 * r1 * (particle.best_position - particle.position)
                    + c2 * r2 * (global_best - particle.position)
                )

                particle.position = np.clip(particle.position + particle.velocity, 0, 1)

        end_time = time.time()
        print("MOPSO finished at: ", end_time)
        print("Time taken: ", end_time - start_time)

        return swarm

    def visualize_pareto_front(self, pareto_fronts, labels, case):
        print(f"Visualizing Pareto front for case {case}")
        print(f"Number of fronts: {len(pareto_fronts)}")

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体为黑体
        plt.rcParams["axes.unicode_minus"] = (
            False  # 解决保存图像是负号'-'显示为方块的问题
        )

        if case == "A" or case == "B":
            for front, label in zip(pareto_fronts, labels):
                print(f"Processing front for {label}")
                print(f"Front type: {type(front)}")
                print(f"Front length: {len(front)}")

                x_values = []
                y_values = []

                for sol in front:
                    print(f"Solution type: {type(sol)}")
                    if label == "FTPOA" or label == "POA":
                        if isinstance(sol, dict) and "objectives" in sol:
                            objectives = sol["objectives"]
                            x_values.append(objectives.get("power_loss", float("inf")))
                            y_values.append(
                                objectives.get(
                                    (
                                        "voltage_deviation"
                                        if case == "A"
                                        else "rps_investment"
                                    ),
                                    float("inf"),
                                )
                            )
                    elif label == "NSGA-II":
                        if isinstance(sol, dict) and "objectives" in sol:
                            objectives = sol["objectives"]
                            x_values.append(objectives.get("power_loss", float("inf")))
                            y_values.append(
                                objectives.get(
                                    (
                                        "voltage_deviation"
                                        if case == "A"
                                        else "rps_investment"
                                    ),
                                    float("inf"),
                                )
                            )
                    elif label == "MOPSO":
                        if hasattr(sol, "fitness"):
                            x_values.append(sol.fitness[0])
                            y_values.append(sol.fitness[1])
                    else:
                        print(f"Unexpected solution format for {label}: {sol}")

                print(f"X values: {x_values}")
                print(f"Y values: {y_values}")

                # 过滤掉无穷大的值
                valid_points = [
                    (x, y)
                    for x, y in zip(x_values, y_values)
                    if not (math.isinf(x) or math.isinf(y))
                ]
                if valid_points:
                    x_values, y_values = zip(*valid_points)
                    plt.scatter(x_values, y_values, label=label)
                else:
                    print(f"No valid data points for {label}")

            plt.xlabel("功率损耗 (PL)")
            plt.ylabel("总电压变化 (TVV)" if case == "A" else "RPS机组总投资 (TCRPS)")
        elif case == "C":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            for front, label in zip(pareto_fronts, labels):
                print(f"Processing front for {label}")
                print(f"Front type: {type(front)}")
                print(f"Front length: {len(front)}")

                x_values = []
                y_values = []
                z_values = []

                for sol in front:
                    print(f"Solution type: {type(sol)}")
                    if label == "FTPOA" or label == "POA":
                        if isinstance(sol, dict) and "objectives" in sol:
                            objectives = sol["objectives"]
                            x_values.append(objectives.get("power_loss", float("inf")))
                            y_values.append(
                                objectives.get("voltage_deviation", float("inf"))
                            )
                            z_values.append(
                                objectives.get("rps_investment", float("inf"))
                            )
                    elif label == "NSGA-II":
                        if isinstance(sol, dict) and "objectives" in sol:
                            objectives = sol["objectives"]
                            x_values.append(objectives.get("power_loss", float("inf")))
                            y_values.append(
                                objectives.get("voltage_deviation", float("inf"))
                            )
                            z_values.append(
                                objectives.get("rps_investment", float("inf"))
                            )
                    elif label == "MOPSO":
                        if hasattr(sol, "fitness"):
                            x_values.append(sol.fitness[0])
                            y_values.append(sol.fitness[1])
                            z_values.append(sol.fitness[2])
                    else:
                        print(f"Unexpected solution format for {label}: {sol}")

                print(f"X values: {x_values}")
                print(f"Y values: {y_values}")
                print(f"Z values: {z_values}")

                # 过滤掉无穷大的值
                valid_points = [
                    (x, y, z)
                    for x, y, z in zip(x_values, y_values, z_values)
                    if not (math.isinf(x) or math.isinf(y) or math.isinf(z))
                ]
                if valid_points:
                    x_values, y_values, z_values = zip(*valid_points)
                    ax.scatter(x_values, y_values, z_values, label=label)
                else:
                    print(f"No valid data points for {label}")

            ax.set_xlabel("功率损耗 (PL)")
            ax.set_ylabel("总电压变化 (TVV)")
            ax.set_zlabel("RPS机组总投资 (TCRPS)")

        plt.title(f"Case {case}: Pareto前沿比较")
        plt.legend()
        plt.grid(True)
        plt.show()


    def optimize_NSGA(self, algorithm, case):
        # Set up logging
        logging.basicConfig(
            filename="nsga2调查.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )

        if algorithm == "NSGA-II":
            logging.info("开始NSGA-II优化")
            # Determine number of objectives based on the case
            num_objectives = 2 if case in ["A", "B"] else 3
            # Set up the problem with 7 variables and the determined number of objectives
            problem = Problem(7, num_objectives)
            problem.types[:] = [Real(0, 1)] * 7

            def objective_function(vars):
                # Split variables into DG and RPS outputs
                dg_outputs = vars[:4]
                rps_outputs = vars[4:]

                logging.debug(
                    f"Evaluating solution: dg_outputs={dg_outputs}, rps_outputs={rps_outputs}"
                )

                try:
                    # Evaluate the solution
                    power_loss, voltage_deviation, rps_investment, constraints_violated = (
                        self.evaluate_solution(dg_outputs, rps_outputs)
                    )

                    logging.debug(
                        f"Evaluation results: power_loss={power_loss}, voltage_deviation={voltage_deviation}, rps_investment={rps_investment}, constraints_violated={constraints_violated}"
                    )

                    # If constraints are violated, return infinity
                    if constraints_violated:
                        logging.warning("Constraints violated, returning infinity")
                        return [float("inf")] * num_objectives

                    # Return objectives based on the case
                    if case == "A":
                        return [power_loss, voltage_deviation]
                    elif case == "B":
                        return [power_loss, rps_investment]
                    else:  # case C
                        return [power_loss, voltage_deviation, rps_investment]
                except Exception as e:
                    logging.error(f"Error in objective function: {str(e)}")
                    logging.error(f"Input variables: {vars}")
                    return [float("inf")] * num_objectives

            # Assign the objective function to the problem
            problem.function = objective_function

            # Initialize the NSGA-II algorithm
            algorithm = NSGAII(problem, population_size=self.population_size)

            logging.info(
                f"NSGA-II参数: population_size={self.population_size}, max_iterations={self.max_iterations}"
            )

            # Main optimization loop
            for i in range(self.max_iterations):
                logging.info(f"开始迭代 {i+1}/{self.max_iterations}")
                algorithm.step()

                # Record the best solution of each iteration
                best_solution = min(algorithm.result, key=lambda x: x.objectives[0])
                logging.info(f"迭代 {i+1} 最佳解: {best_solution.variables}")
                logging.info(f"迭代 {i+1} 目标值: {best_solution.objectives}")

                # Check for infinity values
                if any(
                    math.isinf(obj) or math.isnan(obj) for obj in best_solution.objectives
                ):
                    logging.warning(f"迭代 {i+1} 发现无穷大或NaN的目标值")
                    logging.debug(f"导致无穷大或NaN的输入值: {best_solution.variables}")

                    # Try to recalculate the objective function and log intermediate results
                    try:
                        recalculated_objectives = objective_function(
                            best_solution.variables
                        )
                        logging.debug(f"重新计算的目标值: {recalculated_objectives}")
                    except Exception as e:
                        logging.error(f"重新计算目标函数时发生错误: {str(e)}")

            logging.info("NSGA-II优化完成")

            # After the optimization loop
            inf_solutions = [
                sol
                for sol in algorithm.result
                if any(math.isinf(obj) or math.isnan(obj) for obj in sol.objectives)
            ]
            logging.info(f"Number of solutions with infinity or NaN: {len(inf_solutions)}")
            if inf_solutions:
                logging.info("Sample of problematic solutions:")
                for sol in inf_solutions[:5]:  # Log the first 5 problematic solutions
                    logging.info(f"Variables: {sol.variables}")
                    logging.info(f"Objectives: {sol.objectives}")

            # Convert results to the desired format
            pareto_front = []
            for solution in algorithm.result:
                pareto_front.append(
                    {
                        "dg_outputs": solution.variables[:4],
                        "rps_outputs": solution.variables[4:],
                        "objectives": {
                            "power_loss": solution.objectives[0],
                            "voltage_deviation": solution.objectives[1],
                            "rps_investment": (
                                solution.objectives[2]
                                if len(solution.objectives) > 2
                                else None
                            ),
                        },
                    }
                )

            logging.info(f"Pareto前沿大小: {len(pareto_front)}")
            logging.debug(f"Pareto前沿详情: {pareto_front}")

            return pareto_front


def json_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# 最佳50-100
population_size = 10
max_iterations = 10
# 最佳10-20
max_iterations_pf = 10
system = IEEE69BusSystem(max_iterations_pf, population_size, max_iterations)
case = "A"

setp = 2
nsga2_pareto_front_a = system.optimize_NSGA("NSGA-II", case)


# 1
# --------------------------------------------------------------------------
# ftpoa_best_a, ftpoa_pareto_front_a = system.optimize("FTPOA", case)
# nsga2_pareto_front_a = system.optimize("NSGA-II", case)
# mopso_pareto_front_a = system.optimize("MOPSO", case)
# poa_best_a, poa_pareto_front_a = system.optimize("POA", case)


# def compare_algorithms(algorithms_results, system):
#     # 创建输出目录
#     today = datetime.now().strftime("%Y-%m-%d")
#     output_dir = os.path.join("output", "compare", today)
#     os.makedirs(output_dir, exist_ok=True)

#     log_file_path = os.path.join(output_dir, "comparison_log.txt")
#     with open(log_file_path, "w") as log_file:
#         for alg, result in algorithms_results.items():
#             print(f"Processing algorithm: {alg}")
#             log_file.write(f"Algorithm: {alg}\n")
#             log_file.write(f"Result: {result}\n")
#             best_compromise = system.fuzzy_logic_selection(result["pareto_front"], alg)
#             log_file.write(f"Best compromise solution: {best_compromise}\n")
#             log_file.write("\n" + "=" * 50 + "\n\n")

#     print(f"Comparison log saved to: {log_file_path}")

#     # 图形比较
#     pareto_fronts = [result["pareto_front"] for result in algorithms_results.values()]
#     labels = list(algorithms_results.keys())

#     plt.figure(figsize=(12, 8))
#     system.visualize_pareto_front(pareto_fronts, labels, case="A")

#     # 保存图片
#     img_file_path = os.path.join(output_dir, "pareto_front_comparison.png")
#     plt.savefig(img_file_path)
#     plt.close()

#     print(f"Pareto front comparison image saved to: {img_file_path}")


# algorithms_results = {
#     "FTPOA": {"pareto_front": ftpoa_pareto_front_a},
#     "NSGA-II": {"pareto_front": nsga2_pareto_front_a},
#     "MOPSO": {"pareto_front": mopso_pareto_front_a},
#     "POA": {"pareto_front": poa_pareto_front_a},
# }

# compare_algorithms(algorithms_results, system)
# -----------------------------------------------------------------------------------
