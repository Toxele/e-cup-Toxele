# python baseline.py --orders ml_ozon_logistic_dataSetOrders.json --couriers ml_ozon_logistic_dataSetCouriers.json --durations_json ml_ozon_logistic_dataDurations.json --durations_db durations.sqlite --output solution.json

# !/usr/bin/env python3
import argparse
import json
import sqlite3
import time
import random
import struct
import multiprocessing
from pathlib import Path
from functools import lru_cache
from collections import defaultdict, Counter
import ijson
from tqdm import tqdm
import psutil

# Импортируем OR-Tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- Глобальные константы ---
WAREHOUSE_ID_INTERNAL, WAREHOUSE_ID_OUTPUT = 0, 1
MAX_WORK_TIME = 12 * 3600
PENALTY = 3000
COURIERS_TO_USE = 280

# --- Константы для предобработки ---
CHUNK_SIZE = 250_000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# --- Константы для решателя ---
LNS_ITERATIONS = 2 # Крутим
LNS_TIME_LIMIT_PER_ITERATION_SEC = 110 #120 # Крутим
LNS_DESTROY_PERCENTAGE = 0.20 # Крутим
BALANCING_PENALTY_COEFFICIENT = 10 # Крутим
TOTAL_VRP_TIME_LIMIT_SEC = 55 * 60 # 55 * 60 = 55 минут

# --- НОВАЯ КОНСТАНТА ДЛЯ КЛАСТЕРИЗАЦИИ ---
NUM_CLUSTERS = 15


# --- БЛОК ПРЕДОБРАБОТКИ ДАННЫХ (без изменений) ---
def _process_chunk(chunk):
    counts = Counter(rec["from"] for rec in chunk)
    batch = [(rec["from"], rec["to"], rec["dist"]) for rec in chunk]
    return counts, batch


def build_caches(durations_json_path, db_path, bin_path, index_path):
    print("--- Starting Optimized Cache Building ---")
    print(f"Using {NUM_WORKERS} worker processes.")
    total_size = durations_json_path.stat().st_size
    print("\n1/3: Parsing JSON and building SQLite DB in parallel...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-2097152; DROP TABLE IF EXISTS dists;
        CREATE TABLE dists (f INTEGER, t INTEGER, d INTEGER);
    """)
    conn.commit()
    global_counts = Counter()
    with durations_json_path.open("rb") as f, tqdm(total=total_size, unit="B", unit_scale=True,
                                                   desc="Parsing & DB Write") as pbar:
        pool = multiprocessing.Pool(NUM_WORKERS)
        parser = ijson.items(f, "item", buf_size=64 * 1024)
        while True:
            chunk = []
            try:
                for _ in range(CHUNK_SIZE):
                    rec = next(parser)
                    chunk.append({"from": int(rec["from"]), "to": int(rec["to"]), "dist": int(rec["dist"])})
            except StopIteration:
                pass
            if not chunk: break
            results = pool.imap_unordered(_process_chunk, [chunk])
            for counts, batch in results:
                global_counts.update(counts)
                if batch: cur.executemany("INSERT INTO dists(f,t,d) VALUES(?,?,?)", batch)
            conn.commit()
            pbar.update(f.tell() - pbar.n)
        pool.close()
        pool.join()
    print("SQLite DB built. Creating index...")
    cur.execute("CREATE INDEX idx_f ON dists(f);")
    conn.commit()
    print("Index created successfully.")
    print("\n2/3: Building optimized binary format from SQLite DB...")
    index_data = {}
    current_offset = 0
    pair_format = struct.Struct('=II')
    sorted_keys = sorted(global_counts.keys())
    for from_id in sorted_keys:
        count = global_counts[from_id]
        index_data[from_id] = {"offset": current_offset, "count": count}
        current_offset += count * pair_format.size
    index_path.write_text(json.dumps(index_data), encoding="utf-8")
    with open(bin_path, "wb") as f_bin:
        print("Reading from SQLite and writing to binary file (this may take a while)...")
        query_result = cur.execute("SELECT f, t, d FROM dists ORDER BY f;")
        with tqdm(total=sum(global_counts.values()), desc="Writing binary data") as pbar:
            for _, to_id, dist in query_result:
                f_bin.write(pair_format.pack(to_id, dist))
                pbar.update(1)
    conn.close()
    print("\n3/3: Optimized caches built successfully.")


# --- БЛОК БЫСТРОГО ДОСТУПА К ДАННЫМ (ДЛЯ TSP) (без изменений) ---
class OptimizedDurationsReader:
    def __init__(self, bin_path, index_path):
        print("Initializing OptimizedDurationsReader for TSP stage...")
        index_data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.index = {int(k): v for k, v in index_data.items()}
        self.bin_file = open(bin_path, "rb")
        self.pair_format = struct.Struct('=II')
        print("Reader is ready.")

    def get_matrix_for_points(self, point_ids):
        matrix = defaultdict(lambda: 10_000_000)
        point_ids_set = set(point_ids)
        for from_id in point_ids:
            matrix[(from_id, from_id)] = 0
            if from_id in self.index:
                record = self.index[from_id]
                self.bin_file.seek(record["offset"])
                data_chunk = self.bin_file.read(record["count"] * self.pair_format.size)
                for to_id, dist in self.pair_format.iter_unpack(data_chunk):
                    if to_id in point_ids_set: matrix[(from_id, to_id)] = dist
        return matrix

    def close(self):
        self.bin_file.close()


# --- УТИЛИТАРНЫЕ ФУНКЦИИ (без изменений) ---
def connect_db(db_path):
    uri = f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only=ON;")
    return conn


def ram_mb(): return psutil.Process().memory_info().rss / (1024 * 1024)


def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


# --- ОСНОВНАЯ ЛОГИКА ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True)
    ap.add_argument("--couriers", required=True)
    ap.add_argument("--durations_json", required=True)
    ap.add_argument("--durations_db", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # 1. ПОДГОТОВКА ДАННЫХ (без изменений)
    print("--- Data Pre-processing Stage ---")
    start_time = time.time()
    durations_json_path = Path(args.durations_json)
    db_path = Path(args.durations_db)
    bin_path = db_path.with_suffix(".bin")
    index_path = db_path.with_suffix(".json_index")
    if not db_path.exists() or not bin_path.exists() or not index_path.exists():
        build_caches(durations_json_path, db_path, bin_path, index_path)
    else:
        print("All caches (SQLite, binary format) already exist. Skipping build.")
    print(f"Preprocessing finished in {time.time() - start_time:.2f} seconds.")

    # 2. ЗАГРУЗКА БИЗНЕС-ДАННЫХ (без изменений)
    print(f"\n--- Loading Business Data --- RAM: {ram_mb():.1f} MB")
    orders_json = load_json(args.orders)
    orders = {o["ID"]: o for o in orders_json["Orders"]}
    print(f"Loaded {len(orders)} orders. RAM: {ram_mb():.1f} MB")
    couriers_json = load_json(args.couriers)
    couriers_all = [c["ID"] for c in couriers_json["Couriers"]]
    couriers_ids = couriers_all[:COURIERS_TO_USE]
    print(f"Loaded {len(couriers_ids)} couriers. RAM: {ram_mb():.1f} MB")

    # 3. ЭТАП TSP (без изменений)
    print("\n--- TSP Stage (using optimized binary reader) ---")
    durations_reader = OptimizedDurationsReader(bin_path, index_path)
    mp_orders = defaultdict(list)
    for oid, o in orders.items(): mp_orders[o["MpId"]].append(oid)
    polygons_data = {}
    for mp_id, order_ids in tqdm(mp_orders.items(), desc="Solving TSP for polygons", unit="poly"):
        if not order_ids: continue
        local_dist_matrix = durations_reader.get_matrix_for_points(order_ids)
        manager = pywrapcp.RoutingIndexManager(len(order_ids), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback_tsp(from_index, to_index):
            from_oid = order_ids[manager.IndexToNode(from_index)]
            to_oid = order_ids[manager.IndexToNode(to_index)]
            return local_dist_matrix.get((from_oid, to_oid), 10_000_000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback_tsp)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            route_cost = solution.ObjectiveValue()
            seq = [0] * len(order_ids)
            index = routing.Start(0)
            for i in range(len(order_ids)):
                seq[i] = order_ids[manager.IndexToNode(index)]
                index = solution.Value(routing.NextVar(index))
            polygons_data[mp_id] = {"orders": seq, "internal_travel_time": route_cost}
    durations_reader.close()
    print(f"TSP stage completed for {len(polygons_data)} polygons.")

    # 4. ЭТАП VRP: СТРАТЕГИЯ "РАЗДЕЛЯЙ И ВЛАСТВУЙ"
    print(f"\n--- VRP Stage (Divide and Conquer Strategy) ---")
    conn = connect_db(db_path)
    cur = conn.cursor()

    @lru_cache(maxsize=8_000_000)
    def get_dist_vrp(a, b):
        if a == b: return 0
        query = "SELECT d FROM dists WHERE (f=? AND t=?) OR (f=? AND t=?) LIMIT 1;"
        row = cur.execute(query, (a, b, b, a)).fetchone()
        return int(row[0]) if row else 10_000_000

    for mp_id, data in polygons_data.items():
        seq = data["orders"]
        start_node = min(seq, key=lambda oid: get_dist_vrp(WAREHOUSE_ID_INTERNAL, oid))
        start_idx = seq.index(start_node)
        data["orders"] = seq[start_idx:] + seq[:start_idx]

    courier_mp_service_time = defaultdict(dict)
    for courier in couriers_json["Couriers"]:
        if courier["ID"] not in couriers_ids: continue
        service_times = {s["MpID"]: s["ServiceTime"] for s in courier.get("ServiceTimeInMps", [])}
        for mp_id, p_data in polygons_data.items():
            total_st = sum(service_times.get(orders[oid]["MpId"], 300) for oid in p_data["orders"])
            courier_mp_service_time[courier["ID"]][mp_id] = total_st

    all_polygon_ids = list(polygons_data.keys())
    random.shuffle(all_polygon_ids)

    polygon_clusters = []
    chunk_size = (len(all_polygon_ids) + NUM_CLUSTERS - 1) // NUM_CLUSTERS
    for i in range(0, len(all_polygon_ids), chunk_size):
        polygon_clusters.append(all_polygon_ids[i:i + chunk_size])

    final_routes_solution = defaultdict(list)
    total_work_time = 0

    available_couriers = list(couriers_ids)

    for i, cluster_poly_ids in enumerate(polygon_clusters):
        print(f"\n--- Processing Cluster {i + 1}/{NUM_CLUSTERS} with {len(cluster_poly_ids)} polygons ---")

        num_clusters_left = len(polygon_clusters) - i
        num_couriers_for_cluster = (len(available_couriers) + num_clusters_left - 1) // num_clusters_left
        couriers_for_cluster = available_couriers[:num_couriers_for_cluster]
        available_couriers = available_couriers[num_couriers_for_cluster:]

        if not couriers_for_cluster or not cluster_poly_ids:
            print("Skipping empty cluster or no available couriers.")
            continue

        polygons_for_cluster_data = {mp_id: polygons_data[mp_id] for mp_id in cluster_poly_ids}

        mp_id_to_vrp_idx = {mp_id: i + 1 for i, mp_id in enumerate(cluster_poly_ids)}
        vrp_idx_to_mp_id = {i + 1: mp_id for i, mp_id in enumerate(cluster_poly_ids)}
        num_locations = len(polygons_for_cluster_data) + 1
        num_vehicles = len(couriers_for_cluster)
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, WAREHOUSE_ID_INTERNAL)
        routing = pywrapcp.RoutingModel(manager)

        def travel_time_callback(from_index, to_index):
            from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            if from_node == WAREHOUSE_ID_INTERNAL:
                return get_dist_vrp(WAREHOUSE_ID_INTERNAL,
                                    polygons_for_cluster_data[vrp_idx_to_mp_id[to_node]]["orders"][
                                        0]) if to_node != WAREHOUSE_ID_INTERNAL else 0
            if to_node == WAREHOUSE_ID_INTERNAL:
                return get_dist_vrp(polygons_for_cluster_data[vrp_idx_to_mp_id[from_node]]["orders"][-1],
                                    WAREHOUSE_ID_INTERNAL)
            return get_dist_vrp(polygons_for_cluster_data[vrp_idx_to_mp_id[from_node]]["orders"][-1],
                                polygons_for_cluster_data[vrp_idx_to_mp_id[to_node]]["orders"][0])

        time_dimensions = []
        for vehicle_idx, cid in enumerate(couriers_for_cluster):
            def service_and_internal_travel_time_for_courier(index, courier_id=cid):
                node = manager.IndexToNode(index)
                if node == WAREHOUSE_ID_INTERNAL: return 0
                mp_id = vrp_idx_to_mp_id[node]
                p_data = polygons_for_cluster_data[mp_id]
                return p_data["internal_travel_time"] + courier_mp_service_time[courier_id][mp_id]

            def final_cost_callback_for_courier(from_idx, to_idx,
                                                service_func=service_and_internal_travel_time_for_courier):
                travel_cost = travel_time_callback(from_idx, to_idx)
                service_cost = service_func(from_idx)
                return travel_cost + service_cost

            final_cost_callback_index = routing.RegisterTransitCallback(final_cost_callback_for_courier)
            routing.SetArcCostEvaluatorOfVehicle(final_cost_callback_index, vehicle_idx)
            routing.AddDimension(final_cost_callback_index, 0, MAX_WORK_TIME, True, f'Time_{vehicle_idx}')
            time_dimensions.append(routing.GetDimensionOrDie(f'Time_{vehicle_idx}'))

        for node_idx in range(1, num_locations):
            mp_id = vrp_idx_to_mp_id[node_idx]
            penalty_cost = len(polygons_for_cluster_data[mp_id]["orders"]) * PENALTY
            routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty_cost)

        for dim in time_dimensions:
            dim.SetGlobalSpanCostCoefficient(BALANCING_PENALTY_COEFFICIENT)

        # --- ИЗМЕНЕНИЕ: Инициализация переменных для LNS ---
        best_solution_assignment = None
        last_successful_solution = None

        for lns_iter in range(LNS_ITERATIONS):
            print(f"  Cluster {i + 1} - LNS Iteration {lns_iter + 1}/{LNS_ITERATIONS}...")

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
            search_parameters.time_limit.FromSeconds(LNS_TIME_LIMIT_PER_ITERATION_SEC)
            search_parameters.log_search = True

            current_solution = None
            if best_solution_assignment is None:
                current_solution = routing.SolveWithParameters(search_parameters)
            else:
                # --- ИЗМЕНЕНИЕ: Используем last_successful_solution для извлечения маршрутов ---
                if not last_successful_solution:
                    print("    Cannot perform LNS: no previous successful solution found.")
                    break

                prev_routes = []
                for v_idx in range(num_vehicles):
                    prev_route = []
                    current_idx = routing.Start(v_idx)
                    current_idx = last_successful_solution.Value(routing.NextVar(current_idx))
                    while not routing.IsEnd(current_idx):
                        node_idx_from_route = manager.IndexToNode(current_idx)
                        prev_route.append(node_idx_from_route)
                        current_idx = last_successful_solution.Value(routing.NextVar(current_idx))
                    prev_routes.append(prev_route)

                assignment_to_modify = routing.ReadAssignmentFromRoutes(prev_routes, True)

                vehicles_to_free_indices = random.sample(range(num_vehicles),
                                                         k=int(num_vehicles * LNS_DESTROY_PERCENTAGE))
                for vehicle_idx_to_free in vehicles_to_free_indices:
                    for node_to_free in prev_routes[vehicle_idx_to_free]:
                        assignment_to_modify.Deactivate(manager.NodeToIndex(node_to_free))

                current_solution = routing.SolveFromAssignmentWithParameters(assignment_to_modify, search_parameters)

            # --- ИЗМЕНЕНИЕ: Обновляем обе переменные ---
            if current_solution and current_solution.ObjectiveValue() < (
            best_solution_assignment.ObjectiveValue() if best_solution_assignment else float('inf')):
                best_solution_assignment = routing.solver().Assignment(current_solution)
                last_successful_solution = current_solution
                print(f"    >>> New best solution for cluster! Score: {best_solution_assignment.ObjectiveValue()}")

        # --- ИЗМЕНЕНИЕ: Используем last_successful_solution для сбора результатов ---
        if last_successful_solution:
            print(f"  Finished processing cluster {i + 1}. Best score: {last_successful_solution.ObjectiveValue()}")
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                courier_id = couriers_for_cluster[vehicle_id]
                route = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != WAREHOUSE_ID_INTERNAL:
                        mp_id = vrp_idx_to_mp_id[node_index]
                        route.extend(polygons_for_cluster_data[mp_id]["orders"])
                    index = last_successful_solution.Value(routing.NextVar(index))
                if route:
                    final_routes_solution[courier_id].extend(route)
                    total_work_time += last_successful_solution.Min(
                        time_dimensions[vehicle_id].CumulVar(routing.End(vehicle_id)))

    conn.close()

    # --- 5. ФОРМАТИРОВАНИЕ И СОХРАНЕНИЕ ФИНАЛЬНОГО РЕШЕНИЯ (без изменений) ---
    print("\n--- Formatting and saving the final combined solution ---")

    assigned_orders_count = sum(len(r) for r in final_routes_solution.values())
    unassigned_orders_count = len(orders) - assigned_orders_count
    penalty = unassigned_orders_count * PENALTY
    final_score = total_work_time + penalty

    routes_with_wh = [{"courier_id": cid, "route": [WAREHOUSE_ID_OUTPUT] + r + [WAREHOUSE_ID_OUTPUT]} for cid, r in
                      final_routes_solution.items() if r]
    solution_json = {"routes": routes_with_wh}
    Path(args.output).write_text(json.dumps(solution_json, indent=2), encoding="utf-8")

    print(f"\n--- FINAL RESULTS ---")
    print(f"Saved solution to {args.output}")
    print(f"Assigned orders: {assigned_orders_count}/{len(orders)}")
    print(f"Total work time: {int(total_work_time)}s")
    print(f"Penalty for {unassigned_orders_count} unassigned orders: {int(penalty)}s")
    print(f"Final score (Time + Penalty): {int(final_score)}s")
    print(f"Final RAM: {ram_mb():.1f} MB")


if __name__ == "__main__":
    main()