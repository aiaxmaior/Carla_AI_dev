# TrafficManager.py

import carla
import random
import logging

def spawn_traffic(client, world, num_vehicles=50, num_pedestrians=30):
    """
    Spawns vehicles and pedestrians and sets them to autopilot using the Traffic Manager.

    Args:
        client (carla.Client): The CARLA client.
        world (carla.World): The CARLA world.
        num_vehicles (int): The number of vehicles to spawn.
        num_pedestrians (int): The number of pedestrians to spawn.

    Returns:
        dict: A dictionary containing lists of the spawned 'vehicles' and 'pedestrians'.
    """
    spawned_actors = {'vehicles': [], 'pedestrians': []}
    
    # --- Spawn Vehicles ---
    traffic_manager = client.get_trafficmanager()
    # Every vehicle spawned will be assigned a different port in this range
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    # IMPORTANT: Set TrafficManager to synchronous mode. This will ensure it only
    # advances when the world ticks, which is what we want for client-driven synchronous sims.
    traffic_manager.set_synchronous_mode(True) 

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    # Filter out dangerous or less suitable vehicles for traffic
    safe_blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]
    safe_blueprints = [x for x in safe_blueprints if not x.id.endswith('microlino')]
    safe_blueprints = [x for x in safe_blueprints if not x.id.endswith('carlacola')]
    
    spawn_points = world.get_map().get_spawn_points()
    num_of_spawn_points = len(spawn_points)

    if num_vehicles > num_of_spawn_points:
        logging.warning(f"{num_vehicles} vehicles requested but only {num_of_spawn_points} spawn points available. Spawning {num_of_spawn_points}.")
        num_vehicles = num_of_spawn_points
    
    # Shuffle spawn points to get a good distribution
    random.shuffle(spawn_points)

    # Use a batch command to spawn all vehicles at once for performance
    batch_spawn_cmds = []
    for i in range(num_vehicles):
        blueprint = random.choice(safe_blueprints)
        # Use a spawn point from the shuffled list, pop to ensure uniqueness
        spawn_point = spawn_points.pop() 
        blueprint.set_attribute('role_name', 'autopilot') # Assign a role name for traffic
        batch_spawn_cmds.append(carla.command.SpawnActor(blueprint, spawn_point))
    
    results = client.apply_batch_sync(batch_spawn_cmds, True)
    for res in results:
        if not res.error:
            vehicle_actor = world.get_actor(res.actor_id)
            # Set autopilot to true and associate with the traffic manager's port
            vehicle_actor.set_autopilot(True, traffic_manager.get_port())
            spawned_actors['vehicles'].append(vehicle_actor)

    logging.info(f"Spawned {len(spawned_actors['vehicles'])} vehicles with Traffic Manager.")

    # --- Spawn Pedestrians ---
    walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')

    batch_walker_spawn_cmds = []
    for _ in range(num_pedestrians):
        walker_bp = random.choice(walker_blueprints)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        spawn_point = carla.Transform(world.get_random_location_from_navigation())
        if spawn_point is None: # Defensive check
            logging.warning("Could not find a suitable pedestrian spawn location. Skipping a pedestrian.")
            continue
        batch_walker_spawn_cmds.append(carla.command.SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch_walker_spawn_cmds, True)
    
    # Spawn a controller for each successfully spawned walker
    batch_controller_spawn_cmds = []
    # Store temporary references to walkers that spawned successfully
    successful_walkers = []
    for res in results:
        if not res.error:
            walker_actor = world.get_actor(res.actor_id)
            successful_walkers.append(walker_actor)
            cmd = carla.command.SpawnActor(controller_bp, carla.Transform(), walker_actor)
            batch_controller_spawn_cmds.append(cmd)
    
    controller_results = client.apply_batch_sync(batch_controller_spawn_cmds, True)
    
    # Start the AI for each controller
    for i, res in enumerate(controller_results):
        if not res.error:
            controller_actor = world.get_actor(res.actor_id)
            spawned_actors['pedestrians'].append(successful_walkers[i]) # Add pedestrian to list
            spawned_actors['pedestrians'].append(controller_actor) # Also add controller to cleanup list
            controller_actor.start()
            controller_actor.go_to_location(world.get_random_location_from_navigation())
            controller_actor.set_max_speed(1 + random.random()) # Speed between 1 and 2 m/s
        else:
            # If controller failed, ensure the pedestrian is destroyed too
            if i < len(successful_walkers):
                successful_walkers[i].destroy()
                logging.warning(f"Failed to spawn controller for pedestrian {successful_walkers[i].id}. Destroying pedestrian.")


    logging.info(f"Spawned {len(spawned_actors['pedestrians']) // 2} pedestrians and controllers.") # Divide by 2 because each pedestrian is a pair of actors (walker + controller)
    
    return spawned_actors
