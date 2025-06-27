# sys_tasks.py
import subprocess
import platform
import time
import os
import signal
import logging
import psutil # You may need to install this: pip install psutil

# Configure basic logging for this module
logger = logging.getLogger(__name__)

def get_running_processes():
  """Returns a list of dictionaries, each containing process information."""
  processes = []
  for process in psutil.process_iter(['pid', 'name', 'cmdline']): # Added 'cmdline' for more robust process identification
    try:
      process_info = process.info
      processes.append(process_info)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass # Some processes might be gone by the time we try to get their info.
  return processes

def sig_kill_engine(processes):
    """
    Attempts to forcefully terminate CARLA server processes by name.
    Looks for 'CarlaUE4-Linux-Shipping' (main executable) or 'CarlaUE4.sh' (launcher script).
    """
    # Names to look for. Add variations if your CARLA server uses a different name.
    carla_process_names = ['CarlaUE4-Linux-Shipping', 'CarlaUE4.sh', 'CarlaUE5.sh', 'CarlaUE5-Linux-Shipping', 'CarlaUE4.exe', 'CarlaUE5.exe']
    
    killed_count = 0
    for p_info in processes:
        p_name = p_info.get('name', '')
        p_pid = p_info.get('pid')
        p_cmdline = p_info.get('cmdline') # Get command line for broader matching

        # Check if process name or any part of cmdline matches CARLA server names
        found_carla = False
        for carla_name in carla_process_names:
            # Check process name
            if carla_name.lower() in p_name.lower():
                 found_carla = True
                 break
            # Check command line arguments if they exist and are iterable
            if isinstance(p_cmdline, list) and any(carla_name.lower() in arg.lower() for arg in p_cmdline):
                found_carla = True
                break
        
        if found_carla and p_pid:
            # Corrected print statement with a check for p_cmdline being iterable
            cmdline_str = ' '.join(p_cmdline) if isinstance(p_cmdline, list) else "N/A"
            print(f"Found CARLA process: PID: {p_pid}, Name: '{p_name}', Cmdline: '{cmdline_str}'. Attempting SIGKILL.")
            try:
                os.kill(p_pid, signal.SIGKILL)
                print(f"PID: {p_pid} ('{p_name}') killed forcefully.")
                killed_count += 1
            except ProcessLookupError:
                print(f"PID: {p_pid} ('{p_name}') already gone.")
            except OSError as e:
                print(f"Error killing PID: {p_pid} ('{p_name}'): {e}")
    if killed_count == 0:
        print("No active CARLA server processes found or terminated by sig_kill_engine.")


def terminate_popen_process_gracefully(process_obj: subprocess.Popen, process_name="Process", timeout=10):
    """
    Gracefully terminates a subprocess.Popen object.
    Sends SIGTERM, waits, then sends SIGKILL if it doesn't terminate.

    Args:
        process_obj (subprocess.Popen): The Popen object representing the process.
        process_name (str): A descriptive name for the process for logging.
        timeout (int): Seconds to wait for graceful termination before forcing.

    Returns:
        bool: True if the process was terminated (either gracefully or forcefully), False otherwise.
    """
    if process_obj is None:
        logger.info(f"{process_name} object is None, nothing to terminate.")
        return True # Considered terminated as there's nothing to do

    if process_obj.poll() is None:  # Check if process exists and is running
        logger.info(f"Terminating {process_name} (PID: {process_obj.pid})...")
        process_obj.terminate()  # Send SIGTERM (or equivalent on Windows)
        try:
            process_obj.wait(timeout=timeout)
            logger.info(f"{process_name} (PID: {process_obj.pid}) terminated gracefully.")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                f"{process_name} (PID: {process_obj.pid}) did not terminate gracefully after {timeout}s, forcing kill."
            )
            process_obj.kill()  # Send SIGKILL (or equivalent on Windows)
            try:
                process_obj.wait(timeout=2) # Short wait to confirm kill
                logger.info(f"{process_name} (PID: {process_obj.pid}) killed forcefully.")
            except subprocess.TimeoutExpired:
                logger.error(f"Failed to confirm kill for {process_name} (PID: {process_obj.pid}) even after SIGKILL.")
                return False # Failed to confirm kill
            except Exception as e: # Catch other potential errors during wait after kill
                logger.error(f"Error waiting for {process_name} (PID: {process_obj.pid}) after kill: {e}")
            return True
        except Exception as e:
            logger.error(f"Error during termination of {process_name} (PID: {process_obj.pid}): {e}")
            return False
    else:
        logger.info(f"{process_name} (PID: {process_obj.pid if hasattr(process_obj, 'pid') else 'N/A'}) already terminated (poll result: {process_obj.poll()}).")
        return True

def terminate_processes_by_name(names_to_kill, timeout=5):
    """
    Finds and terminates processes matching a list of names using psutil.
    Tries SIGTERM first, then SIGKILL if timeout is reached.

    Args:
        names_to_kill (list): A list of process names (case-insensitive partial match on Linux/macOS, exact on Windows for exe).
                              e.g., ["CarlaUE4.sh", "CarlaUE4-Linux-Shipping"]
        timeout (int): Seconds to wait for graceful termination if signal_type is "TERM".

    Returns:
        int: Count of processes successfully signaled for termination.
    """
    if not isinstance(names_to_kill, list):
        names_to_kill = [names_to_kill]
    
    lower_names_to_kill = [name.lower() for name in names_to_kill]
    processes_terminated_count = 0
    procs_to_terminate = []

    # First, find all matching processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            p_name = proc.info['name'] if proc.info['name'] else ""
            p_cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ""

            found = False
            for target_name_lower in lower_names_to_kill:
                if target_name_lower in p_name.lower():
                    found = True
                    break
                # Checking cmdline is good for shell scripts like CarlaUE4.sh
                if target_name_lower in p_cmdline.lower(): 
                    found = True
                    break
            
            if found:
                procs_to_terminate.append(proc)
                logger.info(f"Found matching process: PID={proc.pid}, Name='{p_name}', Cmdline='{p_cmdline}'")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue # Process might have died, or we don't have permission

    if not procs_to_terminate:
        logger.info(f"No running processes found matching names: {names_to_kill}")
        return 0

    # Now, terminate them
    for proc in procs_to_terminate:
        try:
            logger.info(f"Attempting to terminate PID: {proc.pid}, Name: '{proc.name()}' with SIGTERM")
            proc.terminate() # Send SIGTERM
        except psutil.NoSuchProcess:
            logger.info(f"Process PID {proc.pid} already terminated before action.")
            processes_terminated_count += 1
            continue
        except psutil.AccessDenied:
            logger.error(f"Access denied to terminate PID {proc.pid}. Try with higher privileges.")
            continue
        except Exception as e:
            logger.error(f"Error sending SIGTERM to PID {proc.pid}: {e}")
            continue
    
    # Wait for graceful termination
    gone, alive = psutil.wait_procs(procs_to_terminate, timeout=timeout)
    
    for p in gone:
        logger.info(f"PID: {p.pid} terminated gracefully.")
        processes_terminated_count += 1
    
    # Force kill any remaining processes
    if alive:
        for p in alive:
            logger.warning(f"PID: {p.pid} did not terminate after {timeout}s (SIGTERM). Forcing kill (SIGKILL).")
            try:
                p.kill()
                logger.info(f"PID: {p.pid} killed forcefully.")
                processes_terminated_count += 1
            except psutil.NoSuchProcess:
                logger.info(f"PID: {p.pid} was already gone before SIGKILL.")
                processes_terminated_count += 1 # It's gone, so count it
            except Exception as e:
                logger.error(f"Error during SIGKILL for PID {p.pid}: {e}")

    return processes_terminated_count
