"""
Parallel processing utilities for alabi with MPI/multiprocessing compatibility.

This module provides utilities to handle the interaction between MPI (used by PyMultiNest)
and Python's multiprocessing module (used by other alabi functions). When MPI is active,
multiprocessing pools can cause conflicts, so this module provides safe alternatives.
"""

import os
import warnings
import multiprocess as mp


def is_mpi_available():
    """
    Check if MPI is available on the system.
    
    :returns: bool
        True if MPI (mpi4py) is importable, False otherwise.
    """
    try:
        import mpi4py
        return True
    except ImportError:
        return False


def is_mpi_active():
    """
    Check if MPI is currently active/initialized.
    
    This function checks if MPI has been initialized and is currently running
    in a parallel context (i.e., multiple MPI processes).
    
    :returns: bool
        True if MPI is active with multiple processes, False otherwise.
    """
    if not is_mpi_available():
        return False
        
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        # MPI is considered "active" if we have multiple processes
        return comm.Get_size() > 1
    except:
        return False


def is_multiprocessing_safe():
    """
    Check if it's safe to use multiprocessing pools.
    
    Multiprocessing can conflict with MPI, so this function checks
    if multiprocessing should be avoided.
    
    :returns: bool
        True if multiprocessing is safe to use, False if it should be avoided.
    """
    # Check if we're in an MPI environment
    if is_mpi_active():
        return False
        
    # Check for environment variables that indicate MPI usage
    mpi_env_vars = [
        'OMPI_COMM_WORLD_SIZE',  # OpenMPI
        'PMI_SIZE',              # Intel MPI, MPICH
        'SLURM_NTASKS',          # SLURM with MPI
        'LSB_DJOB_NUMPROC',      # LSF
    ]
    
    for var in mpi_env_vars:
        if var in os.environ:
            try:
                size = int(os.environ[var])
                if size > 1:
                    return False
            except (ValueError, KeyError):
                pass
    
    return True


def safe_multiprocessing_pool(ncore, force_disable=False):
    """
    Create a multiprocessing pool if safe, otherwise return None.
    
    This function checks if multiprocessing is safe to use and creates
    a pool accordingly. If multiprocessing is not safe (e.g., due to MPI),
    it returns None and the calling code should fall back to serial execution.
    
    :param ncore: int
        Number of cores to use for the pool.
        
    :param force_disable: bool, optional
        If True, always return None (disable multiprocessing). Default is False.
        
    :returns: multiprocessing.Pool or None
        Pool object if safe to use, None if multiprocessing should be avoided.
        
    :example:
        >>> pool = safe_multiprocessing_pool(4)
        >>> if pool is not None:
        ...     results = pool.map(func, data)
        ...     pool.close()
        ...     pool.join()
        ... else:
        ...     results = [func(x) for x in data]  # Serial fallback
    """
    if force_disable:
        return None
        
    if not is_multiprocessing_safe():
        warnings.warn(
            "Multiprocessing disabled due to MPI environment. "
            "Using serial execution to avoid conflicts.",
            RuntimeWarning
        )
        return None
    
    try:
        return mp.Pool(ncore)
    
    except Exception as e:
        warnings.warn(
            f"Failed to create multiprocessing pool: {e}. "
            "Falling back to serial execution.",
            RuntimeWarning
        )
        return None


def get_safe_ncore(requested_ncore):
    """
    Get the safe number of cores to use for parallel processing.
    
    If multiprocessing is not safe (e.g., due to MPI), returns 1
    to force serial execution.
    
    :param requested_ncore: int
        The originally requested number of cores.
        
    :returns: int
        Safe number of cores (1 if multiprocessing is unsafe).
    """
    if not is_multiprocessing_safe():
        return 1
    return requested_ncore


def safe_pool_map(func, data, ncore, chunksize=None):
    """
    Apply a function to data using multiprocessing if safe, otherwise serial.
    
    This is a convenience function that handles the common pattern of
    applying a function to a list of data with optional multiprocessing.
    
    :param func: callable
        Function to apply to each element of data.
        
    :param data: iterable
        Data to apply function to.
        
    :param ncore: int
        Number of cores to use (if multiprocessing is safe).
        
    :param chunksize: int or None, optional
        Chunk size for pool.map(). Default is None.
        
    :returns: list
        Results of applying func to data.
        
    :example:
        >>> def square(x):
        ...     return x**2
        >>> results = safe_pool_map(square, [1, 2, 3, 4], ncore=2)
        >>> print(results)  # [1, 4, 9, 16]
    """
    if ncore <= 1:
        return [func(x) for x in data]
    
    pool = safe_multiprocessing_pool(ncore)
    if pool is not None:
        try:
            if chunksize is not None:
                results = pool.map(func, data, chunksize=chunksize)
            else:
                results = pool.map(func, data)
            return list(results)  # Ensure return type is list
        finally:
            pool.close()
            pool.join()
    else:
        # Fallback to serial execution
        return [func(x) for x in data]


def get_parallel_info():
    """
    Get information about the current parallel environment.
    
    :returns: dict
        Dictionary with information about MPI availability, status,
        and multiprocessing safety.
    """
    info = {
        'mpi_available': is_mpi_available(),
        'mpi_active': is_mpi_active(),
        'multiprocessing_safe': is_multiprocessing_safe(),
        'cpu_count': mp.cpu_count(),
    }
    
    if info['mpi_active']:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            info['mpi_size'] = comm.Get_size()
            info['mpi_rank'] = comm.Get_rank()
        except:
            pass
    
    return info


def print_parallel_info():
    """
    Print information about the current parallel environment.
    
    Useful for debugging parallel processing issues.
    """
    info = get_parallel_info()
    
    print("Parallel Environment Information:")
    print(f"  MPI Available: {info['mpi_available']}")
    print(f"  MPI Active: {info['mpi_active']}")
    print(f"  Multiprocessing Safe: {info['multiprocessing_safe']}")
    print(f"  CPU Count: {info['cpu_count']}")
    
    if 'mpi_size' in info:
        print(f"  MPI Size: {info['mpi_size']}")
        print(f"  MPI Rank: {info['mpi_rank']}")
    
    # Check for relevant environment variables
    env_vars = ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'LSB_DJOB_NUMPROC']
    for var in env_vars:
        if var in os.environ:
            print(f"  {var}: {os.environ[var]}")
