import cProfile
import pstats
import io
import os
import psutil
import torch
from .logger import log

def profile():
    def decorator(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            try:
                pr.enable()
                result = func(*args, **kwargs)
                pr.disable()
            except ValueError as e:
                if log:
                    log.error(f"Profiling error: {e}")
                return func(*args, **kwargs)
            finally:
                try:
                    s = io.StringIO()
                    sortby = pstats.SortKey.CUMULATIVE
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    ps.print_stats()
                    if log:
                        log.info(s.getvalue())
                except TypeError as e:
                    if log:
                        log.error(f"Profiling stats error: {e}")

                process = psutil.Process(os.getpid())
                cpu_usage = process.cpu_percent(interval=1)
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)  # in MB

                if log:
                    log.info(f"CPU usage: {cpu_usage}%")
                    log.info(f"Memory usage: {memory_usage:.2f} MB")

                if torch.cuda.is_available():
                    allocated_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # in MB
                    cached_gpu_memory = torch.cuda.memory_reserved() / (1024 * 1024)  # in MB
                    if log:
                        log.info(f"Allocated GPU memory: {allocated_gpu_memory:.2f} MB")
                        log.info(f"Cached GPU memory: {cached_gpu_memory:.2f} MB")

            return result
        return wrapper
    return decorator
