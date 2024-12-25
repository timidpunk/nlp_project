from transformers import pipeline
import torch
from functools import cache
import time
import tracemalloc

device = 'cuda:1'

@cache
def get_timers():
    return {
        'start' : torch.cuda.Event(enable_timing=True),
        'end' : torch.cuda.Event(enable_timing=True)
    }

class timer:
    """
    returns time in miliseconds
    """
    def __init__(self, type='default'):
        self.type = type

    def __enter__(self):
        tracemalloc.start()
        if self.type == 'torch':
            self.start, self.end = get_timers().values()
            self.start.record()
        else:
            self.start = time.time()
        
        return self  # Return self to access the context manager's attributes in the 'with' block
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.type == 'torch':
            self.end.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start.elapsed_time(self.end)
        else:
            self.elapsed_time = (time.time()- self.start) * 1000

        self.elapsed_memory = tracemalloc.get_traced_memory()[-1]
        self.gpu_memory = torch.cuda.memory_allocated(int(device[-1])) / 1024**3
        tracemalloc.stop()

def test_kv_cache(model, tokenizer, num_iters=10, text='<|startoftext|>', timer_type='torch'):
    model.eval()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    res = {}
    generator_kwargs = {
        'text_inputs': text,
        'max_length': num_iters,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 0.9,
        'do_sample': True
    }

    with timer(timer_type) as elapsed_time_cache:
        outs_cache = generator(
            **generator_kwargs,
            use_cache=True
        )
    torch.cuda.empty_cache()
    res['cache'] = {
        'text_generated' : outs_cache[0]['generated_text'],
        'elapsed_time' : elapsed_time_cache.elapsed_time,
        'input_text' : text,
        'ram_memory' : elapsed_time_cache.elapsed_memory, 
        'gpu_memory' : elapsed_time_cache.gpu_memory
    }

    with timer(timer_type) as elapsed_time_no_cache:
        outs_no_cache = generator(
            **generator_kwargs,
            use_cache=False
        )
    torch.cuda.empty_cache() 
    res['no_cache'] = {
        'text_generated' :  outs_no_cache[0]['generated_text'],
        'elapsed_time' : elapsed_time_no_cache.elapsed_time,
        'input_text' : text,
        'ram_memory' : elapsed_time_no_cache.elapsed_memory,
        'gpu_memory' : elapsed_time_no_cache.gpu_memory
    }

    return res
