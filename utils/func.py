from typing import Callable


def call_all(*callbacks: Callable):
    def aggregate_callback(*args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)
    return aggregate_callback