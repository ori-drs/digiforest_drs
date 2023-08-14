#!/usr/bin/env python
import time

"""
A simple Timer class with context manager support
"""


class Timer:
    def __init__(self, ema=1.0):
        """_summary_

        Args:
            ema (float, optional): weight for exponential moving average ema*meas + (1 - ema)*average. Defaults to 0.0.
        """
        self._ema = ema
        self._key = ""
        self._tics = {}
        self._dts = {}

    def __call__(self, key="last"):
        self._key = key
        return self

    def __str__(self):
        out = ""
        for k, v in self._dts.items():
            out += f"{k:<10}: {v:<5.2f} s\n"
        return out

    def timing(self):
        return time.perf_counter()

    # Context manager interface
    def __enter__(self):
        self.tic(key=self._key)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.toc(key=self._key)

    # Normal interface
    def tic(self, key="last"):
        self._tics[key] = self.timing()

    def toc(self, key="last"):
        toc = self.timing()

        dt = toc - self._tics[key]
        if key in self._dts:
            self._dts[key] = self._ema * dt + (1.0 - self._ema) * self._dts[key]
        else:
            self._dts[key] = dt

        return self._dts[key]

    @property
    def dt(self, key="last"):
        return self._dts[key]


if __name__ == "__main__":
    print("Tic-toc interface")
    t = Timer()
    t.tic()
    time.sleep(1)
    t.toc()
    print(t)

    print("Context manager interface")
    timer = Timer()
    with timer("test0.5"):
        time.sleep(0.5)
    with timer("test1"):
        time.sleep(1)
    print(timer)

    # Average time with context manager
    alpha = 0.5
    print(f"Context manager with Exponential Moving average (weight={alpha})")
    timer = Timer(ema=alpha)
    for i in range(10):
        with timer("test_accumulation"):
            print(f"sleep for {i/10} s")
            time.sleep(i / 10)
    print(timer)
