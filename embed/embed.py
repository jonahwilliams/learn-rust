from ctypes import cdll
import time

lib = cdll.LoadLibrary("target/release/libembed.dylib")

t1 = time.time()
lib.process()
t2 = time.time()

print(t2 - t1)
