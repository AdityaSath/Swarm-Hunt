import torch
import torchrl
import vmas
import agilerl
import pettingzoo
import supersuit
import mpe2

print("torch", torch.__version__)
print("torchrl", torchrl.__version__)
print("vmas", vmas.__version__)
print("agilerl", __import__("importlib.metadata").metadata.version("agilerl"))
print("pettingzoo", pettingzoo.__version__)
print("mpe2", mpe2.__version__)
print("supersuit", supersuit.__version__)