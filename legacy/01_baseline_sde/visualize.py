import matplotlib.pyplot as plt

from double_well import DoubleWellSDE


print("Start SDE demo")


sde = DoubleWellSDE(
    noise=0.6
)


trajectory = sde.simulate(
    x0=-1,
    steps=5000
)


print("Trajectory length:", len(trajectory))
print("Min:", trajectory.min())
print("Max:", trajectory.max())


plt.figure(
    figsize=(10,4)
)


plt.plot(
    trajectory
)


plt.xlabel(
    "Step"
)

plt.ylabel(
    "State x"
)


plt.title(
    "Double Well SDE Dynamics"
)


plt.show()