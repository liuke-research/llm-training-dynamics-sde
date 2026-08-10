import matplotlib.pyplot as plt


from langevin_llm import LangevinLLM



model=LangevinLLM(
    noise=0.5
)



trajectory=model.simulate(
    theta0=-1,
    steps=5000
)



plt.figure(
    figsize=(10,4)
)


plt.plot(
    trajectory
)


plt.xlabel(
    "Training step"
)


plt.ylabel(
    "Parameter state θ"
)


plt.title(
    "Langevin LLM Training Dynamics"
)


plt.show()