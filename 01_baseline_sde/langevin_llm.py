import numpy as np


class LangevinLLM:


    """
    Continuous approximation of SGD dynamics:

    dθ = -∇L(θ)dt + σdW

    """


    def __init__(
        self,
        lr=0.01,
        noise=0.1,
        dt=0.01
    ):

        self.lr = lr
        self.noise = noise
        self.dt = dt



    def gradient(
        self,
        theta
    ):

        """
        Toy loss landscape

        L(theta)=1/4 theta^4 -1/2 theta^2

        """

        return theta**3-theta



    def step(
        self,
        theta
    ):


        noise = np.random.randn()


        theta_new = (
            theta
            -
            self.lr
            *
            self.gradient(theta)
            *
            self.dt
            +
            self.noise
            *
            np.sqrt(self.dt)
            *
            noise
        )


        return theta_new



    def simulate(
        self,
        theta0,
        steps
    ):


        trajectory=[]

        theta=theta0


        for _ in range(steps):

            theta=self.step(theta)

            trajectory.append(theta)


        return np.array(trajectory)