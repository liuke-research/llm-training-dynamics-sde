import numpy as np


class DoubleWellSDE:


    def __init__(
        self,
        noise=0.5,
        dt=0.01
    ):

        self.noise = noise
        self.dt = dt



    # U(x)=1/4*x^4-1/2*x^2
    # gradient=x^3-x

    def gradient(
        self,
        x
    ):

        return x**3-x



    def step(
        self,
        x
    ):

        dw = np.random.randn()


        x_new = (
            x
            -
            self.gradient(x)
            *
            self.dt
            +
            self.noise
            *
            np.sqrt(self.dt)
            *
            dw
        )


        return x_new



    def simulate(
        self,
        x0,
        steps
    ):

        trajectory=[]

        x=x0

        for i in range(steps):

            x=self.step(x)

            trajectory.append(x)


        return np.array(
            trajectory
        )