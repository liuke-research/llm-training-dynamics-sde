import numpy as np



class LangevinLLM:


    """
    Langevin approximation of SGD dynamics

    dθ = -∇L(θ)dt + σdW

    """



    def __init__(
        self,
        lr=0.01,
        noise=0.2,
        dt=0.01
    ):

        self.lr=lr

        self.noise=noise

        self.dt=dt



    def gradient(
        self,
        theta
    ):

        """
        toy loss landscape

        L(theta)=1/4 theta^4-1/2 theta^2

        """

        return theta**3-theta



    def step(
        self,
        theta
    ):


        dW=np.random.randn()


        theta = (
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
            dW
        )


        return theta



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


        return np.array(
            trajectory
        )