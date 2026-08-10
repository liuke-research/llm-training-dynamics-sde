import numpy as np
import pandas as pd



class TrainingLogGenerator:


    def __init__(
        self,
        steps=10000,
        seed=42
    ):

        self.steps = steps

        np.random.seed(seed)



    def generate(self):


        step = np.arange(
            self.steps
        )


        # =====================
        # loss decay
        # =====================

        loss = (
            5*np.exp(
                -step/3000
            )
            +
            0.5
        )


        # training noise

        noise = np.random.normal(
            0,
            0.05,
            self.steps
        )


        loss += noise



        # =====================
        # create instability
        # =====================

        spike_position = int(
            self.steps*0.75
        )


        loss[
            spike_position:
            spike_position+300
        ] += np.linspace(
            0,
            2,
            300
        )



        # =====================
        # gradient norm
        # =====================


        grad_norm = (
            np.abs(
                np.gradient(loss)
            )
            *
            100
            +
            np.random.rand(
                self.steps
            )
        )



        # =====================
        # learning rate
        # =====================


        lr = np.ones(
            self.steps
        )*0.001



        df = pd.DataFrame({

            "step":step,

            "loss":loss,

            "grad_norm":grad_norm,

            "learning_rate":lr

        })


        return df





if __name__=="__main__":


    generator = TrainingLogGenerator()


    data = generator.generate()


    data.to_csv(
        "training_log.csv",
        index=False
    )


    print(
        "training_log.csv generated"
    )