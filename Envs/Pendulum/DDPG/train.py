from Envs.Pendulum.DDPG.BaselineNoise import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--NB_TRAINING_CYCLES", type=int, default=5)
    parser.add_argument("--NOISE", type=str, default="Gaussian")
    parser.add_argument("--REWARD_TYPE", type=str, default="dense")
    parser.add_argument("--PLOTTING", action="store_true", help="Enable plotting.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--training_steps", type=int, default=15000)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--buffer_length", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=100)

    args = parser.parse_args()

    train(args)