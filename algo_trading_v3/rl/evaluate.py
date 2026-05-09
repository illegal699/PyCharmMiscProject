# rl/evaluate.py
import pandas as pd
import matplotlib.pyplot as plt
from rl.environment import TradingEnv
from rl.agent import TradingAgent


def evaluate_model(model_path: str, df: pd.DataFrame):
    env = TradingEnv(df)
    agent = TradingAgent(env)
    agent.load(model_path)

    obs, _ = env.reset()
    done = False
    equity_curve = []

    while not done:
        action = agent.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        equity_curve.append(env.equity)

    # Wizualizacja
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title("Equity Curve - Backtest")
    plt.xlabel("Kroki")
    plt.ylabel("Equity (USDT)")
    plt.grid(True)
    plt.show()

    print(f"Final Equity: ${env.equity:,.2f} | Return: {(env.equity / env.initial_capital - 1) * 100:.2f}%")


if __name__ == "__main__":
    df = pd.read_csv("data/raw/test_data.csv")
    evaluate_model("models/final_model.zip", df)