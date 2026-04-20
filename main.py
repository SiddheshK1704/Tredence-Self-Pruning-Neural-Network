from train import train_model, evaluate
from utils import plot_gates
import config

def main():
    results = []

    for lam in config.LAMBDA_VALUES:
        print(f"\nRunning for lambda = {lam}")

        model, testloader = train_model(lam)
        acc, sparsity, gates = evaluate(model, testloader)

        print(f"Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

        results.append((lam, acc, sparsity))

    plot_gates(gates)

    print("\nFinal Results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()