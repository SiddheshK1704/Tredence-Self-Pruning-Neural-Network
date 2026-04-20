import matplotlib.pyplot as plt

def plot_gates(gates):
    plt.figure()
    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig("gate_distribution.png")
    plt.close()