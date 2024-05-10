import matplotlib.pyplot as plt
import pickle
import os

if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    with open("./results/results_base_cifar10_r18.pkl", "rb") as f:
        info_base_cifar10_r18 = pickle.load(f)

    with open("./results/results_base_cifar10_r50.pkl", "rb") as f:
        info_base_cifar10_r50 = pickle.load(f)

    with open("./results/results_ib_cifar10_r18.pkl", "rb") as f:
        info_ib_cifar10_r18 = pickle.load(f)
    
    with open("./results/results_ib_cifar10_r50.pkl", "rb") as f:
        info_ib_cifar10_r50 = pickle.load(f)
    
    
    # Plot loss graph for base and ib models with resnet18
    plt.figure(figsize=(10, 8))
    plt.plot(info_base_cifar10_r18["train_loss"], label="Baseline (Train)", color="indianred", linewidth=3)
    plt.plot(info_ib_cifar10_r18["train_loss"], label="InfoBatch (Train)", color="darkorange", linewidth=3)
    plt.plot(info_base_cifar10_r18["test_loss"], label="Baseline (Test)", color="mediumseagreen", linewidth=3, linestyle="--")
    plt.plot(info_ib_cifar10_r18["test_loss"], label="InfoBatch (Test)", color="slateblue", linewidth=3, linestyle="--")

    plt.xlabel("Epoch", fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel("Loss", fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=20)
    plt.grid(color="gray", linestyle="--", linewidth=2)
    plt.savefig("./figures/loss_base_ib_cifar10_r18.png")

    # Plot loss graph for base and ib models with resnet50
    plt.figure(figsize=(10, 8))
    plt.plot(info_base_cifar10_r50["train_loss"], label="Baseline (Train)", color="indianred", linewidth=3)
    plt.plot(info_ib_cifar10_r50["train_loss"], label="InfoBatch (Train)", color="darkorange", linewidth=3)
    plt.plot(info_base_cifar10_r50["test_loss"], label="Baseline (Test)", color="mediumseagreen", linewidth=3, linestyle="--")
    plt.plot(info_ib_cifar10_r50["test_loss"], label="InfoBatch (Test)", color="slateblue", linewidth=3, linestyle="--")

    plt.xlabel("Epoch", fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel("Loss", fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=20)
    plt.grid(color="gray", linestyle="--", linewidth=2)
    plt.savefig("./figures/loss_base_ib_cifar10_r50.png")

    # Plot accuracy graph for base and ib models with resnet18
    plt.figure(figsize=(10, 8))
    plt.plot(info_base_cifar10_r18["train_acc"], label="Baseline (Train)", color="indianred", linewidth=3)
    plt.plot(info_ib_cifar10_r18["train_acc"], label="InfoBatch (Train)", color="darkorange", linewidth=3)
    plt.plot(info_base_cifar10_r18["test_acc"], label="Baseline (Test)", color="mediumseagreen", linewidth=3, linestyle="--")
    plt.plot(info_ib_cifar10_r18["test_acc"], label="InfoBatch (Test)", color="slateblue", linewidth=3, linestyle="--")

    plt.xlabel("Epoch", fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel("Accuracy", fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=20)
    plt.grid(color="gray", linestyle="--", linewidth=2)
    plt.savefig("./figures/acc_base_ib_cifar10_r18.png")

    # Plot accuracy graph for base and ib models with resnet50
    plt.figure(figsize=(10, 8))
    plt.plot(info_base_cifar10_r50["train_acc"], label="Baseline (Train)", color="indianred", linewidth=3)
    plt.plot(info_ib_cifar10_r50["train_acc"], label="InfoBatch(Train)", color="darkorange", linewidth=3)
    plt.plot(info_base_cifar10_r50["test_acc"], label="Baseline (Test)", color="mediumseagreen", linewidth=3, linestyle="--")
    plt.plot(info_ib_cifar10_r50["test_acc"], label="InfoBatch (Test)", color="slateblue", linewidth=3, linestyle="--")

    plt.xlabel("Epoch", fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel("Accuracy", fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=20)
    plt.grid(color="gray", linestyle="--", linewidth=2)
    plt.savefig("./figures/acc_base_ib_cifar10_r50.png")


    # print the best accuracy
    print("ResNet18 Baseline Best Accuracy: ", info_base_cifar10_r18["best_acc"],"%")
    print("ResNet18 InfoBatch Best Accuracy: ", info_ib_cifar10_r18["best_acc"],"%")
    print("ResNet50 Baseline Best Accuracy: ", info_base_cifar10_r50["best_acc"],"%")
    print("ResNet50 InfoBatch Best Accuracy: ", info_ib_cifar10_r50["best_acc"],"%")

    # print the total time
    print("ResNet18 Baseline Total Time: ", info_base_cifar10_r18["total_time"]/60, "minutes")
    print("ResNet18 InfoBatch Total Time: ", info_ib_cifar10_r18["total_time"]/60, "minutes")
    print("ResNet50 Baseline Total Time: ", info_base_cifar10_r50["total_time"]/60, "minutes")
    print("ResNet50 InfoBatch Total Time: ", info_ib_cifar10_r50["total_time"]/60, "minutes")

    # Print the ratio of total time between baseline and infobatch
    print("ResNet18 Time Ratio Percentage: ", 100-(info_ib_cifar10_r18["total_time"]/info_base_cifar10_r18["total_time"])*100,"%")
    print("ResNet50 Time Ratio Percentage: ", 100-(info_ib_cifar10_r50["total_time"]/info_base_cifar10_r50["total_time"])*100,"%")