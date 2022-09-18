from torchvision import models
import numpy as np

# import trainer as trainer_module
import data_loader
import matplotlib.pyplot as plt
import adversarial_perturbation

def main():
    # model_dict = {
    #     'resnet': models.resnet152(pretrained=True),
    #     'inceptionv3' : models.inception_v3(pretrained=True)
    # }
    model_list = ['resnet152', 'inception_v3']

    # trainer = trainer_module.trainer()
    trainset, testset = data_loader.load_data()
    testset = None
    print(f'Loaded dataset: train ({len(trainset)}) val ({len(testset)})')
    accuracy = None #trainer.train(trainset,testset)
   
    for model_name in model_list:
        net = eval("torchvision.models.{}(pretrained=True)".format(model_name))
        v, fooling_rates, accuracies, total_iterations = adversarial_perturbation.generate(accuracy,trainset, testset, net)

        np.save(f'./perturbations/pert_uap_{model_name}.npy', v)

        plt.title("Fooling Rates over Universal Iterations"); plt.xlabel("Universal Algorithm Iter"); plt.ylabel("Fooling Rate on test data")
        plt.plot(total_iterations,fooling_rates) # plt.show()
        plt.savefig(f'./{model_name}/fooling_rates.png'); plt.clf()

        plt.title("Accuracy over Universal Iterations"); plt.xlabel("Universal Algorithm Iter"); plt.ylabel("Accuracy on Test data")
        plt.plot(total_iterations, accuracies) # plt.show()
        plt.savefig(f'./{model_name}/accuracies.png'); plt.clf()

if __name__ == "__main__":
    main()