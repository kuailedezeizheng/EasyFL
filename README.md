# EasyFL: Federated Learning Backdoor Attack Platform

# Dependency Requirements
1. Python version >= 3.9
2. Torch version >= 2.1

# Usage
1. Create a 'save' directory to store accuracy images.
2. Modify the 'config.toml' file in the 'configs' directory based on your requirements. You can use the provided template 'template_config.toml' as a reference. Three preset configurations are available:
    - (CIFAR - 10, MobileNet)
    - (MNIST, LeNet)
    - (CIFAR - 100, ResNet - 18)
3. Customize neural network architectures by defining your own models in the 'models' directory.
4. Run the 'main.py' file.

# Algorithm Steps
1. Load data.
2. Sample data.
3. Build the global model.
4. Initialize global model parameters.
5. Select participating clients.
6. Distribute the global model.
7. Train client models sequentially.
8. Aggregate client model parameters to obtain the global model.
9. Return to step 6, repeating until the end of an epoch.
10. Obtain the final global model.

# License
This project is licensed under the terms of the GNU General Public License v3.0 (GPL - 3.0). Feel free to explore, modify, and share your contributions under the conditions specified by the license.

EasyFL is designed to be a platform for federated learning backdoor attacks, providing a readable and maintainable environment for neural network research and experimentation. Customize configurations and models to suit your specific use case. Enjoy exploring the world of federated learning attacks with EasyFL!
