import numpy as np
import scipy.io as sio
from displayData import display_data
import matplotlib.pyplot as plt
from predict import predict

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex3data1.mat')
m = data['X'].shape[0]
X, y = data['X'], data['y'].flatten()

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[:100], :]

display_data(sel)
plt.show()

input('Program paused. Press enter to continue.\n')

# ================ Part 2: Loading Pameters ================
print('\nLoading Saved Neural Network Parameters ...\n')
weights = sio.loadmat('ex3weights.mat')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
# print(Theta1.shape)  # (25,401)
# print(Theta2.shape)  # (10,26)

# ================= Part 3: Implement Predict =================
pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: {:.1f}%\n'.format(np.mean(pred == y) * 100))

input('Program paused. Press enter to continue.\n')

# Randomly permute examples
rp = np.random.permutation(range(m))
for i in range(m):
    # Display
    print('\nDisplaying Example Image\n')
    example = X[rp[i]]
    example = example.reshape((1, example.size))
    display_data(example)
    plt.show()

    pred = predict(Theta1, Theta2, example)
    print('\nNeural Network Prediction: {} (digit {})\n'.format(pred, np.mod(pred, 10)))

    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
