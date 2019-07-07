import numpy as np
from processEmail import process_email
from emailFeatures import email_features
import scipy.io as sio
from sklearn.svm import SVC
from getVocabList import get_vocab_list

# ==================== Part 1: Email Preprocessing ====================
print('\nPreprocessing sample email (emailSample1.txt)\n')

file_content = open('emailSample1.txt', 'r').read()
word_indices = process_email(file_content)

print('Word Indices: \n')
print(' {}'.format(word_indices))
print('\n\n')

input('Program paused. Press enter to continue.\n')

# ==================== Part 2: Feature Extraction ====================
print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
file_content = open('emailSample1.txt', 'r').read()
word_indices = process_email(file_content)
features = email_features(word_indices)

print('Length of feature vector: {}\n'.format(features.size))
print('Number of non-zero entries: {}\n'.format(np.sum(features > 0)))

input('Program paused. Press enter to continue.\n')

# =========== Part 3: Train Linear SVM for Spam Classification ========

# % Load the Spam Email dataset
# % You will have X, y in your environment
data = sio.loadmat('spamTrain.mat')
X, y = data['X'], data['y'].flatten()

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
model = SVC(C, kernel='linear')
model.fit(X, y)

p = model.predict(X)

print('Training Accuracy: {}%\n'.format(np.mean(p == y) * 100))

# =================== Part 4: Test Spam Classification ================

# % Load the test dataset
# % You will have Xtest, ytest in your environment
data_test = sio.loadmat('spamTest.mat')
Xtest, ytest = data_test['Xtest'], data_test['ytest'].flatten()

print('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(Xtest)

print('Test Accuracy: {}%\n'.format(np.mean(p == ytest) * 100))
input('Program paused. Press enter to continue.\n')

# ================= Part 5: Top Predictors of Spam ====================
vocablist = get_vocab_list()
indices = np.argsort(model.coef_).flatten()[::-1]
print(indices)

for i in range(15):
    print(' {} ({})'.format(vocablist[indices[i]], model.coef_.flatten()[indices[i]]))

print('\n\n')
input('Program paused. Press enter to continue.\n')

# =================== Part 6: Try Your Own Emails =====================
filename = 'spamSample2.txt'

# Read and predict
file_content = open(filename).read()
word_indices = process_email(file_content)
x = email_features(word_indices)
x = np.reshape(x,(x.size,1)).T
p = model.predict(x)

print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename,p))
print('(1 indicates spam, 0 indicates not spam)\n\n')