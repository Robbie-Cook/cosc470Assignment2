import numpy as np

# Read in training data and labels

# Some useful parsing functions

# male/female -> 0/1
def parseSexLabel(string):
    if (string.startswith('male')):
        return 0
    if (string.startswith('female')):
        return 1
    print("ERROR parsing sex from " + string)

# child/teen/adult/senior -> 0/1/2/3
def parseAgeLabel(string):
    if (string.startswith('child')):
        return 0
    if (string.startswith('teen')):
        return 1
    if (string.startswith('adult')):
        return 2
    if (string.startswith('senior')):
        return 3
    print("ERROR parsing age from " + string)

# serious/smiling -> 0/1
def parseExpLabel(string):
    if (string.startswith('serious')):
        return 0
    if (string.startswith('smiling') or string.startswith('funny')):
        return 1
    print("ERROR parsing expression from " + string)

# Count number of training instances

numTraining = 0

for line in open ("MITFaces/faceDR"):
    numTraining += 1

dimensions = 128*128

trainingFaces = np.zeros([numTraining,dimensions])
trainingSexLabels = np.zeros(numTraining) # Sex - 0 = male; 1 = female
trainingAgeLabels = np.zeros(numTraining) # Age - 0 = child; 1 = teen; 2 = male 
trainingExpLabels = np.zeros(numTraining) # Expression - 0 = serious; 1 = smiling

index = 0
for line in open ("MITFaces/faceDR"):
    # Parse the label data
    parts = line.split()
    trainingSexLabels[index] = parseSexLabel(parts[2])
    trainingAgeLabels[index] = parseAgeLabel(parts[4])
    trainingExpLabels[index] = parseExpLabel(parts[8])
    # Read in the face
    fileName = "MITFaces/rawdata/" + parts[0]
    fileIn = open(fileName, 'rb')
    trainingFaces[index,:] = np.fromfile(fileIn, dtype=np.uint8,count=dimensions)/255.0
    fileIn.close()
    # And move along
    index += 1

# Count number of validation/testing instances

numValidation = 0
numTesting = 0

# Assume they're all Validation
for line in open ("MITFaces/faceDS"):
    numValidation += 1

# And make half of them testing
numTesting = int(numValidation/2)
numValidation -= numTesting

validationFaces = np.zeros([numValidation,dimensions])
validationSexLabels = np.zeros(numValidation) # Sex - 0 = male; 1 = female
validationAgeLabels = np.zeros(numValidation) # Age - 0 = child; 1 = teen; 2 = male 
validationExpLabels = np.zeros(numValidation) # Expression - 0 = serious; 1 = smiling

testingFaces = np.zeros([numTesting,dimensions])
testingSexLabels = np.zeros(numTesting) # Sex - 0 = male; 1 = female
testingAgeLabels = np.zeros(numTesting) # Age - 0 = child; 1 = teen; 2 = male 
testingExpLabels = np.zeros(numTesting) # Expression - 0 = serious; 1 = smiling

index = 0
for line in open ("MITFaces/faceDS"):
    # Parse the label data
    parts = line.split()

    if (index < numTesting):
        testingSexLabels[index] = parseSexLabel(parts[2])
        testingAgeLabels[index] = parseAgeLabel(parts[4])
        testingExpLabels[index] = parseExpLabel(parts[8])
        # Read in the face
        fileName = "MITFaces/rawdata/" + parts[0]
        fileIn = open(fileName, 'rb')
        testingFaces[index,:] = np.fromfile(fileIn, dtype=np.uint8,count=dimensions)/255.0
        fileIn.close()
    else:
        vIndex = index - numTesting
        validationSexLabels[vIndex] = parseSexLabel(parts[2])
        validationAgeLabels[vIndex] = parseAgeLabel(parts[4])
        validationExpLabels[vIndex] = parseExpLabel(parts[8])
        # Read in the face
        fileName = "MITFaces/rawdata/" + parts[0]
        fileIn = open(fileName, 'rb')
        validationFaces[vIndex,:] = np.fromfile(fileIn, dtype=np.uint8,count=dimensions)/255.0
        fileIn.close()
        
    # And move along
    index += 1










'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import tensorflow as tf
from tensorflow import keras

batch_size = 128
epochs = 12

x_train = trainingFaces
y_train = trainingSexLabels
x_test = testingFaces
y_test = testingSexLabels

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Conv2D(16, kernel_size=(1,1),activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=epochs,
          verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])