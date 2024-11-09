import numpy as np

def load_data(fileName, digits):
    data = []
    with open(fileName, 'r') as f:
        for line in f:
            row = list(map(float, line.strip().split()))
            if row[0] in digits:
                data.append(row)
    
    return np.array(data)

def save_data(data, fileName):
    np.savetxt(fileName, data, fmt='%.4f')

def main():
    trainFile = "ZipDigits.train.txt"
    testFile = "ZipDigits.test.txt"
    
    filteredTrainFile = "FilteredZipDigits.train.txt"
    filteredTestFile = "FilteredZipDigits.test.txt"
    
    digitsToKeep  = [1.0, 5.0]
    trainData = load_data(trainFile, digitsToKeep)
    testData = load_data(testFile, digitsToKeep)
    
    
    save_data(testData, filteredTestFile)
    save_data(trainData, filteredTrainFile)
    
    print(f"Filtered training data saved to {filteredTrainFile}")
    print(f"Filtered test data saved to {filteredTestFile}")

if __name__ == '__main__':
    main()