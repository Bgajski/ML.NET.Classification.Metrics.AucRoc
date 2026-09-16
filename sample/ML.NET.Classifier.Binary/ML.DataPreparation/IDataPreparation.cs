using Microsoft.ML;

public interface IDataPreparation
{
    // prepares generic data for training/testing
    (IDataView TrainingData, IDataView TestData) PrepareData(MLContext mlContext, string filePath, int featureCount);
}

// extends data preparation interface
public interface IBinaryDataPreparation : IDataPreparation
{
    (IDataView TrainingData, IDataView TestData) PrepareDataForLogisticRegression(MLContext mlContext, string filePath, int featureCount);

    (IDataView TrainingData, IDataView TestData) PrepareDataForAveragedPerceptron(MLContext mlContext, string filePath, int featureCount);
}
