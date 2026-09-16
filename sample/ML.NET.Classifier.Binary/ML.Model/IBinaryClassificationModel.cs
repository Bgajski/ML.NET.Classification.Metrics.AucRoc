using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;

/// <summary>
/// interface for binary classification model implementations
/// defines training methods for logistic regression and averaged perceptron,
/// as well as methods for evaluation, transformation and prediction extraction
/// </summary>
public interface IBinaryClassificationModel
{
    ITransformer TrainLogisticRegression(IDataView trainingData);

    ITransformer TrainAveragedPerceptron(IDataView trainingData);

    // ML.NET default-threshold metrics
    BinaryClassificationMetrics EvaluateModel(IDataView testData);

    // classification metrics from threshold adjusted predictions
    ML.Model.BinaryEvaluationResult EvaluateThresholdMetrics(IDataView testData);

    // transforms data using trained model
    IDataView Transform(IDataView data);

    // extracts raw scores and actual labels for ROC
    (List<float> probs, List<bool> labels) GetPredictionsAndLabels(IDataView testData);

    // finds threshold that maximizes f1 score using validation data
    double OptimizeThresholdByF1(IDataView validationData);
}
