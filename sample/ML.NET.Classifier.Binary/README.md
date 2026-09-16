# ML.NET.Classifier.Binary

## Overview

`ML.NET.Classifier.Binary` is a Windows Forms sample showing how to use the published [ML.NET.Classification.Metrics.AucRoc 0.5.0](https://www.nuget.org/packages/ML.NET.Classification.Metrics.AucRoc/0.5.0) package with an ML.NET binary classifier.

The application trains Logistic Regression or Averaged Perceptron, prepares model scores and correct labels, sends them to the AucRoc package, and displays the returned ROC curve and AUC.

## Requirements

- Windows x64
- .NET 8.0 SDK or newer
- ML.NET 4.0.2

## Install the online AucRoc package

The package is installed in `ML.Performance/ML.Performance.csproj`

```xml
<PackageReference
  Include="ML.NET.Classification.Metrics.AucRoc"
  Version="0.5.0" />
```

To install the same package in another project:

```powershell
dotnet add "YourProject.csproj" package `
    ML.NET.Classification.Metrics.AucRoc `
    --version 0.5.0 `
    --source "https://api.nuget.org/v3/index.json"
```

## Windows x64 configuration

Version `0.5.0` contains a Windows x64 native library. The executable project, `ML.Design` contains:

```xml
<PlatformTarget>x64</PlatformTarget>
<RuntimeIdentifier>win-x64</RuntimeIdentifier>
```

This allows NuGet to copy the packaged `coremetrics.dll` into the application output directory.

## Restore and run

Open `ML.NET.Classifier.sln` inside the sample folder, or run:

```powershell
dotnet restore ".\ML.NET.Classifier.sln"
dotnet build ".\ML.NET.Classifier.sln" -r win-x64 --no-restore
dotnet run --project ".\ML.Design\ML.Design.csproj" -r win-x64 --no-restore
```

Load `ML.Dataset/diabetes_bin.csv`, select an algorithm, train the model, and select the Receiver Operating Characteristic graph.

## Implementation overview

| File | Purpose |
| --- | --- |
| `ML.DataPreparation/BinaryDataPreparation.cs` | Loads the CSV and creates training, validation and test data |
| `ML.Model/BinaryClassifiactionModel.cs` | Trains the selected model and evaluates the final test data |
| `ML.Model/BinaryEvaluationResult.cs` | Stores aligned raw scores and Boolean labels |
| `ML.Design/FormInterface.cs` | Connects training, evaluation, graph display and the metrics panel |
| `ML.Performance/ReceiverCurveCharacteristic.cs` | Calls `CoreMetricsEvaluator.EvaluateRoc` and plots its returned points |
| `ML.Performance/PerformanceVisualizer.cs` | Returns package AUC from the graph code to the form |
| `ML.Performance/PerformanceDescription.cs` | Formats package AUC for the performance metrics panel |

## How the sample works

### 1. Training, validation and test data

`BinaryDataPreparation` first reserves a final test set. `FormInterface.TrainBinaryModel` divides the remaining data into training and validation sets:

```csharp
var split = BinaryDataPreparation.SplitBinaryData(
    _mlContext,
    _trainingData,
    0.20,
    42,
    "Training/validation split");

IDataView train = split.TrainSet;
IDataView validation = split.TestSet;
```

The training set fits the model. The validation set selects the best decision threshold by F1 score. The untouched final test set produces the displayed results.

### 2. Logistic Regression and Averaged Perceptron

`FormInterface.TrainBinaryModel` calls one of these methods:

```csharp
_binaryModel.TrainLogisticRegression(train);
```

or:

```csharp
_binaryModel.TrainAveragedPerceptron(train);
```

| Model | Threshold optimization | ROC/AUC input |
| --- | --- | --- |
| Logistic Regression | Calibrated `Probability` | Raw `Score` |
| Averaged Perceptron | Raw `Score` | Raw `Score` |

ROC/AUC uses raw `Score` for both models because ROC evaluates ranking across all thresholds. It does not use the final Boolean `PredictedLabel`.

### 3. Preparing scores and labels

`BinaryClassificationModel.EvaluateThresholdMetrics` creates one cached prediction set from the final test data. `BinaryEvaluationResult` then prepares both collections from the same ordered rows:

```csharp
Scores = rows
    .Select(row => row.Score)
    .ToList()
    .AsReadOnly();

Labels = rows
    .Select(row => row.Label)
    .ToList()
    .AsReadOnly();
```

This guarantees that `Scores[i]` belongs to `Labels[i]`. AucRoc requires equal length arrays containing at least one positive and one negative label.

### 4. Calling `CoreMetricsEvaluator.EvaluateRoc`

The package API is imported in `ML.Performance/ReceiverCurveCharacteristic.cs`:

```csharp
using ML.NET.Metrics.Evaluation.Public;
```

The actual package call is inside `CalculateReceiverCurve`:

```csharp
return CoreMetricsEvaluator.EvaluateRoc(
    scores: scoreArray,
    labels: labelArray,
    mode: RocMode.Exact,
    clampScoresToUnitInterval: false);
```

It returns:

- `curve.Points`: ordered ROC points containing FPR, TPR and threshold
- `curve.Score`: AUC calculated for those same points

### 5. Displaying the ROC graph

`ReceiverCurveCharacteristic.ShowReceiverCurve` adds every returned point to the blue ROC line:

```csharp
foreach (RocPoint point in curve.Points)
{
    rocSeries.Points.AddXY(point.Fpr, point.Tpr);
}
```

The red markers are sampled from the same points to keep the graph readable. The dashed diagonal represents random selection.

### 6. Displaying AUC

`ShowReceiverCurve` returns the package result:

```csharp
return curve.Score;
```

`PerformanceVisualizer` returns it to `FormInterface.UpdateGraphDisplay` as `nativeAuc`. The form compares it with ML.NET's AUC as a consistency check, but the value sent to the performance panel is the package result:

```csharp
var descriptions =
    _performanceDescription.GetMetricsReceiverCurve(nativeAuc);

DisplayMetricsInColor(
    txtBoxPerformanceMetric_p2,
    descriptions);
```

The displayed AUC and plotted ROC curve therefore come from the same `EvaluateRoc` result.

## Exact and Binned ROC curves

| Mode | Input | Behaviour |
| --- | --- | --- |
| `RocMode.Exact` | Probabilities or unrestricted finite raw scores | Uses every unique score threshold and handles tied scores together |
| `RocMode.Binned` | Normally probabilities in `[0,1]` | Uses a selected number of fixed thresholds between one and zero |

This sample uses Exact mode because both supported models provide raw scores:

```csharp
MetricsCurve<RocPoint> curve =
    CoreMetricsEvaluator.EvaluateRoc(
        scores,
        labels,
        mode: RocMode.Exact,
        clampScoresToUnitInterval: false);
```

For calibrated probabilities, a general Binned call is:

```csharp
MetricsCurve<RocPoint> curve =
    CoreMetricsEvaluator.EvaluateRoc(
        probabilities,
        labels,
        mode: RocMode.Binned,
        buckets: 200,
        clampScoresToUnitInterval: false);
```

Exact mode gives the complete empirical ROC curve. Binned mode returns fewer fixed threshold points and can be useful for larger datasets or simpler graphs.

## License

This sample is distributed under the MIT License. See `LICENSE.txt`.
