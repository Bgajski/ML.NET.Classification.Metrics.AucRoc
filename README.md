# ML.NET.Classification.Metrics.AucRoc

## 🧾 Overview

ML.NET.Classification.Metrics.AucRoc is an unofficial library for ML.NET, intended to be used as the [ML.NET.Classification.Metrics.AucRoc](https://www.nuget.org/packages/ML.NET.Classification.Metrics.AucRoc/0.5.0) package to generate ROC curves and calculate the Area Under the ROC Curve (AUC) for binary classification models.

The library enables users to:

- Generate Exact and Binned ROC curves
- Calculate AUC from the generated ROC points
- Access the threshold, True Positive Rate and False Positive Rate of every point
- Use probabilities or raw model scores
- Display the returned points in charts and performance evaluation tools
- Calculate ROC and AUC through the native C++ `coremetrics.dll` implementation

## [ML.NET.Classifier binary classification examples](https://github.com/Bgajski/ML.NET.Classification.Metrics.AucRoc/tree/main/sample)

### Logistic Regression

![ML.NET.Classifier_logistic_regression](/sample/ML.NET.Classifier.Binary.ReadmeExtra/ML.NET.logistic.png)

### Averaged Perceptron

![ML.NET.Classifier_averaged_perceptron](/sample/ML.NET.Classifier.Binary.ReadmeExtra/ML.NET.perceptron.png)

## Binary classification metrics overview

A binary classification model separates data into two groups, such as positive and negative, diabetes and no diabetes, or spam and ham.

The model returns a score for every tested row. A larger score means that the row is considered more likely to belong to the positive class. The ROC curve evaluates how the model behaves when the decision threshold is changed.

The library expects:

- `scores (float[])`: model probabilities or raw scores
- `labels (bool[])`: correct labels, where `true` is positive and `false` is negative

The arrays must have the same length and use the same row order.

## 🧮 ROC curve calculation

For every threshold, a row is predicted as positive when:

```text
score >= threshold
```

The library counts:

- True Positives (TP): positive rows correctly predicted as positive
- False Positives (FP): negative rows incorrectly predicted as positive
- Positives (P): total number of positive rows
- Negatives (N): total number of negative rows

The coordinates of every ROC point are calculated as:

```text
True Positive Rate  = TP / P
False Positive Rate = FP / N
```

The ROC graph uses:

- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (TPR)

When the threshold is lowered, more rows are classified as positive. The curve therefore moves from `(0, 0)` toward `(1, 1)`.

The diagonal line represents random classification. A useful model normally produces a curve above this line and closer to the upper-left corner.

## Exact and Binned ROC curves

| Mode   | Calculation | Main use |
| ------ | ----------- | -------- |
| Exact  | Uses every unique model score as a threshold | Complete ROC curve and exact empirical AUC |
| Binned | Uses a fixed number of thresholds from 1.0 to 0.0 | Faster calculation and fixed-size graph |

### Exact ROC

Exact mode sorts all rows by score from highest to lowest and creates a new ROC point after every unique score threshold.

Rows with the same score are processed together. This is important because processing tied scores separately could create artificial ROC points based only on row order.

The curve also contains the start and end points:

```text
(0, 0)
(1, 1)
```

Exact means that no unique score threshold is skipped. It does not mean that the classification model itself has perfect accuracy.

### Binned ROC

Binned mode uses a selected number of equally spaced thresholds.

For `B` buckets, the thresholds are:

```text
threshold[j] = 1 - j / B
```

With `buckets: 200`, the library evaluates 201 thresholds between `1.0` and `0.0`.

A larger bucket count produces a more detailed curve. A smaller bucket count produces fewer points and a faster approximation.

Binned mode should be used with probabilities in the `[0,1]` range.

## 🧮 AUC calculation

AUC is the total area under the ROC curve. The library calculates it from the same ROC points that it returns to the application.

The area between two neighbouring points is calculated with the trapezoidal rule:

```text
Area = (FPR2 - FPR1) * (TPR1 + TPR2) / 2
```

The final result is the sum of all trapezoids:

```text
AUC = sum of all calculated areas
```

This ensures that the displayed ROC curve and the displayed AUC describe the same result.

AUC can also be understood as the probability that the model gives a randomly selected positive row a higher score than a randomly selected negative row. Tied scores receive half credit.

| AUC value | Meaning |
| --------- | ------- |
| 1.00      | All positive rows are ranked above all negative rows |
| 0.50      | Ranking is similar to random selection |
| Below 0.50 | The ranking is mostly reversed |

AUC measures model ranking across all thresholds. It is different from Accuracy, Precision and Recall, which describe results at one selected threshold.

## Scores and probabilities

Use `Probability` when the trained model provides calibrated values between `0` and `1`:

```csharp
clampScoresToUnitInterval: true
```

Use the raw `Score` with Exact mode when the model does not provide `Probability`:

```csharp
clampScoresToUnitInterval: false
```

Exact ROC depends on the order of scores, so raw scores do not need to be between `0` and `1`.

Clamping applies the following calculation:

```text
clampedScore = min(1, max(0, score))
```

Raw scores should not normally be clamped because different values below `0` or above `1` would become tied. This can change the ROC points and AUC.

The final `PredictedLabel` should not be used as the ROC score. It contains only `true` or `false` and removes the threshold information required to build a detailed curve.

## ⚙️ Native C++ implementation

The library uses a managed .NET API and a native Windows x64 C++ library.

Calculation process:

1. The ML.NET application produces scores and labels.
2. `CoreMetricsEvaluator` sends the arrays to `coremetrics.dll`.
3. The C++ implementation sorts or bins the scores.
4. C++ calculates ROC points and AUC.
5. The results are returned to .NET as `MetricsCurve<RocPoint>`.
6. The application uses the points to display the ROC graph.

## ML.NET.Classifier.Binary implementation

The README inside [`ML.NET.Classifier.Binary`](https://github.com/Bgajski/ML.NET.Classification.Metrics.AucRoc/tree/main/sample/ML.NET.Classifier.Binary) will provide application-specific instructions showing:

- How to install the AucRoc package
- Where scores and labels are prepared
- How Logistic Regression and Averaged Perceptron are handled
- Where `CoreMetricsEvaluator.EvaluateRoc` is called
- How AUC is displayed in the performance metrics panel
- How ROC points are displayed on the graph

## 📦 Libraries

- .NET 8.0
- ML.NET
- Windows x64 native C++ library

## 🛠️ Requirements

- .NET 8.0 SDK or newer
- Windows x64
- Visual Studio 2022 or VS Code with C# Dev Kit
- Binary classification results containing aligned scores and labels

## License

This project is licensed under the MIT License.
