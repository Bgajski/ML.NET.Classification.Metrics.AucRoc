using System.Data;
using Microsoft.ML;
using ML.Data;
using ML.DataPreparation;
using ML.Model;
using ML.Performance;

namespace ML.Design
{
    public partial class FormInterface : Form
    {
        #region Fields

        private BinaryClassificationModel _binaryModel;
        private IDataView _trainingData;
        private IDataView _testData;
        private BinaryEvaluationResult? _evaluation;

        private readonly IDataLoad _dataLoad;

        private readonly IPerformanceDescription
            _performanceDescription;

        private readonly IModelDescription
            _modelDescription;

        private readonly IPerformanceVisualizer
            _performanceVisualizer;

        private readonly MLContext _mlContext;

        #endregion

        #region Constructor

        public FormInterface()
        {
            InitializeComponent();

            _mlContext = new MLContext(seed: 42);

            _dataLoad = new DataLoad();

            _performanceDescription =
                new PerformanceDescription();

            _modelDescription =
                new ModelDescription();

            _performanceVisualizer =
                new PerformanceVisualizer();

            InitializeEventHandlers();

            // ROC is displayed after training
            // on the results page
            HideModelPerformanceHeading(this);

            comboBox_graph_p2.Items.Clear();
            comboBox_graph_p2.Text = "";
        }

        #endregion

        #region Initialization

        private void HideModelPerformanceHeading(
            Control parent)
        {
            foreach (Control control in parent.Controls)
            {
                if (control is Label &&
                    string.Equals(
                        control.Text.Trim(),
                        "Model performance",
                        StringComparison.OrdinalIgnoreCase))
                {
                    control.Visible = false;
                }

                if (control.HasChildren)
                {
                    HideModelPerformanceHeading(control);
                }
            }
        }

        private void InitializeEventHandlers()
        {
            algorithmSelection_p2.SelectedIndexChanged -=
                algorithmSelection_p2_SelectedIndexChanged;

            algorithmSelection_p2.SelectedIndexChanged +=
                algorithmSelection_p2_SelectedIndexChanged;

            train_Button_p2.Click -=
                train_Button_p2_Click;

            train_Button_p2.Click +=
                train_Button_p2_Click;

            button_csv_reset.Click -=
                button_csv_reset_Click;

            button_csv_reset.Click +=
                button_csv_reset_Click;

            comboBox_graph_p2.SelectedIndexChanged -=
                comboBox_graph_p2_SelectedIndexChanged;

            comboBox_graph_p2.SelectedIndexChanged +=
                comboBox_graph_p2_SelectedIndexChanged;

            button_exit.Click -=
                button_exit_Click;

            button_exit.Click +=
                button_exit_Click;
        }

        #endregion

        #region Event Handlers

        private void button_csv_p1_Click(
            object sender,
            EventArgs e)
        {
            OpenFileDialog openFileDialog =
                new OpenFileDialog
                {
                    Filter =
                        "CSV files (*.csv)|*.csv|" +
                        "All files (*.*)|*.*",

                    RestoreDirectory = true
                };

            if (openFileDialog.ShowDialog() ==
                DialogResult.OK)
            {
                try
                {
                    HandleFileLoading(
                        openFileDialog.FileName);

                    txtBoxPerformanceMetric_p2.Clear();
                }
                catch (Exception ex)
                {
                    MessageBox.Show(
                        "Error loading data: " +
                        ex.Message);
                }
            }
        }

        private void button_csv_reset_Click(
            object sender,
            EventArgs e)
        {
            ResetForm();
            txtBoxPerformanceMetric_p2.Clear();
        }

        private void train_Button_p2_Click(
            object sender,
            EventArgs e)
        {
            if (_binaryModel == null ||
                _trainingData == null ||
                _testData == null)
            {
                MessageBox.Show(
                    "Load a binary classification " +
                    "dataset first.");

                return;
            }

            try
            {
                HandleModelTraining();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error training model: " +
                    ex.Message);
            }
        }

        private void comboBox_graph_p2_SelectedIndexChanged(
            object sender,
            EventArgs e)
        {
            UpdateGraphDisplay();
        }

        private void button_exit_Click(
            object sender,
            EventArgs e)
        {
            Application.Exit();
        }

        private void comboBox_graph_p1_SelectedIndexChanged(
            object sender,
            EventArgs e)
        {
        }

        private void algorithmSelection_p2_SelectedIndexChanged(
            object sender,
            EventArgs e)
        {
            if (algorithmSelection_p2.SelectedItem != null &&
                algorithmSelection_p2.SelectedItem.ToString() != "")
            {
                _evaluation = null;

                comboBox_graph_p2.Items.Clear();
                comboBox_graph_p2.Text = "";

                txtBoxPerformanceMetric_p2.Clear();
                txtBoxMetric_p2.Clear();

                dataChar_csv_p2.Series.Clear();
                dataChar_csv_p2.ChartAreas.Clear();
                dataChar_csv_p2.Titles.Clear();
                dataChar_csv_p2.Annotations.Clear();
                dataChar_csv_p2.Invalidate();
            }
        }

        #endregion

        #region Methods

        private void HandleFileLoading(
            string filePath)
        {
            ResetForm();

            var (dataTable, featureCount) =
                _dataLoad.LoadData(filePath);

            textBox_csv_p1.Text =
                Path.GetFileName(filePath);

            txtBoxMetric_p2.Text = "";
            txtBoxPerformanceMetric_p2.Text = "";

            comboBox_graph_p2.Items.Clear();
            comboBox_graph_p2.Text = "";

            dataChar_csv_p2.Series.Clear();
            dataChar_csv_p2.ChartAreas.Clear();
            dataChar_csv_p2.Titles.Clear();
            dataChar_csv_p2.Invalidate();

            algorithmSelection_p2.SelectedIndex = -1;
            algorithmSelection_p2.Text = "";

            _binaryModel = null;

            SplitDataForTraining(
                filePath,
                new BinaryDataPreparation(),
                featureCount);

            _binaryModel =
                new BinaryClassificationModel(
                    _mlContext);

            SetInitialTrainingOptions(
                "binary classification");

            ResetAdditionalChartAndTextbox();
        }

        private void HandleModelTraining()
        {
            _evaluation = null;

            txtBoxMetric_p2.Clear();
            txtBoxPerformanceMetric_p2.Clear();
            dataChar_csv_p2.Series.Clear();

            string algorithmName = "";

            TrainBinaryModel(
                algorithmSelection_p2.Text,
                ref algorithmName);

            BinaryEvaluationResult binaryMetrics =
                _binaryModel.EvaluateThresholdMetrics(
                    _testData);

            _evaluation = binaryMetrics;

            List<(string text, Color color)> descriptions =
                _modelDescription
                    .GetMetricsDescriptionBinary(
                        binaryMetrics);

            DisplayMetricsInColor(
                txtBoxMetric_p2,
                descriptions);

            PopulateGraphOptions();
        }

        private void TrainBinaryModel(
            string selectedAlgorithm,
            ref string algorithmName)
        {
            // Preparation already reserved the final test set
            // Validation is used only for threshold optimization
            var split = BinaryDataPreparation.SplitBinaryData(
                _mlContext,
                _trainingData,
                0.20,
                42,
                "Training/validation split");

            IDataView train = split.TrainSet;
            IDataView validation = split.TestSet;

            // Create a fresh seeded model so repeated training is reproducible
            _binaryModel =
                new BinaryClassificationModel(
                    new MLContext(seed: 42));

            switch (selectedAlgorithm)
            {
                case "Logistic regression":
                    _binaryModel.TrainLogisticRegression(train);
                    algorithmName = "LbfgsLogisticRegression";
                    break;

                case "Averaged Perceptron":
                    _binaryModel.TrainAveragedPerceptron(train);
                    algorithmName = "AveragedPerceptron";
                    break;

                default:
                    throw new InvalidOperationException(
                        $"Unrecognized algorithm: {selectedAlgorithm}");
            }

            _binaryModel.OptimizeThresholdByF1(validation);
        }

        private void PopulateGraphOptions()
        {
            comboBox_graph_p2.Items.Clear();

            comboBox_graph_p2.DropDownStyle =
                ComboBoxStyle.DropDownList;

            comboBox_graph_p2.Items.Add(
                "Receiver operating characteristic curve");

            comboBox_graph_p2.SelectedIndex = 0;
        }

        private void UpdateGraphDisplay()
        {
            if (_evaluation == null ||
                comboBox_graph_p2.SelectedItem?.ToString() !=
                "Receiver operating characteristic curve")
            {
                return;
            }

            IReadOnlyList<float> scores =
                _evaluation.Scores;

            IReadOnlyList<bool> labels =
                _evaluation.Labels;

            dataChar_csv_p2.Annotations.Clear();

            // The native package returns the plotted points
            // and their matching AUC
            double nativeAuc =
                _performanceVisualizer
                    .VisualizeReceiverCurve(
                        scores,
                        labels,
                        dataChar_csv_p2,
                        "Binary Classification - ROC Curve");

            // Reference check
            // The displayed value is still the AUC returned
            // by ML.NET.Classification.Metrics.AucRoc
            double mlNetAuc =
                _evaluation
                    .DefaultThresholdMetrics
                    .AreaUnderRocCurve;

            if (!double.IsFinite(mlNetAuc) ||
                Math.Abs(nativeAuc - mlNetAuc) > 1e-6)
            {
                throw new InvalidOperationException(
                    $"Native C++ AUC ({nativeAuc:G17}) " +
                    $"differs from ML.NET AUC " +
                    $"({mlNetAuc:G17}).");
            }

            List<(string text, Color color)> descriptions =
                _performanceDescription
                    .GetMetricsReceiverCurve(
                        nativeAuc);

            DisplayMetricsInColor(
                txtBoxPerformanceMetric_p2,
                descriptions);
        }

        private void SplitDataForTraining(
            string filePath,
            IDataPreparation dataPreparation,
            int featureCount)
        {
            if (dataPreparation is
                IBinaryDataPreparation binaryDataPreparation)
            {
                var (trainData, testData) =
                    binaryDataPreparation.PrepareData(
                        _mlContext,
                        filePath,
                        featureCount);

                _trainingData = trainData;
                _testData = testData;
            }
        }

        private void ResetForm()
        {
            textBox_csv_p1.Text = "";
            txtBoxMetric_p2.Text = "";
            txtBoxPerformanceMetric_p2.Text = "";

            comboBox_graph_p2.Items.Clear();
            comboBox_graph_p2.Text = "";

            dataChar_csv_p2.Series.Clear();
            dataChar_csv_p2.ChartAreas.Clear();
            dataChar_csv_p2.Titles.Clear();
            dataChar_csv_p2.Annotations.Clear();
            dataChar_csv_p2.Invalidate();

            algorithmSelection_p2.Items.Clear();
            algorithmSelection_p2.SelectedIndex = -1;
            algorithmSelection_p2.Text = "";

            _binaryModel = null;
            _evaluation = null;
            _trainingData = null;
            _testData = null;

            ResetAdditionalChartAndTextbox();
        }

        private void DisplayMetricsInColor(
            RichTextBox richTextBox,
            List<(string text, Color color)> descriptions)
        {
            richTextBox.Clear();

            foreach (var (text, color) in descriptions)
            {
                AppendText(
                    richTextBox,
                    text + "\n",
                    color);
            }
        }

        private void AppendText(
            RichTextBox box,
            string text,
            Color color)
        {
            box.SelectionStart =
                box.TextLength;

            box.SelectionLength = 0;
            box.SelectionColor = color;

            box.AppendText(text);

            box.SelectionColor =
                box.ForeColor;
        }

        private void ResetAdditionalChartAndTextbox()
        {
            dataChar_csv_p2.Series.Clear();
            dataChar_csv_p2.ChartAreas.Clear();
            dataChar_csv_p2.Titles.Clear();
            dataChar_csv_p2.Annotations.Clear();
            dataChar_csv_p2.Invalidate();

            txtBoxPerformanceMetric_p2.Clear();
        }

        private void SetInitialTrainingOptions(
            string classificationResult)
        {
            algorithmSelection_p2.Items.Clear();

            algorithmSelection_p2.Items.Add(
                "Logistic regression");

            algorithmSelection_p2.Items.Add(
                "Averaged Perceptron");

            algorithmSelection_p2.SelectedItem =
                "Logistic regression";
        }

        #endregion
    }
}