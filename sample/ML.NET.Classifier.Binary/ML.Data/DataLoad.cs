using System;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;
using CsvHelper;
using CsvHelper.Configuration;

namespace ML.Data
{
    /// <summary>
    /// loads CSV data into a DataTable and calculates the number of input features
    /// </summary>
    public class DataLoad : IDataLoad
    {
        /// <summary>
        /// loads data from a CSV file and returns the DataTable and feature count
        /// </summary>
        public (DataTable dataTable, int featureCount) LoadData(string filePath)
        {
            // check if file path is valid
            if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
                throw new ArgumentException(
                    "File path does not exist.",
                    nameof(filePath));

            var dataTable = new DataTable();

            // configure CSV reader
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                Delimiter = ",",
                HasHeaderRecord = true,
                IgnoreBlankLines = true,
                BadDataFound = null,
                HeaderValidated = null,
                MissingFieldFound = null,
                TrimOptions = TrimOptions.Trim,
                PrepareHeaderForMatch = h => h.Header?.Trim()
            };

            try
            {
                using var reader = new StreamReader(filePath);
                using var csv = new CsvReader(reader, config);
                using var dataReader = new CsvDataReader(csv);

                dataTable.Load(dataReader);
            }
            catch (HeaderValidationException ex)
            {
                throw new Exception(
                    "CSV header mismatch – duplicate or missing column.",
                    ex);
            }

            // check if dataset contains data
            if (dataTable.Rows.Count == 0 || dataTable.Columns.Count == 0)
                throw new Exception("Loaded data table is empty.");

            // find binary label column
            var checker = new CheckCharacteristic();
            string labelColumn = checker.FindBinaryColumn(dataTable);

            if (labelColumn == null)
                throw new Exception(
                    "No valid binary label column was found.");

            // count input features, excluding the label
            int featureCount = dataTable.Columns
                .Cast<DataColumn>()
                .Count(column => !column.ColumnName.Equals(
                    labelColumn,
                    StringComparison.OrdinalIgnoreCase));

            return (dataTable, featureCount);
        }
    }
}