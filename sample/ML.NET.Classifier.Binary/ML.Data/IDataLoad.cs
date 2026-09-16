using System.Data;

namespace ML.Data
{
    /// <summary>
    /// provides a method for loading CSV data
    /// </summary>
    public interface IDataLoad
    {
        (DataTable dataTable, int featureCount) LoadData(string filePath);
    }
}