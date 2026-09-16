using System.Data;

namespace ML.Data
{
    /// <summary>
    /// provides methods for detecting binary classification columns
    /// </summary>
    public interface ICheckCharacteristic
    {
        string FindBinaryColumn(DataTable dataTable);
    }
}