using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace ML.Data
{
    /// <summary>
    /// analyzes a DataTable and finds a valid binary label column
    /// </summary>
    public class CheckCharacteristic : ICheckCharacteristic
    {
        /// <summary>
        /// finds a column containing exactly two binary values: 0 and 1, or true and false
        /// </summary>
        public string FindBinaryColumn(DataTable table)
        {
            if (table == null || table.Rows.Count == 0)
                return null;

            foreach (DataColumn column in table.Columns)
            {
                // binary label should not contain null values
                if (table.AsEnumerable().Any(row => row.IsNull(column)))
                    continue;

                var distinct = new HashSet<object>();

                foreach (DataRow row in table.Rows)
                {
                    var value = row[column];
                    distinct.Add(value);

                    if (distinct.Count > 2)
                        break;
                }

                if (distinct.Count == 2 && distinct.All(IsBinaryValue))
                    return column.ColumnName;
            }

            return null;
        }

        /// <summary>
        /// checks whether a value can be used as a binary value
        /// </summary>
        private static bool IsBinaryValue(object value)
        {
            return value switch
            {
                bool => true,
                byte b => b is 0 or 1,
                short s => s is 0 or 1,
                int i => i is 0 or 1,
                long l => l is 0 or 1,
                string s => s.Trim() is "0" or "1" or "true" or "false",
                _ => false
            };
        }
    }
}