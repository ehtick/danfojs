/**
*  @license
* Copyright 2022 JsData. All rights reserved.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==========================================================================
*/
import DataFrame from "../core/frame";
import { ArrayType1D, ArrayType2D } from "../shared/types";
import { variance, std, median, mode } from "mathjs";
import concat from "../transformers/concat";
import Series from "../core/series";

/**
 * The class performs all groupby operation on a dataframe
 * involving all aggregate funciton
 * @param {colDict} colDict Object of unique keys in the group by column
 * @param {keyCol} keyCol Array contains the column names
 * @param {data} Array the dataframe data
 * @param {columnName} Array of all column name in the dataframe.
 * @param {colDtype} Array columns dtype
 */
export default class Groupby {
  private _colDict: Map<string, { [key: string]: ArrayType1D }> = new Map();
  keyCol: ArrayType1D;
  data?: ArrayType2D | null;
  columnName: ArrayType1D;
  colDtype: ArrayType1D;
  colIndex: ArrayType1D;
  groupDict?: any;
  groupColNames?: Array<string>;
  keyToValue: Map<string, ArrayType1D> = new Map();
  // Cache for optimized key generation
  private keyGeneratorCache: Map<string, (values: ArrayType1D) => string> =
    new Map();

  constructor(
    keyCol: ArrayType1D,
    data: ArrayType2D | null,
    columnName: ArrayType1D,
    colDtype: ArrayType1D,
    colIndex: ArrayType1D
  ) {
    this.keyCol = keyCol;
    this.data = data;
    this.columnName = columnName;
    //this.dataTensors = {}; //store the tensor version of the groupby data
    this.colDtype = colDtype;
    this.colIndex = colIndex;
  }

  /**
   * Generate optimized key generation function based on column types
   */
  private getKeyGenerator(): (values: ArrayType1D) => string {
    const cacheKey = this.colIndex.join("|");

    if (this.keyGeneratorCache.has(cacheKey)) {
      return this.keyGeneratorCache.get(cacheKey)!;
    }

    // Analyze column types to determine best key generation strategy
    let allNumeric = true;
    let allInteger = true;

    for (let i = 0; i < this.colIndex.length; i++) {
      const colIdx = this.colIndex[i] as number;
      const dtype = this.colDtype[colIdx];
      if (dtype === "string") {
        allNumeric = false;
        allInteger = false;
        break;
      }
      // Check if it's integer-like
      if (dtype === "float32" || dtype === "float64") {
        allInteger = false;
      }
    }

    let keyGenerator: (values: ArrayType1D) => string;

    if (allInteger && this.colIndex.length === 1) {
      // Single integer column - fastest path
      keyGenerator = (values: ArrayType1D) => String(values[0]);
    } else if (allNumeric && this.colIndex.length === 1) {
      // Single numeric column
      keyGenerator = (values: ArrayType1D) => String(values[0]);
    } else if (allInteger) {
      // Multiple integer columns - use custom concatenation
      keyGenerator = (values: ArrayType1D) => {
        let result = String(values[0]);
        for (let i = 1; i < values.length; i++) {
          result += "-" + String(values[i]);
        }
        return result;
      };
    } else if (allNumeric) {
      // Multiple numeric columns
      keyGenerator = (values: ArrayType1D) => {
        let result = String(values[0]);
        for (let i = 1; i < values.length; i++) {
          result += "-" + String(values[i]);
        }
        return result;
      };
    } else {
      // Mixed types - fall back to join (but with pre-converted strings)
      keyGenerator = (values: ArrayType1D) => {
        const stringValues = new Array(values.length);
        for (let i = 0; i < values.length; i++) {
          stringValues[i] = String(values[i]);
        }
        return stringValues.join("-");
      };
    }

    this.keyGeneratorCache.set(cacheKey, keyGenerator);
    return keyGenerator;
  }

  /**
   * Generate group object data needed for group operations
   * let data = [ [ 1, 2, 3 ], [ 4, 5, 6 ], [ 20, 30, 40 ], [ 39, 89, 78 ] ];
   * let cols = [ "A", "B", "C" ];
   * let df = new dfd.DataFrame(data, { columns: cols });
   * let groupDf = df.groupby([ "A" ]);
   * The following internal object is generated and save to this.colDict
   * {
   *  '1': { A: [ 1 ], B: [ 2 ], C: [ 3 ] },
   *  '4': { A: [ 4 ], B: [ 5 ], C: [ 6 ] },
   *  '20': { A: [ 20 ], B: [ 30 ], C: [ 40 ] },
   *  '39': { A: [ 39 ], B: [ 89 ], C: [ 78 ] }
   * }
   * Since for groupby using more than one columns is index via '-'
   * e.g for df.groupby(['A','B'])
   * the result will look like this
   * {
   *  '1-2': {A: [ 1 ], B: [ 2 ], C: [ 3 ]},
   *  '4-5': {A: [ 4 ], B: [ 5 ], C: [ 6 ]}
   * }
   * but in doing analysis on a specific column like this
   * df.groupby(['A','B']).col(['C'])
   * will have the following set of internal result
   * {
   *  '1-2': { C: [ 3 ]},
   *  '4-5': {C: [ 6 ]}
   * }
   * In building our multindex type of DataFrame for this data,
   * we've somehow loose track of value for column A and B.
   * This could actually be generated by using split('-') on the object keys
   * e.g '1-2'.split('-') will give us the value for A and B.
   * But we might have weird case scenerio where A and B value has '-`
   * e.g
   * {
   *  '1--2-': { C: [ 3 ]},
   *  '4--5-': {C: [ 6 ]}
   * }
   * using `.split('-') might not work well
   * Hence we create a key-value `keyToValue` object to store index and their
   * associated value
   * NOTE: In the previous implementation we made use of Graph representation
   * for the group by data and Depth First search (DFS). But we decided to use key-value
   * object in javascript as an hashmap to reduce search time compared to using Grpah and DFS
   */
  group(): Groupby {
    const self = this;

    // Guard clause: if data is null or undefined, return early
    if (!this.data) {
      return this;
    }

    // Pre-compute column indices for faster access
    const colIndices = this.colIndex as number[];
    const columnNames = this.columnName as string[];
    const keyGenerator = this.getKeyGenerator();

    this.data.forEach((current) => {
      // Extract group key values more efficiently
      const keyValues: ArrayType1D = [];
      for (let i = 0; i < colIndices.length; i++) {
        keyValues.push(current[colIndices[i]]);
      }

      // Use optimized key generation
      const keyString = keyGenerator(keyValues);

      // Cache key-to-value mapping only once
      if (!this.keyToValue.has(keyString)) {
        this.keyToValue.set(keyString, keyValues);
      }

      // Get or create group data
      let groupData = this._colDict.get(keyString);
      if (groupData) {
        // Add to existing group - direct array access
        for (let i = 0; i < columnNames.length; i++) {
          groupData[columnNames[i]].push(current[i]);
        }
      } else {
        // Create new group
        groupData = {};
        for (let i = 0; i < columnNames.length; i++) {
          groupData[columnNames[i]] = [current[i]];
        }
        this._colDict.set(keyString, groupData);
      }
    });

    return this;
  }

  /**
   * Generate new internal groupby data
   * group = df.groupby(['A', 'B']).col('C')
   * This filter the colDict property as generated by `.group()`
   * it filter each group to contain only column `C` in their internal object
   * e.g
   * {
   *  '1-2': {A: [ 1 ], B: [ 2 ], C: [ 3 ]},
   *  '4-5': {A: [ 4 ], B: [ 5 ], C: [ 6 ]}
   * }
   * to
   * {
   *  '1-2': { C: [ 3 ]},
   *  '4-5': {C: [ 6 ]}
   * }
   * @param colNames column names
   * @return Groupby
   */
  col(colNames: ArrayType1D | undefined): Groupby {
    if (typeof colNames === "undefined") {
      colNames = this.columnName.filter((_, index) => {
        return !this.colIndex.includes(index);
      });
    }

    // Validate column names
    const colNamesArray = colNames as string[];
    for (const colName of colNamesArray) {
      if (!this.columnName.includes(colName))
        throw new Error(`Column ${colName} does not exist in groups`);
    }

    // Create new Map with filtered columns (avoid deep copying)
    const newColDict = new Map<string, { [key: string]: ArrayType1D }>();

    for (const [key, values] of Array.from(this._colDict.entries())) {
      const filteredData: { [key: string]: ArrayType1D } = {};
      for (const colName of colNamesArray) {
        filteredData[colName] = values[colName];
      }
      newColDict.set(key, filteredData);
    }

    const gp = new Groupby(
      this.keyCol,
      null,
      this.columnName,
      this.colDtype,
      this.colIndex
    );
    gp._colDict = newColDict;
    gp.groupColNames = colNamesArray;
    gp.keyToValue = this.keyToValue;

    return gp;
  }

  /**
   * Perform all groupby arithmetic operations
   * In the previous implementation all groups data are
   * stord as DataFrame, which involve lot of memory usage
   * Hence each groups are just pure javascrit object
   * and all arithmetic operation is done directly on javascript
   * arrays.
   * e.g
   * using this internal data
   * {
   *  '1-2': {A: [ 1,3 ], B: [ 2,5 ], C: [ 3, 5 ]},
   *  '4-5': {A: [ 4,1 ], B: [ 5,0 ], C: [ 6, 12 ]}
   * }
   * 1) using groupby(['A', 'B']).arithmetic("mean")
   * result: * {
   *  '1-2': {A_mean: [ 2 ], B_mean: [ 3.5 ], C_mean: [ 4 ]},
   *  '4-5': {A_mean: [ 2.5 ], B: [ 2.5 ], C_mean: [ 9 ]}
   * }
   * 2) .arithmetic({
   *    A: 'mean',
   *    B: 'sum',
   *    C: 'min'
   * })
   * result:
   * {
   *  '1-2': {A_mean: [ 2 ], B_sum: [ 7 ], C_min: [ 3 ]},
   *  '4-5': {A_mean: [ 2.5 ], B_sum: [ 5 ], C_min: [ 6 ]}
   * }
   * 3) .arithmetic({
   *    A: 'mean',
   *    B: 'sum',
   *    C: ['min', 'max']
   * })
   * result:
   * {
   *  '1-2': {A_mean: [ 2 ], B_sum: [ 7 ], C_min: [ 3 ], C_max: [5]},
   *  '4-5': {A_mean: [ 2.5 ], B_sum: [ 5 ], C_min: [ 6 ], C_max: [12]}
   * }
   * @param operation
   */
  private arithemetic(
    operation: { [key: string]: Array<string> | string } | string
  ): Map<string, { [key: string]: Array<number> }> {
    const opsName = [
      "mean",
      "sum",
      "count",
      "mode",
      "std",
      "var",
      "cumsum",
      "cumprod",
      "cummax",
      "cummin",
      "median",
      "min",
      "max",
    ];

    // Validate operations
    if (typeof operation === "string") {
      if (!opsName.includes(operation)) {
        throw new Error(`group operation: ${operation} is not valid`);
      }
    } else {
      Object.keys(operation).forEach((key) => {
        let ops = operation[key];
        if (Array.isArray(ops)) {
          for (let op of ops) {
            if (!opsName.includes(op)) {
              throw new Error(
                `group operation: ${op} for column ${key} is not valid`
              );
            }
          }
        } else {
          if (!opsName.includes(ops)) {
            throw new Error(
              `group operation: ${ops} for column ${key} is not valid`
            );
          }
        }
      });
    }

    const resultMap = new Map<string, { [key: string]: Array<number> }>();
    const groupColNames: Array<string> = this.groupColNames as Array<string>;

    for (const [key, values] of Array.from(this._colDict.entries())) {
      const colVal: { [key: string]: Array<number> } = {};

      for (let colKey = 0; colKey < groupColNames.length; colKey++) {
        const colName = groupColNames[colKey];
        const colIndex = this.columnName.indexOf(colName);
        const colDtype = this.colDtype[colIndex];
        const operationVal =
          typeof operation === "string" ? operation : operation[colName];

        if (colDtype === "string" && operationVal !== "count") {
          throw new Error(`Can't perform math operation on column ${colName}`);
        }

        if (typeof operation === "string") {
          const colName2 = `${colName}_${operation}`;
          colVal[colName2] = this.singleMathOperation(
            values[colName] as Array<number>,
            operation
          );
        } else {
          if (Array.isArray(operation[colName])) {
            // Use multi-pass aggregation for multiple operations on same column
            const operations = operation[colName] as string[];
            const results = this.multiPassAggregation(
              operations,
              values[colName] as Array<number>
            );

            for (const ops of operations) {
              const colName2 = `${colName}_${ops}`;
              colVal[colName2] = results[ops];
            }
          } else {
            const ops: string = operation[colName] as string;
            const colName2 = `${colName}_${ops}`;
            colVal[colName2] = this.singleMathOperation(
              values[colName] as Array<number>,
              ops
            );
          }
        }
      }
      resultMap.set(key, colVal);
    }
    return resultMap;
  }

  /**
   * Convert array to typed array for better performance on numeric operations
   */
  private optimizeNumericArray(
    colVal: Array<number>
  ): Float64Array | Array<number> {
    // Use typed arrays for pure numeric data to improve performance
    try {
      // Check if all values are numeric
      let allNumeric = true;
      for (let i = 0; i < colVal.length && allNumeric; i++) {
        if (typeof colVal[i] !== "number" || !isFinite(colVal[i])) {
          allNumeric = false;
        }
      }

      if (allNumeric && colVal.length > 10) {
        // Only use for larger arrays
        return new Float64Array(colVal);
      }
    } catch (e) {
      // Fall back to regular array if typed array creation fails
    }

    return colVal;
  }

  /**
   * Optimized math operations for typed arrays
   */
  private fastMathOperations = {
    sum: (arr: Float64Array | Array<number>): number => {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
      }
      return sum;
    },

    min: (arr: Float64Array | Array<number>): number => {
      let min = arr[0];
      for (let i = 1; i < arr.length; i++) {
        if (arr[i] < min) min = arr[i];
      }
      return min;
    },

    max: (arr: Float64Array | Array<number>): number => {
      let max = arr[0];
      for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
      }
      return max;
    },

    mean: (arr: Float64Array | Array<number>): number => {
      return this.fastMathOperations.sum(arr) / arr.length;
    },
  };

  /**
   * Single-pass multi-aggregation for maximum performance
   * Computes multiple operations in one pass through the data
   */
  private multiPassAggregation(
    operations: string[],
    colVal: Array<number>
  ): { [key: string]: Array<number> } {
    const results: { [key: string]: Array<number> } = {};
    const needsSum = operations.includes("sum") || operations.includes("mean");
    const needsMinMax =
      operations.includes("min") || operations.includes("max");
    const needsCumulative = operations.some((op) => op.startsWith("cum"));

    // Optimize array for numeric operations
    const optimizedArray = this.optimizeNumericArray(colVal);
    const length = optimizedArray.length;

    // Use optimized operations for basic aggregations
    let sum: number | undefined;
    let min: number | undefined;
    let max: number | undefined;

    if (needsSum) {
      sum = this.fastMathOperations.sum(optimizedArray);
    }
    if (needsMinMax) {
      min = this.fastMathOperations.min(optimizedArray);
      max = this.fastMathOperations.max(optimizedArray);
    }

    // Assign results for basic operations
    for (const op of operations) {
      switch (op) {
        case "sum":
          results[op] = [sum!];
          break;
        case "count":
          results[op] = [length];
          break;
        case "mean":
          results[op] = [sum! / length];
          break;
        case "min":
          results[op] = [min!];
          break;
        case "max":
          results[op] = [max!];
          break;
        case "std":
          results[op] = [std(colVal)];
          break;
        case "var":
          results[op] = [variance(colVal)];
          break;
        case "median":
          results[op] = [median(colVal)];
          break;
        case "mode":
          results[op] = [mode(colVal)];
          break;
      }
    }

    // Handle cumulative operations separately (they need arrays)
    for (const op of operations) {
      if (op.startsWith("cum")) {
        results[op] = this.singleMathOperation(colVal, op);
      }
    }

    return results;
  }

  /**
   * Single operation computation (fallback for individual operations)
   */
  private singleMathOperation(
    colVal: Array<number>,
    op: string
  ): Array<number> {
    // Use optimized operations for basic math when possible
    const optimizedArray = this.optimizeNumericArray(colVal);

    switch (op) {
      case "sum":
        return [this.fastMathOperations.sum(optimizedArray)];
      case "mean":
        return [this.fastMathOperations.mean(optimizedArray)];
      case "min":
        return [this.fastMathOperations.min(optimizedArray)];
      case "max":
        return [this.fastMathOperations.max(optimizedArray)];
      case "count":
        return [optimizedArray.length];
      default:
        // Fall back to original implementation for complex operations
        const operation =
          Groupby.mathOperations[op as keyof typeof Groupby.mathOperations];
        return operation ? operation(colVal) : [];
    }
  }

  // Function lookup table for arithmetic operations (better performance than switch)
  private static readonly mathOperations = {
    max: (colVal: Array<number>): Array<number> => {
      let max = colVal[0];
      for (let i = 1; i < colVal.length; i++) {
        if (colVal[i] > max) max = colVal[i];
      }
      return [max];
    },
    min: (colVal: Array<number>): Array<number> => {
      let min = colVal[0];
      for (let i = 1; i < colVal.length; i++) {
        if (colVal[i] < min) min = colVal[i];
      }
      return [min];
    },
    sum: (colVal: Array<number>): Array<number> => {
      let sum = 0;
      for (let i = 0; i < colVal.length; i++) {
        sum += colVal[i];
      }
      return [sum];
    },
    count: (colVal: Array<number>): Array<number> => [colVal.length],
    mean: (colVal: Array<number>): Array<number> => {
      let sum = 0;
      for (let i = 0; i < colVal.length; i++) {
        sum += colVal[i];
      }
      return [sum / colVal.length];
    },
    std: (colVal: Array<number>): Array<number> => [std(colVal)],
    var: (colVal: Array<number>): Array<number> => [variance(colVal)],
    median: (colVal: Array<number>): Array<number> => [median(colVal)],
    mode: (colVal: Array<number>): Array<number> => [mode(colVal)],
    cumsum: (colVal: Array<number>): Array<number> => {
      const data: Array<number> = [];
      let sum = 0;
      for (let i = 0; i < colVal.length; i++) {
        sum += colVal[i];
        data.push(sum);
      }
      return data;
    },
    cummin: (colVal: Array<number>): Array<number> => {
      const data: Array<number> = [colVal[0]];
      let min = colVal[0];
      for (let i = 1; i < colVal.length; i++) {
        if (colVal[i] < min) min = colVal[i];
        data.push(min);
      }
      return data;
    },
    cummax: (colVal: Array<number>): Array<number> => {
      const data: Array<number> = [colVal[0]];
      let max = colVal[0];
      for (let i = 1; i < colVal.length; i++) {
        if (colVal[i] > max) max = colVal[i];
        data.push(max);
      }
      return data;
    },
    cumprod: (colVal: Array<number>): Array<number> => {
      const data: Array<number> = [];
      let prod = 1;
      for (let i = 0; i < colVal.length; i++) {
        prod *= colVal[i];
        data.push(prod);
      }
      return data;
    },
  };

  /**
   * Peform all arithmetic logic (legacy method - use singleMathOperation instead)
   * @param colVal
   * @param ops
   */
  private groupMathLog(colVal: Array<number>, ops: string): Array<number> {
    return this.singleMathOperation(colVal, ops);
  }

  /**
   * Takes in internal groupby internal data and convert
   * them to a single data frame.
   * @param colDict
   */
  private toDataFrame(
    colDict: Map<string, { [key: string]: ArrayType1D }>
  ): DataFrame {
    const data: { [key: string]: ArrayType1D } = {};
    const keys = this.colKeyDict(colDict);

    // Handle empty case - return empty DataFrame with proper column structure
    if (keys.length === 0) {
      const columns: string[] = [];
      // Add key column names
      for (let keyIdx = 0; keyIdx < this.keyCol.length; keyIdx++) {
        const keyName = this.keyCol[keyIdx] as string;
        columns.push(keyName);
        data[keyName] = [];
      }
      // Add group column names if they exist
      if (this.groupColNames) {
        for (const colName of this.groupColNames) {
          columns.push(colName);
          data[colName] = [];
        }
      }
      return new DataFrame([], { columns });
    }

    // Initialize data structure more efficiently
    let isFirstGroup = true;

    for (const key of keys) {
      const value = colDict.get(key)!;
      const valueEntries = Object.entries(value);
      const oneValue = valueEntries[0][1] as ArrayType1D;
      const valueLen = oneValue.length;

      if (isFirstGroup) {
        // Initialize arrays for the first group
        // Add key columns with pre-allocated arrays (faster than Array.fill)
        for (let keyIdx = 0; keyIdx < this.keyCol.length; keyIdx++) {
          const keyName = this.keyCol[keyIdx] as string;
          const keyValue = this.keyToValue.get(key)![keyIdx];
          const keyArray = new Array(valueLen);
          for (let i = 0; i < valueLen; i++) {
            keyArray[i] = keyValue;
          }
          data[keyName] = keyArray;
        }

        // Add value columns
        for (const [colName, colValues] of valueEntries) {
          data[colName] = [...colValues];
        }
        isFirstGroup = false;
      } else {
        // Append to existing arrays using batch operations
        // Add key columns with optimized batch assignment
        for (let keyIdx = 0; keyIdx < this.keyCol.length; keyIdx++) {
          const keyName = this.keyCol[keyIdx] as string;
          const keyValue = this.keyToValue.get(key)![keyIdx];
          const existingArray = data[keyName] as any[];
          const startIndex = existingArray.length;

          // Extend array length once, then assign directly
          existingArray.length += valueLen;
          for (let i = 0; i < valueLen; i++) {
            existingArray[startIndex + i] = keyValue;
          }
        }

        // Add value columns with optimized batch copying
        for (const [colName, colValues] of valueEntries) {
          const existingArray = data[colName] as any[];
          const startIndex = existingArray.length;

          // Extend array length once, then copy directly
          existingArray.length += colValues.length;
          for (let i = 0; i < colValues.length; i++) {
            existingArray[startIndex + i] = colValues[i];
          }
        }
      }
    }

    return new DataFrame(data);
  }

  private operations(ops: string): DataFrame {
    // Handle empty case early
    if (this._colDict.size === 0) {
      const columns: string[] = [];
      // Add key column names
      for (let keyIdx = 0; keyIdx < this.keyCol.length; keyIdx++) {
        const keyName = this.keyCol[keyIdx] as string;
        columns.push(keyName);
      }
      // Add result column names
      const targetColumns =
        this.groupColNames ||
        this.columnName.filter((_, index) => !this.colIndex.includes(index));
      for (const colName of targetColumns) {
        columns.push(`${colName}_${ops}`);
      }
      return new DataFrame([], { columns });
    }

    if (!this.groupColNames) {
      let colGroup = this.col(undefined);
      let colDict = colGroup.arithemetic(ops);
      let df = colGroup.toDataFrame(colDict);
      return df;
    }
    let colDict = this.arithemetic(ops);
    let df = this.toDataFrame(colDict);
    return df;
  }

  /**
   * Obtain the count for each group
   * @returns DataFrame
   *
   */
  count(): DataFrame {
    return this.operations("count");
  }

  /**
   * Obtain the sum of columns for each group
   * @returns DataFrame
   *
   */
  sum(): DataFrame {
    return this.operations("sum");
  }

  /**
   * Obtain the standard deviation of columns for each group
   * @returns DataFrame
   */
  std(): DataFrame {
    return this.operations("std");
  }

  /**
   * Obtain the variance of columns for each group
   * @returns DataFrame
   */
  var(): DataFrame {
    return this.operations("var");
  }

  /**
   * Obtain the mean of columns for each group
   * @returns DataFrame
   */
  mean(): DataFrame {
    return this.operations("mean");
  }

  /**
   * Obtain the cumsum of columns for each group
   * @returns DataFrame
   *
   */
  cumSum(): DataFrame {
    return this.operations("cumsum");
  }

  /**
   * Obtain the cummax of columns for each group
   * @returns DataFrame
   */
  cumMax(): DataFrame {
    return this.operations("cummax");
  }

  /**
   * Obtain the cumprod of columns for each group
   * @returns DataFrame
   */
  cumProd(): DataFrame {
    return this.operations("cumprod");
  }

  /**
   * Obtain the cummin of columns for each group
   * @returns DataFrame
   */
  cumMin(): DataFrame {
    return this.operations("cummin");
  }

  /**
   * Obtain the max value of columns for each group
   * @returns DataFrame
   *
   */
  max(): DataFrame {
    return this.operations("max");
  }

  /**
   * Obtain the min of columns for each group
   * @returns DataFrame
   */
  min(): DataFrame {
    return this.operations("min");
  }

  /**
   * Obtain a specific group
   * @param keys Array<string | number>
   * @returns DataFrame
   */
  getGroup(keys: Array<string | number>): DataFrame {
    const dictKey = keys.join("-");
    const colDict = new Map<string, { [key: string]: ArrayType1D }>();
    const groupData = this._colDict.get(dictKey);
    if (groupData) {
      colDict.set(dictKey, groupData);
    }
    return this.toDataFrame(colDict);
  }

  /**
   * Perform aggregation on all groups
   * @param ops
   * @returns DataFrame
   */
  agg(ops: { [key: string]: Array<string> | string }): DataFrame {
    // Handle empty case early
    if (this._colDict.size === 0) {
      const columns: string[] = [];
      // Add key column names
      for (let keyIdx = 0; keyIdx < this.keyCol.length; keyIdx++) {
        const keyName = this.keyCol[keyIdx] as string;
        columns.push(keyName);
      }
      // Add result column names for each operation
      for (const [colName, operations] of Object.entries(ops)) {
        if (Array.isArray(operations)) {
          for (const op of operations) {
            columns.push(`${colName}_${op}`);
          }
        } else {
          columns.push(`${colName}_${operations}`);
        }
      }
      return new DataFrame([], { columns });
    }

    let columns = Object.keys(ops);
    let col_gp = this.col(columns);
    let data = col_gp.arithemetic(ops);
    let df = col_gp.toDataFrame(data);
    return df;
  }

  /**
   * Apply custom aggregator function
   * to each group
   * @param callable
   * @returns DataFrame
   * @example
   * let grp = df.groupby(['A'])
   * grp.apply((x) => x.count())
   */
  apply(callable: (x: DataFrame) => DataFrame | Series): DataFrame {
    const colDict: { [key: string]: DataFrame | Series } = {};
    const keys = this.colKeyDict(this._colDict);

    for (const key of keys) {
      const groupData = this._colDict.get(key)!;
      const valDataframe = new DataFrame(groupData);
      colDict[key] = callable(valDataframe);
    }
    return this.concatGroups(colDict);
  }

  private concatGroups(colDict: {
    [key: string]: DataFrame | Series;
  }): DataFrame {
    let data: Array<DataFrame | Series> = [];
    for (const [key, values] of Object.entries(colDict)) {
      let copyDf: DataFrame;
      if (values instanceof DataFrame) {
        copyDf = values.copy();
      } else {
        let columns = values.index as string[];
        columns = columns.length > 1 ? columns : ["applyOps"];
        copyDf = new DataFrame([values.values], { columns: columns });
      }
      let len = copyDf.shape[0];
      const keyValues = this.keyToValue.get(key)!;
      for (let keyIdx = 0; keyIdx < this.keyCol.length; keyIdx++) {
        const keyName = this.keyCol[keyIdx] as string;
        const keyValue = keyValues[keyIdx];
        // Use pre-allocated array instead of Array.fill()
        const dfValue = new Array(len);
        for (let i = 0; i < len; i++) {
          dfValue[i] = keyValue;
        }

        if (this.groupColNames) {
          copyDf.addColumn(keyName, dfValue, {
            inplace: true,
            atIndex: keyIdx,
          });
        } else {
          copyDf.addColumn(`${keyName}_Group`, dfValue, {
            inplace: true,
            atIndex: keyIdx,
          });
        }
      }
      data.push(copyDf);
    }
    return concat({ dfList: data, axis: 0 }) as DataFrame;
  }

  /**
   * obtain the total number of groups
   * @returns number
   */
  get ngroups(): number {
    return this._colDict.size;
  }

  /**
   * obtaind the internal group data
   * @returns { [key: string]: { [key: string]: ArrayType1D } } (backward compatibility)
   */
  get groups(): { [key: string]: { [key: string]: ArrayType1D } } {
    // Ensure grouping has been done
    if (this._colDict.size === 0) {
      this.group();
    }
    // Convert Map to object for backward compatibility
    const result: { [key: string]: { [key: string]: ArrayType1D } } = {};
    Array.from(this._colDict.entries()).forEach(([key, value]) => {
      result[key] = value;
    });
    return result;
  }

  /**
   * Backward compatibility for colDict property access
   * @returns { [key: string]: { [key: string]: ArrayType1D } }
   */
  get colDict(): { [key: string]: { [key: string]: ArrayType1D } } {
    return this.groups;
  }

  /**
   * Obtain the first row of each group
   * @returns DataFrame
   */
  first(): DataFrame {
    return this.apply((x) => {
      return x.head(1);
    });
  }

  /**
   * Obtain the last row of each group
   * @returns DataFrame
   */
  last(): DataFrame {
    return this.apply((x) => {
      return x.tail(1);
    });
  }

  /**
   * Obtains the dataframe se of each groups
   * @returns DataFrame
   */
  size(): DataFrame {
    return this.apply((x) => {
      return new Series([x.shape[0]]);
    });
  }

  private colKeyDict(
    colDict: Map<string, { [key: string]: ArrayType1D }>
  ): string[] {
    const keyDict: { [key: string]: string[] } = {};
    const firstKeyOrder: string[] = [];

    // Collect keys and group by first key, preserving insertion order
    for (const key of Array.from(colDict.keys())) {
      const firstKey = key.split("-")[0];
      if (firstKey in keyDict) {
        keyDict[firstKey].push(key);
      } else {
        keyDict[firstKey] = [key];
        firstKeyOrder.push(firstKey);
      }
    }

    // Preserve first key appearance order (don't sort alphabetically)
    const sortedFirstKeys = firstKeyOrder;
    const keys: string[] = [];
    for (const firstKey of sortedFirstKeys) {
      // Preserve insertion order within each group
      keys.push(...keyDict[firstKey]);
    }
    return keys;
  }
}
