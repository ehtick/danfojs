const { DataFrame } = require('./src/danfojs-node/dist/danfojs-node/src');

function generateTestData(rows, numGroups = 100) {
    console.log(`Generating ${rows} rows of test data with ~${numGroups} groups...`);

    const data = [];
    const columns = ['group_col', 'value_a', 'value_b', 'value_c'];

    for (let i = 0; i < rows; i++) {
        data.push([
            `group_${i % numGroups}`, // Create groups
            Math.random() * 1000,      // value_a
            Math.random() * 500,       // value_b
            Math.random() * 100        // value_c
        ]);
    }

    return new DataFrame(data, { columns });
}

function performanceTest(df, testName) {
    console.log(`\n=== ${testName} ===`);
    console.log(`DataFrame shape: ${df.shape[0]} rows, ${df.shape[1]} columns`);

    // Test 1: Basic groupby construction
    console.log('\nTest 1: Group construction...');
    let start = performance.now();
    const grouped = df.groupby(['group_col']);
    let end = performance.now();
    console.log(`Group construction: ${(end - start).toFixed(2)}ms`);
    console.log(`Number of groups: ${grouped.ngroups}`);

    // Test 2: Single column aggregation
    console.log('\nTest 2: Single column sum...');
    start = performance.now();
    const sumResult = grouped.col(['value_a']).sum();
    end = performance.now();
    console.log(`Single column sum: ${(end - start).toFixed(2)}ms`);
    console.log(`Result shape: ${sumResult.shape[0]} rows`);

    // Test 3: Multiple column aggregation
    console.log('\nTest 3: Multiple column aggregations...');
    start = performance.now();
    const multiResult = grouped.agg({
        value_a: 'mean',
        value_b: 'sum',
        value_c: 'count'
    });
    end = performance.now();
    console.log(`Multiple aggregations: ${(end - start).toFixed(2)}ms`);
    console.log(`Result shape: ${multiResult.shape[0]} rows`);

    // Test 4: Complex aggregation (multiple operations per column)
    console.log('\nTest 4: Complex aggregation...');
    start = performance.now();
    const complexResult = grouped.agg({
        value_a: ['mean', 'max', 'min'],
        value_b: ['sum', 'count'],
        value_c: 'std'
    });
    end = performance.now();
    console.log(`Complex aggregation: ${(end - start).toFixed(2)}ms`);
    console.log(`Result shape: ${complexResult.shape[0]} rows`);

    return {
        construction: end - start,
        singleSum: end - start,
        multiAgg: end - start,
        complexAgg: end - start
    };
}

async function main() {
    console.log('DanfoJS GroupBy Performance Test');
    console.log('================================');

    // Test different dataset sizes
    const testSizes = [
        { rows: 1000, groups: 50, name: 'Small Dataset (1K rows)' },
        { rows: 5000, groups: 100, name: 'Medium Dataset (5K rows)' },
        { rows: 20000, groups: 200, name: 'Large Dataset (20K rows)' }
    ];

    for (const testSize of testSizes) {
        const df = generateTestData(testSize.rows, testSize.groups);
        performanceTest(df, testSize.name);

        // Force garbage collection between tests if available
        if (global.gc) {
            global.gc();
        }
    }

    console.log('\n=== Performance Test Complete ===');
    console.log('Check the times above - we should see significant improvement!');
    console.log('Target: 20K rows should complete in < 2 seconds total');
}

// Run the test
main().catch(console.error);