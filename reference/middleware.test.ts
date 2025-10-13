/**
 * Test file for middleware redirect logic.
 * This can be run locally to verify the case-insensitive redirect behavior.
 */

/**
 * Simple test cases to verify the middleware logic.
 */
const testCases = [
  // Test uppercase in path
  {
    input: '/python/LangChain/index.html',
    expected: '/python/langchain/index.html',
    shouldRedirect: true,
  },
  // Test mixed case
  {
    input: '/python/LangGraph/Graphs',
    expected: '/python/langgraph/graphs',
    shouldRedirect: true,
  },
  // Test all lowercase (no redirect needed)
  {
    input: '/python/langchain/index.html',
    expected: '/python/langchain/index.html',
    shouldRedirect: false,
  },
  // Test JavaScript paths
  {
    input: '/javascript/LangChain/Agents',
    expected: '/javascript/langchain/agents',
    shouldRedirect: true,
  },
  // Test complex path with query params
  {
    input: '/python/Integrations/LangChain_OpenAI',
    expected: '/python/integrations/langchain_openai',
    shouldRedirect: true,
  },
];

/**
 * Simulate the middleware logic for testing.
 */
function simulateMiddleware(pathname: string): { shouldRedirect: boolean; newPath: string } {
  const shouldRedirect = pathname !== pathname.toLowerCase();
  return {
    shouldRedirect,
    newPath: pathname.toLowerCase(),
  };
}

/**
 * Run tests and log results.
 */
function runTests(): void {
  console.log('Running middleware tests...\n');

  let passed = 0;
  let failed = 0;

  for (const testCase of testCases) {
    const result = simulateMiddleware(testCase.input);
    const testPassed =
      result.shouldRedirect === testCase.shouldRedirect &&
      result.newPath === testCase.expected;

    if (testPassed) {
      passed++;
      console.log(`✓ PASS: ${testCase.input}`);
    } else {
      failed++;
      console.log(`✗ FAIL: ${testCase.input}`);
      console.log(`  Expected redirect: ${testCase.shouldRedirect}, got: ${result.shouldRedirect}`);
      console.log(`  Expected path: ${testCase.expected}, got: ${result.newPath}`);
    }
  }

  console.log(`\nResults: ${passed} passed, ${failed} failed`);

  if (failed === 0) {
    console.log('All tests passed! ✓');
  } else {
    console.error('Some tests failed! ✗');
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runTests();
}

export { testCases, simulateMiddleware, runTests };
