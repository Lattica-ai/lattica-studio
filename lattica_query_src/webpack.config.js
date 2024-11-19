const path = require('path');

module.exports = {
  entry: './src/index.ts', // Entry point for your TypeScript files
  devtool: 'source-map',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    extensions: ['.ts', '.js', '.wasm'], // Resolve TypeScript and JavaScript files
    fallback: {
      "crypto": false,
      "fs": false,
      "path": false,
    }
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.wasm$/,
        type: 'webassembly/async', // load WASM asynchronously
      },
    ],
  },
};
