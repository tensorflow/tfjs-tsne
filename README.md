## tSNE for TensorFlow.js

This library contains a improved tSNE implementation that runs in the browser.


## Installation & Usage

You can use tfjs-tsne via a script tag or via NPM

### Script tag

To use tfjs-tsne via script tag you need to load tfjs first. The following tags
can be put into the head section of your html page to load the library.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tsne"></script>
```

This library will create a `tsne` variable on the global scope.
You can then do the following

```js
// Create some data
const data = tf.randomUniform([2000,10]);

// Get a tsne optimizer
const tsneOpt = tsne.optimizer(data);

// Compute a T-SNE embedding, returns a promise.
// Runs for 1000 iterations be default.
tsneOpt.compute().then(() => {
  // tsne.coordinate returns a *tensor* with x, y coordinates of
  // the embedded data.
  const coordinates = tsneOpt.coordinates();
  coordinates.print();
}) ;
```

### Via NPM

```
yarn add tensorflow@tfjs-tsne
```
or
```
npm install tensorflow@tfjs-tsne
```

Then

```js
import tsne from '@tensorflow/tfjs-tsne';

// Create some data
const data = tf.randomUniform([2000,10]);

// Initialize the tsne optimizer
const tsneOpt = tsne.optimizer(data);

// Compute a T-SNE embedding, returns a promise.
// Runs for 1000 iterations be default.
tsneOpt.compute().then(() => {
  // tsne.coordinate returns a *tensor* with x, y coordinates of
  // the embedded data.
  const coordinates = tsneOpt.coordinates();
  coordinates.print();
}) ;
```

## API

### tsne.optimizer(data: tf.Tensor2d, config?: TSNEConfiguration)

Creates and returns a TSNE optimizer.

- `data` must be a Rank 2 tensor. Shape is [numPoints, dataPointDimensions]
- `config` is an optinal object with the following params (all are optional):
  - perplexity: number — defaults to 30. Max value is 42
  - verbose: boolean — defaults to false
  - exaggeration: number — defaults to 4
  - exaggerationIter: number — defaults to 300
  - exaggerationDecayIter: number — defaults to 200
  - momentum: number — defaults to 0.8

### .compute(iterations: number): Promise<void>

The most direct way to get a tsne projection. Automtatically runs the knn preprocessing
and the tsne optimization. Returns a promise to indicate when it is done.

- iterations the number of iterations to run the tsne optimization for. (The number of knn steps is automatically calculated).

### .iterateKnn(iterations: number): Promise<void>

When running tsne iteratively (see section below). This runs runs the knn preprocessing
for the specified number of iterations.

### .iterate(iterations: number): Promise<void>

When running tsne iteratively (see section below). This runs runs the tsne step for the specified number of iterations.

### .coordinates(normalize: boolean): tf.Tensor

Gets the current x, y coordinates of the projected data as a tensor. By 
default the coordinates are normalized to the range 0-1.

### .coordsArray(normalize: boolean): Promise<number[][]>

Gets the current x, y coordinates of the projected data as a JavaScript array.
By default the coordinates are normalized to the range 0-1. This function is
async and returns a promise.

### Computing tSNE iteratively

While the `.compute` method provides the most direct way to get an embedding. You can also compute the embedding iteratively and have more control over the process.

The first step is computing the KNN graph using iterateKNN.

Then you can compute the tSNE iteratively and examine the result as it evolves.

The code below shows what that would look like

```javascript
const data = tf.randomUniform([2000,10]);
const tsne = tf_tsne.tsne(data);

async function iterativeTsne() {
  // Get the suggested number of iterations to perform.
  const knnIterations = tsne.knnIterations();
  // Do the KNN computation. This needs to complete before we run tsne
  for(let i = 0; i < knnIterations; ++i){
    await tsne.iterateKnn();
    // You can update knn progress in your ui here.
  }

  const tsneIterations = 1000;
  for(let i = 0; i < tsneIterations; ++i){
    await tsne.iterate();
    // Draw the embedding here...
    const coordinates = tsne.coordinates();
    coordinates.print();
  }
}

iterativeTsne();
```

### Limitations

From our current experiments we suggest limiting the data size passed to this implementation
to data with a shape of [10000,100], i.e. up to 10000 points with 100 dimensions each. You can do more but it might slow down.

Above a certain number of data points the computation of the similarities becomes a bottleneck, a problem that we plan to address in the future.


### Implementation
This work makes use of [linear tSNE optimization](https://arxiv.org/abs/1805.10817) for the optimization of the embedding and an optimized brute force computation of the kNN graph in the GPU.

### Reference
Reference to cite if you use this implementation in a research paper:

```
@article{TFjs:tSNE,
  author = {Nicola Pezzotti and Alexander Mordvintsev and Thomas Hollt and Boudewijn P. F. Lelieveldt and Elmar Eisemann and Anna Vilanova},
  title = {Linear tSNE Optimization for the Web},
  year = {2018},
  journal={arXiv preprint arXiv:1805.10817},
}
```
