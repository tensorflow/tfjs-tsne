## tSNE for TensorFlow.js

This library contains a improved tSNE implementation that runs in the browser.


### Computing tSNE with a single call

A simple example for computing the tSNE embedding for a random dataset
containing 2000 in a 10 dimensional space is the following:

```javascript
const data = tf.randomUniform([2000,10]);
const tsne = tf_tsne.tsne(data);
tsne.compute();
const coordinates = tsne.coordinates();
coordinates.print();
```

### Computing tSNE iteratively

You can also compute the embedding iteratively.
First you have to compute the KNN graph, an iterative operation that does not provide any intermediate result.
Then you can compute the tSNE iteratively and drawing the result as it evolves.

```javascript
const data = tf.randomUniform([2000,10]);
const tsne = tf_tsne.tsne(data);

//Get the maximum number of iterations to perform
const knnIterations = tsne.knnIterations();
for(let i = 0; i < knnIterations; ++i){
  tsne.iterateKnn();
  //Notify the percentage of computed KNN
}

const tsneIterations = 1000;
for(let i = 0; i < tsneIterations; ++i){
  tsne.iterate();
  //Draw the embedding here
}
tsne.compute();
const coordinates = tsne.coordinates();
coordinates.print();
```

### Limitations
We experimented with different data size, both in the number of data points and dimensions.
As a rule of thumb, you can safely embed data with a shape of [10000,100].
***You can go up to?!? not sure on how to phrase it...***

Above a certain number of data points the computation of the similarities becomes a bottleneck, a problem that we plan to address in the future.


### Implementation
This work makes use of [linear tSNE optimization](https://arxiv) for the optimization of the embedding and an optimized brute force computation of the kNN graph in the GPU.

### Reference
Reference to cite if you use this implementation in a research paper:
***Are you fine with this or is too pushy?!?***
