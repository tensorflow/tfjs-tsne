/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

import {RearrangedData} from './interfaces';
import {KNNEstimator} from './knn';
import {tensorToDataTexture} from './tensor_to_data_texture';
import {TSNEOptimizer} from './tsne_optimizer';

export interface TSNEConfiguration {
  perplexity?: number;            // Default: 18
  exaggeration?: number;          // Default: 4
  exaggerationIter?: number;      // Default: 300
  exaggerationDecayIter?: number; // Default: 200
  momentum?: number;              // Default: 0.8
  applyGain?: boolean;            // Default: false
  verbose?: boolean;              // Default: false
  knnMode: 'auto'|'bruteForce';
  // Default: auto
}

/**
 * Returns the maximum value of perplexity given the available
 * WebGL capabilities.
 */
export function maximumPerplexity() {
  const backend = tf.ENV.findBackend('webgl') as tf.webgl.MathBackendWebGL;
  if (backend === null) {
    throw Error('WebGL backend is not available');
  }
  const gl = backend.getGPGPUContext().gl;
  const maxVaryingVectors = gl.getParameter(gl.MAX_VARYING_VECTORS);
  // one vector is reserved and each vector contains 4 neighbors;
  const numNeighbors = (maxVaryingVectors - 1) * 4;
  // the maximum perplexity is a third of the maximum number of neighbors
  const maximumPerplexity = Math.floor(numNeighbors / 3);
  return maximumPerplexity;
}

/**
 * Creates and returns a new TSNE embedder.
 *
 * @param data Rank 2 tensor of data to embed
 * @param config configuration options
 */
export function tsne(data: tf.Tensor, config?: TSNEConfiguration) {
  return new TSNE(data, config);
}

export class TSNE {
  private data: tf.Tensor;
  private numPoints: number;
  private numDimensions: number;
  private numNeighbors: number;
  private packedData: {texture: WebGLTexture, shape: RearrangedData};
  private verbose: boolean;
  private knnEstimator: KNNEstimator;
  private optimizer: TSNEOptimizer;
  private config: TSNEConfiguration;
  private initialized: boolean;
  private probabilitiesInitialized: boolean;
  private knnMode: 'auto'|'bruteForce';
  private appliedPerplexity: number;

  constructor(data: tf.Tensor, config?: TSNEConfiguration) {
    this.initialized = false;
    this.probabilitiesInitialized = false;
    this.data = data;
    this.config = config;

    const inputShape = this.data.shape;
    this.numPoints = inputShape[0];
    this.numDimensions = inputShape[1];

    if (inputShape.length !== 2) {
      throw Error('computeTSNE: input tensor must be 2-dimensional');
    }

    // Checking for a valid perplexity value given hardware limitations
    let perplexity = 18;

    if (this.config !== undefined) {
      if (this.config.perplexity !== undefined) {
        perplexity = this.config.perplexity;
      }
    }
    const maxPerplexity = maximumPerplexity();
    if (perplexity > maxPerplexity) {
      throw Error(`computeTSNE: perplexity cannot be greater than` +
                  `${maxPerplexity} on this machine`);
    }
    this.appliedPerplexity = perplexity;
  }

  /**
   * Initialization of the tSNE class. It is an async function as it performs
   * lenghty operations. The function is called the first time the iterateKnn
   * function is called.
   */
  private async initialize(): Promise<void> {
    // Default parameters
    let perplexity = 18;
    let exaggeration = 4;
    let exaggerationIter = 250;
    let exaggerationDecayIter = 150;
    let momentum = 0.5;
    let applyGain = true;
    this.verbose = false;
    this.knnMode = 'auto';

    // Reading user defined configuration
    if (this.config !== undefined) {
      if (this.config.perplexity !== undefined) {
        perplexity = this.config.perplexity;
      }
      if (this.config.exaggeration !== undefined) {
        exaggeration = this.config.exaggeration;
      }
      if (this.config.exaggerationIter !== undefined) {
        exaggerationIter = this.config.exaggerationIter;
      }
      if (this.config.exaggerationDecayIter !== undefined) {
        exaggerationDecayIter = this.config.exaggerationDecayIter;
      }
      if (this.config.momentum !== undefined) {
        momentum = this.config.momentum;
      }
      if (this.config.applyGain !== undefined) {
        applyGain = this.config.applyGain;
      }
      if (this.config.verbose !== undefined) {
        this.verbose = this.config.verbose;
      }
      if (this.config.knnMode !== undefined) {
        this.knnMode = this.config.knnMode;
      }
    }
    this.appliedPerplexity = perplexity;

    // Neighbors must be roughly 3*perplexity and a multiple of 4
    this.numNeighbors = Math.floor((perplexity * 3) / 4) * 4;
    this.packedData = await tensorToDataTexture(this.data);

    if (this.verbose) {
      console.log(`Number of points:\t${this.numPoints}`);
      console.log(`Number of dimensions:\t ${this.numDimensions}`);
      console.log(`Number of neighbors:\t${this.numNeighbors}`);
      console.log(`kNN mode:\t${this.knnMode}`);
    }

    this.knnEstimator = new KNNEstimator(
        this.packedData.texture, this.packedData.shape, this.numPoints,
        this.numDimensions, this.numNeighbors, this.verbose);

    this.optimizer = new TSNEOptimizer(this.numPoints, this.verbose);
    const exaggerationPolyline = [
      {iteration: exaggerationIter, value: exaggeration},
      {iteration: exaggerationIter + exaggerationDecayIter, value: 1}
    ];

    if (this.verbose) {
      console.log(
          `Exaggerating for ${exaggerationPolyline[0].iteration} ` +
          `iterations with a value of ${exaggerationPolyline[0].value}. ` +
          `Exaggeration is removed after ${
              exaggerationPolyline[1].iteration}.`);
    }

    this.optimizer.exaggeration = exaggerationPolyline;
    this.optimizer.momentum = momentum;
    this.optimizer.applyGain = applyGain;

    // We set a large step size (ETA) for large embeddings and we decrease it
    // for small embeddings.
    const maximumEta = 500;
    const minimumEta = 200;
    //const maximumEta = 200;
    //const minimumEta = 200;
    const numPointsMaximumEta = 2000;
    if (this.numPoints > numPointsMaximumEta) {
      this.optimizer.eta = maximumEta;
    } else {
      this.optimizer.eta = minimumEta +
          (maximumEta - minimumEta) * (this.numPoints / numPointsMaximumEta);
    }

    const spacePerPixel = 0.2;
    this.optimizer.spacePerPixel = spacePerPixel;

    this.initialized = true;

    if (this.verbose) {
      console.log('initialized');
    }

  }

  /**
   * Compute the tSNE embedding with a single call. This function will perform
   * the proper number of kNN iterations and the number of optimization
   * iterations provided as input
   * @param {number} iterations Number of iterations to compute. Default = 1000
   */
  async compute(iterations = 1000): Promise<void> {
    const knnIter = this.knnIterations();
    if (this.verbose) {
      console.log(`Number of KNN iterations:\t${knnIter}`);
      console.log('Computing the KNN...');
    }
    await this.iterateKnn(knnIter);
    if (this.verbose) {
      console.log('Computing the tSNE embedding...');
    }
    await this.iterate(iterations);
    if (this.verbose) {
      console.log('Done!');
    }
  }

  /**
   * Run k-nearest neighborhood computation for a given number of iterations
   * @param {number} iterations Number of iterations to compute. Default = 1
   */
  async iterateKnn(iterations = 1): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
    this.probabilitiesInitialized = false;
    for (let iter = 0; iter < iterations; ++iter) {
      this.knnEstimator.iterateKNNDescent();
      const syncCounter = 5;
      if ((this.knnEstimator.iteration % 100) === 0 && this.verbose) {
        console.log(`Iteration KNN:\t${this.knnEstimator.iteration}`);
      }
      if (tf.ENV.get('IS_CHROME') &&
        this.knnEstimator.iteration % syncCounter === 0) {
        // To ensure stability (in Chrome)
        await this.knnEstimator.forceSync();
      }
    }
  }

  /**
   * Run tSNE computation for a given number of iterations
   *
   * @param {number} iterations Number of iterations to compute. Default = 1
   */
  async iterate(iterations = 1): Promise<void> {
    if (!this.probabilitiesInitialized) {
      await this.initializeProbabilities();
    }
    for (let iter = 0; iter < iterations; ++iter) {
      await this.optimizer.iterate();
      if ((this.optimizer.iteration % 100) === 0 && this.verbose) {
        console.log(`Iteration tSNE:\t${this.optimizer.iteration}`);
      }
    }
  }

  /**
   * Return the maximum number of KNN iterations to be performed
   */
  knnIterations() {
    return Math.ceil(this.numPoints / 20);
  }

  /**
   * Return the coordinates of the tSNE embedding in an array.
   *
   * @param normalized boolean indicating whether to normalize
   *                   the coordinates to 0-1 range
   */
  coordinates(normalized = true): tf.Tensor {
    if (normalized) {
      return tf.tidy(() => {
        const rangeX = this.optimizer.maxX - this.optimizer.minX;
        const rangeY = this.optimizer.maxY - this.optimizer.minY;
        const min =
            tf.tensor2d([this.optimizer.minX, this.optimizer.minY], [1, 2]);
        const max =
            tf.tensor2d([this.optimizer.maxX, this.optimizer.maxY], [1, 2]);

        // The embedding is normalized in the 0-1 range while preserving the
        // aspect ratio
        const range = max.sub(min);
        const maxRange = tf.max(tf.tensor(range.dataSync()));
        const offset = tf.tidy(() => {
          if (rangeX < rangeY) {
            return tf.tensor2d([(rangeY - rangeX) / 2, 0], [1, 2]);
          } else {
            return tf.tensor2d([0, (rangeX - rangeY) / 2], [1, 2]);
          }
        });
        return this.optimizer.embedding2D.sub(min).add(offset).div(maxRange);
      });
    } else {
      return this.optimizer.embedding2D;
    }
  }

  /**
   * Return the coordinates of the tSNE embedding in an array.
   *
   * The array will be of the format [[x1, y1] ...]
   *
   * @param normalized boolean indicating whether to normalize
   *                   the coordinates to 0-1 range
   */
  async coordsArray(normalized = true): Promise<number[][]> {
    const coordsData = await this.coordinates(normalized).data();

    const coords = [];
    for (let i = 0; i < coordsData.length; i += 2) {
      coords.push([coordsData[i], coordsData[i + 1]]);
    }

    return coords;
  }

  /**
   * Return the cumulative distance in the KNN graph.
   * It can be used to show how fast the KNN graph converges to the solution
   */
  async knnTotalDistance(): Promise<number> {
    const sum = tf.tidy(() => {
      const distanceTensor = this.knnEstimator.distancesTensor();
      return distanceTensor.sum();
    });
    return (await sum.data())[0];
  }

  async setKnnData(numPoints: number, numNeighbors: number,
                   distances: Float32Array,
                   indices: Uint32Array): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
    this.probabilitiesInitialized = false;
    await this.optimizer.initializeNeighborsFromKNNGraph(
      numPoints, numNeighbors, distances, indices, this.appliedPerplexity);
    this.probabilitiesInitialized = true;
  }

  /**
   * Initialize the joint probability distribution from the computed KNN graph.
   * It is called in the iterate function if there are updates in the KNN graph
   * due to a previous call of the iterateKnn function
   */
  private async initializeProbabilities() {
    if (this.verbose) {
      console.log(`Initializing probabilities`);
    }
    await this.optimizer.initializeNeighborsFromKNNTexture(
        this.knnEstimator.knnShape,
        this.knnEstimator.knn(),
        this.appliedPerplexity);

    if (this.verbose) {
      console.log(`Initialized probabilities from kNN Texture`);
    }
    this.probabilitiesInitialized = true;
  }
}
