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
  perplexity?: number;            // Default: 30
  exaggeration?: number;          // Default: 4
  exaggerationIter?: number;      // Default: 300
  exaggerationDecayIter?: number; // Default: 200
  momentum?: number;              // Default: 0.8
  verbose?: boolean;              // Default: false
  knnMode: 'auto'|'bruteForce'|'kNNDescentProgram'|'random';
  // Default: auto
}

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
  private knnMode: 'auto'|'bruteForce'|'kNNDescentProgram'|'random';

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

    // TODO remove once this is used elsewhere.
    console.log(this.knnMode);
  }

  private async initialize(): Promise<void> {
    // Default parameters
    let perplexity = 30;
    let exaggeration = 4;
    let exaggerationIter = 300;
    let exaggerationDecayIter = 200;
    let momentum = 0.8;
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
      if (this.config.verbose !== undefined) {
        this.verbose = this.config.verbose;
      }
      if (this.config.knnMode !== undefined) {
        this.knnMode = this.config.knnMode;
      }
    }

    // Number of neighbors cannot exceed 128
    if (perplexity > 42) {
      throw Error('computeTSNE: perplexity cannot be greater than 42');
    }
    // Neighbors must be roughly 3*perplexity and a multiple of 4
    this.numNeighbors = Math.floor((perplexity * 3) / 4) * 4;
    this.packedData = await tensorToDataTexture(this.data);

    if (this.verbose) {
      console.log(`Number of points ${this.numPoints}`);
      console.log(`Number of dimensions ${this.numDimensions}`);
      console.log(`Number of neighbors ${this.numNeighbors}`);
    }

    this.knnEstimator = new KNNEstimator(
        this.packedData.texture, this.packedData.shape, this.numPoints,
        this.numDimensions, this.numNeighbors, false);

    this.optimizer = new TSNEOptimizer(this.numPoints, false);
    const exaggerationPolyline = [
      {iteration : exaggerationIter, value : exaggeration},
      {iteration : exaggerationIter + exaggerationDecayIter, value : 1}
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
  }

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

  async iterateKnn(iterations = 1): Promise<boolean> {
    if (!this.initialized) {
      await this.initialize();
    }
    this.probabilitiesInitialized = false;
    for (let iter = 0; iter < iterations; ++iter) {
      this.knnEstimator.iterateBruteForce();
      if ((this.knnEstimator.iteration % 100) === 0 && this.verbose) {
        console.log(`Iteration KNN:\t${this.knnEstimator.iteration}`);
      }
    }
    return true; // TODO
  }
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
  knnIterations() { return Math.ceil(this.numPoints / 20); }

  coordinates(normalized = true): tf.Tensor {
    if (normalized) {
      return tf.tidy(() => {
        const rangeX = this.optimizer.maxX - this.optimizer.minX;
        const rangeY = this.optimizer.maxY - this.optimizer.minY;
        const min =
            tf.tensor2d([ this.optimizer.minX, this.optimizer.minY ], [ 1, 2 ]);
        const max =
            tf.tensor2d([ this.optimizer.maxX, this.optimizer.maxY ], [ 1, 2 ]);

        // The embedding is normalized in the 0-1 range while preserving the
        // aspect ratio
        const range = max.sub(min);
        const maxRange = tf.max(range);
        const offset = tf.tidy(() => {
          if (rangeX < rangeY) {
            return tf.tensor2d([ (rangeY - rangeX) / 2, 0 ], [ 1, 2 ]);
          } else {
            return tf.tensor2d([ 0, (rangeX - rangeY) / 2 ], [ 1, 2 ]);
          }
        });

        return this.optimizer.embedding2D.sub(min).add(offset).div(maxRange);
      });

    } else {
      return this.optimizer.embedding2D;
    }
  }

  knnDistance(): number {
    // TODO
    return 0;
  }

  private async initializeProbabilities() {
    if (this.verbose) {
      console.log(`Initializing probabilities`);
    }
    await this.optimizer.initializeNeighborsFromKNNTexture(
        this.knnEstimator.knnShape, this.knnEstimator.knn());

    this.probabilitiesInitialized = true;
  }
}
