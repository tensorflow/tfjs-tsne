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

import * as dataset_util from './dataset_util';
import * as tf_tsne from './tsne_optimizer';

describe('TSNEOptimizer class', () => {
  it('is initialized correctly', () => {
    const tsne = new tf_tsne.TSNEOptimizer(100, false);
    expect(tsne.minX).toBeLessThan(tsne.maxX);
    expect(tsne.minY).toBeLessThan(tsne.maxY);
    expect(tsne.numberOfPoints).toBe(100);
    expect(tsne.numberOfPointsPerRow).toBe(8);
    expect(tsne.numberOfRows).toBe(13);
    tsne.dispose();
  });
  //
  // it('requires the neighborhoods to perform iterations', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.iterate();
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('requires the neighborhoods to perform iterations', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const embedding2D = tsne.embedding2D;
  //   expect(embedding2D.shape[0]).toBe(100);
  //   expect(embedding2D.shape[1]).toBe(2);
  //   embedding2D.dispose();
  //   tsne.dispose();
  // });
  //
  // it('detects a mismatched data shape', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const knnGraph = dataset_util.generateKNNClusterData(1000, 10, 100);
  //   expect(() => {
  //     tsne.initializeNeighborsFromKNNGraph(
  //         1000, 100, knnGraph.distances, knnGraph.indices);
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('requires 4 tensors', () => {
  //   // Embedding + Gradiend + momentum + exaggeration
  //   const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //   expect(tf.memory().numTensors).toBe(4);
  //   tsne.dispose();
  // });
  //
  // it('disposes its tensors', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //   tsne.dispose();
  //   expect(tf.memory().numTensors).toBe(0);
  // });
  //
  // it('keeps the number of tensors constant during neighbors initialization',
  //    () => {
  //      const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //      const numTensors = tf.memory().numTensors;
  //
  //      const knnGraph = dataset_util.generateKNNClusterData(1000, 10, 100);
  //      expect(tf.memory().numTensors).toBe(numTensors);
  //      tsne.initializeNeighborsFromKNNGraph(
  //          1000, 100, knnGraph.distances, knnGraph.indices);
  //      expect(tf.memory().numTensors).toBe(numTensors);
  //      tsne.dispose();
  //    });
  //
  // it('keeps the number of tensors constant during SGD', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //   const knnGraph = dataset_util.generateKNNClusterData(1000, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       1000, 30, knnGraph.distances, knnGraph.indices);
  //
  //   const numTensors = tf.memory().numTensors;
  //   const numIter = 100;
  //   for (let i = 0; i < numIter; ++i) {
  //     tsne.iterate();
  //   }
  //   expect(tf.memory().numTensors).toBe(numTensors);
  //   tsne.dispose();
  // });
  //
  // it('initializes the iterations to 0', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //   expect(tsne.iteration).toBe(0);
  //   tsne.dispose();
  // });
  //
  // it('counts the iterations', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //   const knnGraph = dataset_util.generateKNNClusterData(1000, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       1000, 30, knnGraph.distances, knnGraph.indices);
  //
  //   const numIter = 10;
  //   for (let i = 0; i < numIter; ++i) {
  //     tsne.iterate();
  //   }
  //   expect(tsne.iteration).toBe(numIter);
  //   tsne.dispose();
  // });
  //
  // it('resets the iterations counter on re-init', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(1000, false);
  //   const knnGraph = dataset_util.generateKNNClusterData(1000, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       1000, 30, knnGraph.distances, knnGraph.indices);
  //
  //   const numIter = 10;
  //   for (let i = 0; i < numIter; ++i) {
  //     tsne.iterate();
  //   }
  //   tsne.initializeEmbedding();
  //
  //   expect(tsne.iteration).toBe(0);
  //   tsne.dispose();
  // });
  //
  // it('has proper ETA getter/setter', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(tsne.eta).toBeCloseTo(2500);
  //   tsne.eta = 1000;
  //   expect(tsne.eta).toBeCloseTo(1000);
  //   tsne.dispose();
  // });
  //
  // it('has proper Momentum getter/setter', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(tsne.momentum).toBeCloseTo(0.8);
  //   tsne.momentum = 0.1;
  //   expect(tsne.momentum).toBeCloseTo(0.1);
  //   tsne.dispose();
  // });
  //
  // it('has proper Exaggeration getter/setter', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(4);
  //   tsne.exaggeration = 3;
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.dispose();
  // });
  //
  // it('does not increase the tensor count when momentum is changed', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const numTensors = tf.memory().numTensors;
  //   tsne.momentum = 0.1;
  //   expect(tf.memory().numTensors).toBe(numTensors);
  //   tsne.dispose();
  // });
  //
  // it('does not increase the tensor count when exaggeration is changed', () =>
  // {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const numTensors = tf.memory().numTensors;
  //   tsne.exaggeration = 4;
  //   expect(tf.memory().numTensors).toBe(numTensors);
  //   tsne.dispose();
  // });
  //
  // it('does not increase the tensor count after an embedding re-init', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const numTensors = tf.memory().numTensors;
  //   tsne.initializeEmbedding();
  //   expect(tf.memory().numTensors).toBe(numTensors);
  //   tsne.dispose();
  // });
  //
  // it('throws if a negative momentum is set', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.momentum = -0.1;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('does not throw if momentum is set to zero', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.momentum = 0.;
  //   }).not.toThrow();
  //   tsne.dispose();
  // });
  //
  // it('throws if a momentum higher than one is set', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.momentum = 2;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('throws if exaggeration is set to a value lower than one', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.exaggeration = 0.9;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('does not throw if exaggeration is set to one', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.exaggeration = 1;
  //   }).not.toThrow();
  //   tsne.dispose();
  // });
  //
  // it('accpets only piecewise linear exaggeration greater' +
  //        ' than or equal to 1 (0)',
  //    () => {
  //      const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //      const ex = [{iteration: 0, value: 2}, {iteration: 100, value: 0.5}];
  //      expect(() => {
  //        tsne.exaggeration = ex;
  //      }).toThrow();
  //      tsne.dispose();
  //    });
  //
  // it('accpets only piecewise linear exaggeration greater' +
  //        ' than or equal to 1 (1)',
  //    () => {
  //      const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //      const ex = [{iteration: 0, value: 0}];
  //      expect(() => {
  //        tsne.exaggeration = ex;
  //      }).toThrow();
  //      tsne.dispose();
  //    });
  //
  // it('accpets exaggeration functions with non negative iteratiosn (0)', () =>
  // {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 0, value: 2}, {iteration: -100, value: 0.5}];
  //   expect(() => {
  //     tsne.exaggeration = ex;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('accpets exaggeration functions with non negative iteratiosn (0)', () =>
  // {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: -100, value: 0.5}];
  //   expect(() => {
  //     tsne.exaggeration = ex;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('accpets exaggeration functions (domain is always increasing 1)', () =>
  // {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 200, value: 2}, {iteration: 100, value: 1}];
  //   expect(() => {
  //     tsne.exaggeration = ex;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('accpets exaggeration functions (domain is always increasing 2)', () =>
  // {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 200, value: 2}, {iteration: 200, value: 1}];
  //   expect(() => {
  //     tsne.exaggeration = ex;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('throws if a non-positive ETA is set', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.eta = -1000;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('throws if ETA is set to zero', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(() => {
  //     tsne.eta = 0;
  //   }).toThrow();
  //   tsne.dispose();
  // });
  //
  // it('has proper exaggeration for the current iteration (0)', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(4);
  //   tsne.exaggeration = 3;
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.dispose();
  // });
  //
  // it('has proper exaggeration for the current iteration (1)', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 0, value: 3}];
  //   tsne.exaggeration = ex;
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.dispose();
  // });
  //
  // it('has proper exaggeration for the current iteration (1)', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 0, value: 3}];
  //   tsne.exaggeration = ex;
  //
  //   const knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       100, 30, knnGraph.distances, knnGraph.indices);
  //   tsne.iterate();
  //   tsne.iterate();
  //
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.dispose();
  // });
  //
  // it('has proper exaggeration for the current iteration (2)', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 0, value: 3}, {iteration: 2, value: 1}];
  //   tsne.exaggeration = ex;
  //   const knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       100, 30, knnGraph.distances, knnGraph.indices);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(2);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
  //   tsne.dispose();
  // });
  //
  // it('has proper exaggeration for the current iteration (3)', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 1, value: 3}, {iteration: 3, value: 1}];
  //   tsne.exaggeration = ex;
  //
  //   const knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       100, 30, knnGraph.distances, knnGraph.indices);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(2);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
  //   tsne.dispose();
  // });
  //
  // it('has proper exaggeration for the current iteration (4)', () => {
  //   const tsne = new tf_tsne.TSNEOptimizer(100, false);
  //   const ex = [{iteration: 0, value: 5}, {iteration: 4, value: 1}];
  //   tsne.exaggeration = ex;
  //
  //   const knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
  //   tsne.initializeNeighborsFromKNNGraph(
  //       100, 30, knnGraph.distances, knnGraph.indices);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(5);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(4);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(2);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
  //   tsne.iterate();
  //   expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
  //   tsne.dispose();
  // });
});
