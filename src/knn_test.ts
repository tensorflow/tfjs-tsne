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

import * as tfc from '@tensorflow/tfjs-core';

import * as gl_util from './gl_util';
import * as tf_knn from './knn';
import {RearrangedData} from './interfaces';

function iterate(knn: tf_knn.KNNEstimator, knnTechnique: string) {
  if (knnTechnique === 'brute force') {
    knn.iterateBruteForce();
  } else if (knnTechnique === 'random sampling') {
    knn.iterateRandomSampling();
  } else if (knnTechnique === 'knn descent') {
    knn.iterateKNNDescent();
  } else {
    throw new Error('unknown knn technique');
  }
}

function knnIntegrityTests(
    knnTechnique: string, dataTexture: WebGLTexture,
    dataFormat: RearrangedData, numPoints: number, numDimensions: number,
    numNeighs: number) {
  it(`kNN increments the iterations
      (${knnTechnique}, #neighs: ${numNeighs})`,
     () => {
       const knn = new tf_knn.KNNEstimator(
           dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
       iterate(knn, knnTechnique);
       expect(knn.iteration).toBe(1);
       iterate(knn, knnTechnique);
       expect(knn.iteration).toBe(2);
     });

  it(`kNN preserves the heap property
      (${knnTechnique}, #neighs: ${numNeighs})`,
     () => {
       const knn = new tf_knn.KNNEstimator(
           dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
       // doing some iterations
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);

       tfc.tidy(() => {
         const distancesTensor = knn.distancesTensor();
         expect(checkHeap(distancesTensor, numPoints, numNeighs)).toBe(true);
       });
     });

  it(`kNN does not have duplicates
      (${knnTechnique}, #neighs: ${numNeighs})`,
     () => {
       const knn = new tf_knn.KNNEstimator(
           dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
       // doing some iterations
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);

       tfc.tidy(() => {
         const indices = knn.indicesTensor();
         expect(checkDuplicates(indices, numPoints, numNeighs)).toBe(true);
       });
     });

  it(`kNN does not have invalid neighbors
      (${knnTechnique}, #neighs: ${numNeighs})`,
     () => {
       const knn = new tf_knn.KNNEstimator(
           dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
       // doing some iterations
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);
       iterate(knn, knnTechnique);

       tfc.tidy(() => {
         const indices = knn.indicesTensor();
         expect(checkInvalidNeighbors(indices, numPoints, numNeighs))
             .toBe(true);
       });
     });
}

//////////////////////////////////////////////////////

describe('KNN [line]\n', () => {
  const numDimensions = 12;
  const pointsPerRow = 10;
  const numRows = 100;
  const numPoints = pointsPerRow * numRows;

  const vec = new Uint8Array(pointsPerRow * numDimensions * numRows);
  for (let i = 0; i < numPoints; ++i) {
    for (let d = 0; d < numDimensions; ++d) {
      vec[i * numDimensions + d] = 255. * i / numPoints;
    }
  }
  const backend = tfc.ENV.findBackend('webgl') as tfc.webgl.MathBackendWebGL;
  const gpgpu = backend.getGPGPUContext();
  const dataTexture = gl_util.createAndConfigureUByteTexture(
      gpgpu.gl, pointsPerRow * numDimensions / 4, numRows, 4, vec);

  const dataFormat = {
    numPoints,
    pointsPerRow,
    pixelsPerPoint: numDimensions / 4,
    numRows,
  };
  /////////////////////////////////////////////////////
  it('Checks for too large a neighborhood', () => {
    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      new tf_knn.KNNEstimator(
          dataTexture, dataFormat, numPoints, numDimensions, 129, false);
    }).toThrow();
  });

  it('k must be a multiple of 4', () => {
    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      new tf_knn.KNNEstimator(
          dataTexture, dataFormat, numPoints, numDimensions, 50, false);
    }).toThrow();
  });

  it('kNN initializes iterations to 0', () => {
    const knn = new tf_knn.KNNEstimator(
        dataTexture, dataFormat, numPoints, numDimensions, 100, false);
    expect(knn.iteration).toBe(0);
  });

  knnIntegrityTests(
      'brute force', dataTexture, dataFormat, numPoints, numDimensions, 100);
  knnIntegrityTests(
      'random sampling', dataTexture, dataFormat, numPoints, numDimensions,
      100);
  knnIntegrityTests(
      'knn descent', dataTexture, dataFormat, numPoints, numDimensions, 100);

  knnIntegrityTests(
      'brute force', dataTexture, dataFormat, numPoints, numDimensions, 48);
  knnIntegrityTests(
      'random sampling', dataTexture, dataFormat, numPoints, numDimensions, 48);
  knnIntegrityTests(
      'knn descent', dataTexture, dataFormat, numPoints, numDimensions, 48);
});

////////////////////////////////////////////////////////////

//TODO switch to await data() in the test
function linearTensorAccess(tensor: tfc.Tensor, i: number): number {
  const elemPerRow = tensor.shape[1];
  const col = i % elemPerRow;
  const row = Math.floor(i / elemPerRow);
  return tensor.get(row, col);
}

// Check if the returned kNN preserves the heap property
function checkHeap(
    distances: tfc.Tensor, numPoints: number, numNeighs: number) {
  const eps = 10e-5;
  for (let i = 0; i < numPoints; ++i) {
    const s = i * numNeighs;
    for (let n = 0; n < numNeighs; ++n) {
      const fatherId = s + n;
      const fatherValue = linearTensorAccess(distances, s + n);
      const sonLeftId = s + n * 2 + 1;
      const sonRightId = s + n * 2 + 2;

      if (sonLeftId < numNeighs &&
          fatherValue - linearTensorAccess(distances, sonLeftId) < -eps) {
        distances.print();
        console.log(`fatherAbs ${fatherId - s}`);
        console.log(`fatherId ${fatherId}`);
        console.log(fatherValue);
        console.log(`sonAbs ${sonLeftId - s}`);
        console.log(`sonId ${sonLeftId}`);
        console.log(linearTensorAccess(distances, sonLeftId));
        return false;
      }
      if (sonRightId < numNeighs &&
          fatherValue - linearTensorAccess(distances, sonRightId) < -eps) {
        distances.print();
        console.log(`fatherAbs ${fatherId - s}`);
        console.log(`fatherId ${fatherId}`);
        console.log(fatherValue);
        console.log(`sonAbs ${sonRightId - s}`);
        console.log(`sonId ${sonRightId}`);
        console.log(linearTensorAccess(distances, sonRightId));
        return false;
      }
    }
  }
  return true;
}

// check if there are no dubplicates in the kNN
function checkDuplicates(
    indices: tfc.Tensor, numPoints: number, numNeighs: number) {
  let duplicates = 0;
  for (let i = 0; i < numPoints; ++i) {
    const s = i * numNeighs;
    for (let n = 0; n < numNeighs; ++n) {
      for (let n1 = n + 1; n1 < numNeighs; ++n1) {
        const value = linearTensorAccess(indices, s + n);
        const value1 = linearTensorAccess(indices, s + n1);
        if (value === value1 && value1 !== -1) {
          duplicates++;
        }
      }
    }
  }
  if (duplicates !== 0) {
    console.log(`Duplicates:\t ${duplicates}`);
    console.log(`Duplicates per point:\t ${duplicates / numPoints}`);
    return false;
  }
  return true;
}

// check if there are no dubplicates in the kNN
function checkInvalidNeighbors(
    indices: tfc.Tensor, numPoints: number, numNeighs: number) {
  // Revise with the new access
  let invalid = 0;
  for (let i = 0; i < numPoints; ++i) {
    const s = i * numNeighs;
    for (let n = 0; n < numNeighs; ++n) {
      const value = linearTensorAccess(indices, s + n);
      if (value === -1) {
        invalid++;
      }
    }
  }
  if (invalid !== 0) {
    console.log(`Invalid:\t ${invalid}`);
    console.log(`Invalid per point:\t ${invalid / numPoints}`);
    return false;
  }
  return true;
}
