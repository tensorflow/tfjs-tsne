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

import * as dataset_util from './dataset_util';
import * as gl_util from './gl_util';
import {RearrangedData} from './interfaces';
import * as knn_util from './knn_util';

// tslint:disable-next-line:no-any
function instanceOfRearrangedData(object: any): object is RearrangedData {
  return 'numPoints' in object && 'pointsPerRow' in object &&
      'pixelsPerPoint' in object && 'numRows' in object;
}

// Allows for computing distances between data in a non standard format
export interface CustomDataDefinition {
  distanceComputationCode: string;
}
// tslint:disable-next-line:no-any
function instanceOfCustomDataDefinition(object: any):
    object is CustomDataDefinition {
  return 'distanceComputationCode' in object;
}

export class KNNEstimator {
  private verbose: boolean;
  private backend: tfc.webgl.MathBackendWebGL;
  private gpgpu: tfc.webgl.GPGPUContext;

  private _iteration: number;
  private numNeighs: number;

  private bruteForceKNNProgram: WebGLProgram;
  private randomSamplingKNNProgram: WebGLProgram;
  private kNNDescentProgram: WebGLProgram;
  private copyDistancesProgram: WebGLProgram;
  private copyIndicesProgram: WebGLProgram;

  private linesVertexIdBuffer: WebGLBuffer;

  private dataTexture: WebGLTexture;
  private knnTexture0: WebGLTexture;
  private knnTexture1: WebGLTexture;

  private knnDataShape: RearrangedData;

  get knnShape(): RearrangedData {
    return this.knnDataShape;
  }
  get iteration() {
    return this._iteration;
  }

  constructor(
      dataTexture: WebGLTexture,
      dataFormat: RearrangedData|CustomDataDefinition, numPoints: number,
      numDimensions: number, numNeighs: number, verbose?: boolean) {
    if (verbose != null) {
      this.verbose = verbose;
    } else {
      verbose = false;
    }
    // Saving the GPGPU context
    this.backend = tfc.ENV.findBackend('webgl') as tfc.webgl.MathBackendWebGL;
    this.gpgpu = this.backend.getGPGPUContext();

    this._iteration = 0;
    this.dataTexture = dataTexture;

    if (numNeighs > 128) {
      throw new Error('kNN size must not be greater than 128');
    }
    if (numNeighs % 4 !== 0) {
      throw new Error('kNN size must be a multiple of 4');
    }
    this.numNeighs = numNeighs;

    // Input Shape
    const knnPointsPerRow =
        Math.ceil(Math.sqrt(numNeighs * numPoints) / numNeighs);
    this.knnDataShape = {
      numPoints,
      pixelsPerPoint: numNeighs,
      pointsPerRow: knnPointsPerRow,
      numRows: Math.ceil(numPoints / knnPointsPerRow)
    };

    this.log('knn-pntsPerRow', this.knnDataShape.pointsPerRow);
    this.log('knn-numRows', this.knnDataShape.numRows);
    this.log('knn-pixelsPerPoint', this.knnDataShape.pixelsPerPoint);

    // Generating the source for computing the distances between points
    let distanceComputationSource: string;
    if (instanceOfRearrangedData(dataFormat)) {
      const rearrangedData = dataFormat as RearrangedData;
      distanceComputationSource =
          dataset_util.generateDistanceComputationSource(rearrangedData);
    } else if (instanceOfCustomDataDefinition(dataFormat)) {
      const customDataDefinition = dataFormat as CustomDataDefinition;
      distanceComputationSource = customDataDefinition.distanceComputationCode;
    }

    // Initialize the WebGL custom programs
    this.initializeTextures();
    this.initilizeCustomWebGLPrograms(distanceComputationSource);
  }

  get pointsPerIteration() {
    return 20;
  }

  // Utility function for printing stuff
  // tslint:disable-next-line:no-any
  private log(str: string, obj?: any) {
    if (this.verbose) {
      if (obj != null) {
        console.log(`${str}: \t${obj}`);
      } else {
        console.log(str);
      }
    }
  }

  private initializeTextures() {
    const initNeigh = new Float32Array(
        this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint * 2 *
        this.knnDataShape.numRows);

    const numNeighs = this.knnDataShape.pixelsPerPoint;
    for (let i = 0; i < this.knnDataShape.numPoints; ++i) {
      for (let n = 0; n < numNeighs; ++n) {
        initNeigh[(i * numNeighs + n) * 2] = -1;
        initNeigh[(i * numNeighs + n) * 2 + 1] = 10e30;
      }
    }
    this.knnTexture0 = gl_util.createAndConfigureTexture(
        this.gpgpu.gl,
        this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint,
        this.knnDataShape.numRows, 2, initNeigh);

    this.knnTexture1 = gl_util.createAndConfigureTexture(
        this.gpgpu.gl,
        this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint,
        this.knnDataShape.numRows, 2, initNeigh);
  }

  private initilizeCustomWebGLPrograms(distanceComputationSource: string) {
    this.copyDistancesProgram = knn_util.createCopyDistancesProgram(this.gpgpu);
    this.copyIndicesProgram = knn_util.createCopyIndicesProgram(this.gpgpu);

    this.bruteForceKNNProgram = knn_util.createBruteForceKNNProgram(
        this.gpgpu, this.numNeighs, distanceComputationSource);
    this.randomSamplingKNNProgram = knn_util.createRandomSamplingKNNProgram(
        this.gpgpu, this.numNeighs, distanceComputationSource);
    this.kNNDescentProgram = knn_util.createKNNDescentProgram(
        this.gpgpu, this.numNeighs, distanceComputationSource);

    const linesVertexId = new Float32Array(this.knnDataShape.numPoints * 2);
    {
      for (let i = 0; i < this.knnDataShape.numPoints * 2; ++i) {
        linesVertexId[i] = i;
      }
    }
    this.linesVertexIdBuffer = tfc.webgl.webgl_util.createStaticVertexBuffer(
        this.gpgpu.gl, linesVertexId);
  }

  iterateBruteForce() {
    if ((this._iteration % 2) === 0) {
      this.iterateGPU(
          this.dataTexture, this._iteration, this.knnTexture0,
          this.knnTexture1);
    } else {
      this.iterateGPU(
          this.dataTexture, this._iteration, this.knnTexture1,
          this.knnTexture0);
    }
    ++this._iteration;
    this.gpgpu.gl.finish();
  }
  iterateRandomSampling() {
    if ((this._iteration % 2) === 0) {
      this.iterateRandomSamplingGPU(
          this.dataTexture, this._iteration, this.knnTexture0,
          this.knnTexture1);
    } else {
      this.iterateRandomSamplingGPU(
          this.dataTexture, this._iteration, this.knnTexture1,
          this.knnTexture0);
    }
    ++this._iteration;
    this.gpgpu.gl.finish();
  }
  iterateKNNDescent() {
    if ((this._iteration % 2) === 0) {
      this.iterateKNNDescentGPU(
          this.dataTexture, this._iteration, this.knnTexture0,
          this.knnTexture1);
    } else {
      this.iterateKNNDescentGPU(
          this.dataTexture, this._iteration, this.knnTexture1,
          this.knnTexture0);
    }
    ++this._iteration;
    this.gpgpu.gl.finish();
  }

  knn(): WebGLTexture {
    if ((this._iteration % 2) === 0) {
      return this.knnTexture0;
    } else {
      return this.knnTexture1;
    }
  }

  distancesTensor(): tfc.Tensor {
    return tfc.tidy(() => {
      const distances = tfc.zeros([
        this.knnDataShape.numRows,
        this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint
      ]);
      const knnTexture = this.knn();
      knn_util.executeCopyDistancesProgram(
          this.gpgpu, this.copyDistancesProgram, knnTexture, this.knnDataShape,
          this.backend.getTexture(distances.dataId));
      return distances;
    });
  }

  indicesTensor(): tfc.Tensor {
    return tfc.tidy(() => {
      const indices = tfc.zeros([
        this.knnDataShape.numRows,
        this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint
      ]);
      const knnTexture = this.knn();
      knn_util.executeCopyIndicesProgram(
          this.gpgpu, this.copyIndicesProgram, knnTexture, this.knnDataShape,
          this.backend.getTexture(indices.dataId));
      return indices;
    });
  }

  private iterateGPU(
      dataTexture: WebGLTexture, _iteration: number,
      startingKNNTexture: WebGLTexture, targetTexture?: WebGLTexture) {
    knn_util.executeKNNProgram(
        this.gpgpu, this.bruteForceKNNProgram, dataTexture, startingKNNTexture,
        _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
  }
  private iterateRandomSamplingGPU(
      dataTexture: WebGLTexture, _iteration: number,
      startingKNNTexture: WebGLTexture, targetTexture?: WebGLTexture) {
    knn_util.executeKNNProgram(
        this.gpgpu, this.randomSamplingKNNProgram, dataTexture,
        startingKNNTexture, _iteration, this.knnDataShape,
        this.linesVertexIdBuffer, targetTexture);
  }
  private iterateKNNDescentGPU(
      dataTexture: WebGLTexture, _iteration: number,
      startingKNNTexture: WebGLTexture, targetTexture?: WebGLTexture) {
    knn_util.executeKNNProgram(
        this.gpgpu, this.kNNDescentProgram, dataTexture, startingKNNTexture,
        _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
  }
}
