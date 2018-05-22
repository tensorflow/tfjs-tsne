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
import {RearrangedData} from './interfaces';

// Returns the GLSL source code needed for
// computing distances in a rearranged data format
// It assumes that dimensions are distributed in the RGBA channels
export function generateDistanceComputationSource(format: RearrangedData):
    string {
  const source = `
    #define DATA_NUM_PACKED_DIMENSIONS ${format.pixelsPerPoint}.
    #define DATA_POINTS_PER_ROW ${format.pointsPerRow}.
    #define DATA_NUM_ROWS ${format.numRows}.
    #define TEXTURE_WIDTH ${format.pointsPerRow * format.pixelsPerPoint}.

    //returns the texture coordinate for point/dimension
    vec2 dataTexCoordinates(int id, int dimension) {
      float id_f = float(id);
      float row = (floor(id_f/DATA_POINTS_PER_ROW)+0.5) / DATA_NUM_ROWS;
      float col = ((mod(id_f,DATA_POINTS_PER_ROW)*(DATA_NUM_PACKED_DIMENSIONS)
                  + float(dimension)) + 0.5) / (TEXTURE_WIDTH);
      return vec2(col,row);
    }

    //compute the euclidean squared distances between two points i and j
    float pointDistanceSquared(int i, int j) {
      vec4 result = vec4(0,0,0,0);
      int num_iter = int(DATA_NUM_PACKED_DIMENSIONS);
      for(int d = 0; d < num_iter; ++d) {
        vec4 vi = texture(data_tex,dataTexCoordinates(i,d));
        vec4 vj = texture(data_tex,dataTexCoordinates(j,d));
        result += (vi-vj)*(vi-vj);
      }
      return (result.r+result.g+result.b+result.a);
    }

    //compute the euclidean squared distances between two points i and j
    vec4 pointDistanceSquaredBatch(int i, int j0, int j1, int j2, int j3) {
      vec4 result = vec4(0,0,0,0);
      int num_iter = int(DATA_NUM_PACKED_DIMENSIONS);
      for(int d = 0; d < num_iter; ++d) {
        vec4 vi = texture(data_tex,dataTexCoordinates(i,d));
        vec4 vj0 = texture(data_tex,dataTexCoordinates(j0,d));
        vec4 vj1 = texture(data_tex,dataTexCoordinates(j1,d));
        vec4 vj2 = texture(data_tex,dataTexCoordinates(j2,d));
        vec4 vj3 = texture(data_tex,dataTexCoordinates(j3,d));
        vj0 = (vi-vj0); vj0 *= vj0;
        vj1 = (vi-vj1); vj1 *= vj1;
        vj2 = (vi-vj2); vj2 *= vj2;
        vj3 = (vi-vj3); vj3 *= vj3;
        result.r += (vj0.r+vj0.g+vj0.b+vj0.a);
        result.g += (vj1.r+vj1.g+vj1.b+vj1.a);
        result.b += (vj2.r+vj2.g+vj2.b+vj2.a);
        result.a += (vj3.r+vj3.g+vj3.b+vj3.a);
      }
      return result;
    }
    `;
  return source;
}

// I don't think any thing calls this? 

// Returns the GLSL source code needed for
// computing distances between MNIST images
export function generateMNISTDistanceComputationSource(): string {
  const source = `
  #define POINTS_PER_ROW 250.
  #define NUM_ROWS 240.
  #define TEXTURE_WIDTH 3500.
  #define TEXTURE_HEIGHT 3360.
  #define DIGIT_WIDTH 14.
  #define NUM_PACKED_DIMENSIONS 196

  //returns the texture coordinate for point/dimension
  vec2 dataTexCoordinates(int id, int dimension) {
    float id_f = float(id);
    float dimension_f = float(dimension);
    float col = ((mod(id_f,POINTS_PER_ROW)*DIGIT_WIDTH));
    float row = (floor(id_f/POINTS_PER_ROW)*DIGIT_WIDTH);

    return (vec2(col,row)+
            vec2(mod(dimension_f,DIGIT_WIDTH),floor(dimension_f/DIGIT_WIDTH))+
            vec2(0.5,0.5)
            )/
            vec2(TEXTURE_WIDTH,TEXTURE_HEIGHT);
  }

  //compute the euclidean squared distances between two points i and j
  float pointDistanceSquared(int i, int j) {
    vec4 result = vec4(0,0,0,0);
    for(int d = 0; d < NUM_PACKED_DIMENSIONS; d+=1) {
      vec4 vi = texture(data_tex,dataTexCoordinates(i,d));
      vec4 vj = texture(data_tex,dataTexCoordinates(j,d));
      result += (vi-vj)*(vi-vj);
    }
    return (result.r+result.g+result.b+result.a);
  }

  //compute the euclidean squared distances between two points i and j
  vec4 pointDistanceSquaredBatch(int i, int j0, int j1, int j2, int j3) {
    vec4 result = vec4(0,0,0,0);
    for(int d = 0; d < NUM_PACKED_DIMENSIONS; d+=1) {
      vec4 vi = texture(data_tex,dataTexCoordinates(i,d));
      vec4 vj0 = texture(data_tex,dataTexCoordinates(j0,d));
      vec4 vj1 = texture(data_tex,dataTexCoordinates(j1,d));
      vec4 vj2 = texture(data_tex,dataTexCoordinates(j2,d));
      vec4 vj3 = texture(data_tex,dataTexCoordinates(j3,d));
      vj0 = (vi-vj0); vj0 *= vj0;
      vj1 = (vi-vj1); vj1 *= vj1;
      vj2 = (vi-vj2); vj2 *= vj2;
      vj3 = (vi-vj3); vj3 *= vj3;
      result.r += (vj0.r+vj0.g+vj0.b+vj0.a);
      result.g += (vj1.r+vj1.g+vj1.b+vj1.a);
      result.b += (vj2.r+vj2.g+vj2.b+vj2.a);
      result.a += (vj3.r+vj3.g+vj3.b+vj3.a);
    }
    return result;
  }
  `;
  return source;
}

// Generates the texture for a KNN containing a number of synthetic clusters
export function generateKNNClusterTexture(
    numPoints: number, numClusters: number,
    numNeighbors: number): {knnGraph: WebGLTexture, dataShape: RearrangedData} {
  // Computing data shape
  const pointsPerRow =
      Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors);
  const numRows = Math.ceil(numPoints / pointsPerRow);
  const dataShape =
      {numPoints, pixelsPerPoint: numNeighbors, numRows, pointsPerRow};

  // Initializing the kNN
  const pointsPerCluster = Math.ceil(numPoints / numClusters);
  const textureValues =
      new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
  for (let i = 0; i < numPoints; ++i) {
    const clusterId = Math.floor(i / pointsPerCluster);
    for (let n = 0; n < numNeighbors; ++n) {
      const id = (i * numNeighbors + n) * 2;
      textureValues[id] = Math.floor(Math.random() * pointsPerCluster) +
          clusterId * pointsPerCluster;
      textureValues[id + 1] = Math.random();
    }
  }

  // Generating texture
  const backend = tfc.ENV.findBackend('webgl') as tfc.webgl.MathBackendWebGL;
  if (backend === null) {
    throw Error('WebGL backend is not available');
  }
  const gpgpu = backend.getGPGPUContext();
  const knnGraph = gl_util.createAndConfigureTexture(
      gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);

  return {knnGraph, dataShape};
}

// Generates the texture for a KNN containing a synthetic line
export function generateKNNLineTexture(numPoints: number, numNeighbors: number):
    {knnGraph: WebGLTexture, dataShape: RearrangedData} {
  // Computing data shape
  const pointsPerRow =
      Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors);
  const numRows = Math.ceil(numPoints / pointsPerRow);
  const dataShape =
      {numPoints, pixelsPerPoint: numNeighbors, numRows, pointsPerRow};

  // Initializing the kNN
  const textureValues =
      new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
  for (let i = 0; i < numPoints; ++i) {
    for (let n = 0; n < numNeighbors; ++n) {
      const id = (i * numNeighbors + n) * 2;
      // Neigh
      textureValues[id] =
          Math.floor(i + n - (numNeighbors / 2) + numPoints) % numPoints;
      textureValues[id + 1] = 1;
    }
  }

  // Generating texture
  const backend = tfc.ENV.findBackend('webgl') as tfc.webgl.MathBackendWebGL;
  if (backend === null) {
    throw Error('WebGL backend is not available');
  }
  const gpgpu = backend.getGPGPUContext();
  const knnGraph = gl_util.createAndConfigureTexture(
      gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);

  return {knnGraph, dataShape};
}

// Does anything other than the tests use these functions? If not they should to
// the test file or a test utils file. This would make it easier to follow what
// key to the implementation. 

// Generates the texture for a KNN containing a number of synthetic clusters
export function generateKNNClusterData(
    numPoints: number, numClusters: number,
    numNeighbors: number): {distances: Float32Array, indices: Uint32Array} {
  const pointsPerCluster = Math.ceil(numPoints / numClusters);
  const distances = new Float32Array(numPoints * numNeighbors);
  const indices = new Uint32Array(numPoints * numNeighbors);

  for (let i = 0; i < numPoints; ++i) {
    const clusterId = Math.floor(i / pointsPerCluster);
    for (let n = 0; n < numNeighbors; ++n) {
      const id = (i * numNeighbors + n);
      distances[id] = Math.random();
      indices[id] = Math.floor(Math.random() * pointsPerCluster) +
          clusterId * pointsPerCluster;
    }
  }
  return {distances, indices};
}

// Generates the texture for a KNN containing a synthetic line
export function generateKNNLineData(numPoints: number, numNeighbors: number):
    {distances: Float32Array, indices: Uint32Array} {
  // Initializing the kNN
  const distances = new Float32Array(numPoints * numNeighbors);
  const indices = new Uint32Array(numPoints * numNeighbors);

  for (let i = 0; i < numPoints; ++i) {
    for (let n = 0; n < numNeighbors; ++n) {
      const id = (i * numNeighbors + n);
      distances[id] = 1;
      indices[id] =
          Math.floor(i + n - (numNeighbors / 2) + numPoints) % numPoints;
    }
  }
  return {distances, indices};
}
