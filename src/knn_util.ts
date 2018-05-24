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
import * as gl_util from './gl_util';

export interface RearrangedData {
  numPoints: number;
  pointsPerRow: number;
  pixelsPerPoint: number;
  numRows: number;
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

function generateFragmentShaderSource(
    distanceComputationSource: string, numNeighbors: number): string {
  const source = `#version 300 es
    precision highp float;
    uniform sampler2D data_tex;
    uniform float num_points;
    uniform float points_per_row_knn;
    uniform float num_rows_knn;
    uniform float num_neighs;
    uniform float iteration;

    #define NUM_PACKED_NEIGHBORS ${numNeighbors / 4}

    flat in vec4 knn[NUM_PACKED_NEIGHBORS];
    flat in int point_id;
    in float neighbor_id;

    const float MAX_DIST = 10e30;

    ${distanceComputationSource}

    out vec4 fragmentColor;
    void main() {
      int id = int(neighbor_id/4.);
      int channel = int(mod(neighbor_id,4.)+0.1);

      if(channel == 0) {
        fragmentColor = vec4(knn[id].r,0,0,1);
      }else if(channel == 1) {
        fragmentColor = vec4(knn[id].g,0,0,1);
      }else if(channel == 2) {
        fragmentColor = vec4(knn[id].b,0,0,1);
      }else if(channel == 3) {
        fragmentColor = vec4(knn[id].a,0,0,1);
      }

      //If the neighbor has a valid id i compute the distance squared
      //otherwise I set it to invalid
      if(fragmentColor.r >= 0.) {
        fragmentColor.g = pointDistanceSquared(int(fragmentColor.r),point_id);
      }else{
        fragmentColor.g = MAX_DIST;
      }
    }
  `;
  return source;
}

function generateVariablesAndDeclarationsSource(numNeighbors: number) {
  const source = `
  precision highp float;
  #define NEIGH_PER_ITER 20
  #define NUM_NEIGHBORS ${numNeighbors}
  #define NUM_NEIGHBORS_FLOAT ${numNeighbors}.
  #define NUM_PACKED_NEIGHBORS ${numNeighbors / 4}
  #define MAX_DIST 10e30

  //attributes
  in float vertex_id;
  //uniforms
  uniform sampler2D data_tex;
  uniform sampler2D starting_knn_tex;
  uniform float num_points;
  uniform float points_per_row_knn;
  uniform float num_rows_knn;
  uniform float num_neighs;
  uniform float iteration;

  //output
  //the indices are packed in varying vectors
  flat out vec4 knn[NUM_PACKED_NEIGHBORS];
  //used to recover the neighbor id in the fragment shader
  out float neighbor_id;
  //used to recover the point id in the fragment shader
  //(for recomputing distances)
  flat out int point_id;

  float distances_heap[NUM_NEIGHBORS];
  int knn_heap[NUM_NEIGHBORS];
  `;
  return source;
}

const randomGeneratorSource = `
//Random function developed by Inigo Quilez
//https://www.shadertoy.com/view/llGSzw
float hash1( uint n ) {
    // integer hash copied from Hugo Elias
	  n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);
}

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}
float random( float f ) {
    const uint mantissaMask = 0x007FFFFFu;
    const uint one          = 0x3F800000u;

    uint h = hash( floatBitsToUint( f ) );
    h &= mantissaMask;
    h |= one;

    float  r2 = uintBitsToFloat( h );
    return r2 - 1.0;
}


// #define HASHSCALE1 .1031
// float random(float p) {
// 	vec3 p3  = fract(vec3(p) * HASHSCALE1);
//   p3 += dot(p3, p3.yzx + 19.19);
//   return fract((p3.x + p3.y) * p3.z);
// }

// const vec2 randomConst = vec2(
//   23.14069263277926, // e^pi (Gelfond's constant)
//    2.665144142690225 // 2^sqrt(2) (Gelfond–Schneider constant)
// );
// float random(float seed) {
//     return fract(cos(dot(vec2(seed,seed), randomConst)) * 12345.6789);
// }

`;

// Reads the KNN-graph from the texture and initialize the heap
const distancesInitializationSource = `
//Reads the distances and indices from the knn texture
void initializeDistances(int pnt_id) {
  //row coordinate in the texture
  float row = (floor(float(pnt_id)/points_per_row_knn)+0.5)/num_rows_knn;
  //column of the first neighbor
  float start_col = mod(float(pnt_id),points_per_row_knn)*NUM_NEIGHBORS_FLOAT;
  for(int n = 0; n < NUM_NEIGHBORS; n++) {
    float col = (start_col+float(n)+0.5);
    //normalized by the width of the texture
    col /= (points_per_row_knn*NUM_NEIGHBORS_FLOAT);
    //reads the index in the red channel and the distances in the green one
    vec4 init = texture(starting_knn_tex,vec2(col,row));

    knn_heap[n] = int(init.r);
    distances_heap[n] = init.g;
  }
}
`;

// Code for handling the heap
const knnHeapSource = `
//Swaps two points in the knn-heap
void swap(int i, int j) {
  float swap_value = distances_heap[i];
  distances_heap[i] = distances_heap[j];
  distances_heap[j] = swap_value;
  int swap_id = knn_heap[i];
  knn_heap[i] = knn_heap[j];
  knn_heap[j] = swap_id;
}

//I can make use of the heap property but
//I have to implement a recursive function
bool inTheHeap(float dist_sq, int id) {
  for(int i = 0; i < NUM_NEIGHBORS; ++i) {
    if(knn_heap[i] == id) {
      return true;
    }
  }
  return false;
}

void insertInKNN(float dist_sq, int j) {
  //not in the KNN
  if(dist_sq >= distances_heap[0]) {
    return;
  }

  //the point is already in the KNN
  if(inTheHeap(dist_sq,j)) {
    return;
  }

  //Insert in the new point in the root
  distances_heap[0] = dist_sq;
  knn_heap[0] = j;
  //Sink procedure
  int swap_id = 0;
  while(swap_id*2+1 < NUM_NEIGHBORS) {
    int left_id = swap_id*2+1;
    int right_id = swap_id*2+2;
    if(distances_heap[left_id] > distances_heap[swap_id] ||
        (right_id < NUM_NEIGHBORS &&
                            distances_heap[right_id] > distances_heap[swap_id])
      ) {
      if(distances_heap[left_id] > distances_heap[right_id]
         || right_id >= NUM_NEIGHBORS) {
        swap(swap_id,left_id);
        swap_id = left_id;
      }else{
        swap(swap_id,right_id);
        swap_id = right_id;
      }
    }else{
      break;
    }
  }
}
`;

const vertexPositionSource = `
  //Line positions
  float row = (floor(float(point_id)/points_per_row_knn)+0.5)/num_rows_knn;
  row = row*2.0-1.0;
  if(line_id < int(1)) {
    //for the first vertex only the position is important
    float col = (mod(float(point_id),points_per_row_knn))/(points_per_row_knn);
    col = col*2.0-1.0;
    gl_Position = vec4(col,row,0,1);
    neighbor_id = 0.;
    return;
  }
  //The computation of the KNN happens only for the second vertex
  float col = (mod(float(point_id),points_per_row_knn)+1.)/(points_per_row_knn);
  col = col*2.0-1.0;
  gl_Position = vec4(col,row,0,1);
`;

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createBruteForceKNNProgram(
    gpgpu: tf.webgl.GPGPUContext, numNeighbors: number,
    distanceComputationSource: string): WebGLProgram {
  const vertexShaderSource = `#version 300 es
    ` +
      generateVariablesAndDeclarationsSource(numNeighbors) +
      distancesInitializationSource + distanceComputationSource +
      knnHeapSource + `
    void main() {
      //Getting the id of the point and the line id (0/1)
      point_id = int((vertex_id / 2.0) + 0.1);
      int line_id = int(mod(vertex_id + 0.1, 2.));
      if(float(point_id) >= num_points) {
        return;
      }

      ${vertexPositionSource}

      //////////////////////////////////
      //KNN computation
      initializeDistances(point_id);
      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {
        //TODO make it more readable

        int j = int(mod(
                    float(point_id + i) //point id + current offset
                    + iteration * float(NEIGH_PER_ITER) //iteration offset
                    + 1.25,// +1 for avoid checking the point itself,
                           // +0.25 for error compensation
                    num_points
                  ));
        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j,j+1,j+2,j+3);
        insertInKNN(dist_squared.r, j);
        insertInKNN(dist_squared.g, j+1);
        insertInKNN(dist_squared.b, j+2);
        insertInKNN(dist_squared.a, j+3);
      }

      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {
        knn[n].r = float(knn_heap[n*4]);
        knn[n].g = float(knn_heap[n*4+1]);
        knn[n].b = float(knn_heap[n*4+2]);
        knn[n].a = float(knn_heap[n*4+3]);
      }

      neighbor_id = NUM_NEIGHBORS_FLOAT;
    }
  `;

  const knnFragmentShaderSource =
      generateFragmentShaderSource(distanceComputationSource, numNeighbors);

  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createRandomSamplingKNNProgram(
    gpgpu: tf.webgl.GPGPUContext, numNeighbors: number,
    distanceComputationSource: string): WebGLProgram {
  const vertexShaderSource = `#version 300 es
    ` +
      generateVariablesAndDeclarationsSource(numNeighbors) +
      distancesInitializationSource + randomGeneratorSource +
      distanceComputationSource + knnHeapSource + `
    void main() {
      //Getting the id of the point and the line id (0/1)
      point_id = int((vertex_id/2.0)+0.1);
      int line_id = int(mod(vertex_id+0.1,2.));
      if(float(point_id) >= num_points) {
        return;
      }

      ${vertexPositionSource}

      //////////////////////////////////
      //KNN computation

      initializeDistances(point_id);
      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {
        //BAD SEED
        //uint seed
        //= uint(float(point_id) + float(NEIGH_PER_ITER)*iteration + float(i));
        //GOOD SEED
        //uint seed
        //= uint(float(point_id) + float(num_points)*iteration + float(i));

        float seed
            = float(float(point_id) + float(num_points)*iteration + float(i));
        int j0 = int(random(seed)*num_points);
        int j1 = int(random(seed+1.)*num_points);
        int j2 = int(random(seed+2.)*num_points);
        int j3 = int(random(seed+3.)*num_points);

        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j0,j1,j2,j3);
        if(j0!=point_id)insertInKNN(dist_squared.r, j0);
        if(j1!=point_id)insertInKNN(dist_squared.g, j1);
        if(j2!=point_id)insertInKNN(dist_squared.b, j2);
        if(j3!=point_id)insertInKNN(dist_squared.a, j3);
      }

      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {
        knn[n].r = float(knn_heap[n*4]);
        knn[n].g = float(knn_heap[n*4+1]);
        knn[n].b = float(knn_heap[n*4+2]);
        knn[n].a = float(knn_heap[n*4+3]);
      }
      neighbor_id = NUM_NEIGHBORS_FLOAT;
    }
  `;

  const knnFragmentShaderSource =
      generateFragmentShaderSource(distanceComputationSource, numNeighbors);

  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createKNNDescentProgram(
    gpgpu: tf.webgl.GPGPUContext, numNeighbors: number,
    distanceComputationSource: string): WebGLProgram {
  const vertexShaderSource = `#version 300 es
    ` +
      generateVariablesAndDeclarationsSource(numNeighbors) +
      distancesInitializationSource + randomGeneratorSource +
      distanceComputationSource + knnHeapSource + `
    int fetchNeighborIdFromKNNTexture(int id, int neighbor_id) {
      //row coordinate in the texture
      float row = (floor(float(id)/points_per_row_knn)+0.5)/num_rows_knn;
      //column of the first neighbor
      float start_col = mod(float(id),points_per_row_knn)*NUM_NEIGHBORS_FLOAT;
      //column of the neighbor of interest
      float col = (start_col+float(neighbor_id)+0.5);
      //normalized by the width of the texture
      col /= (points_per_row_knn*NUM_NEIGHBORS_FLOAT);
      //reads the index in the red channel and the distances in the green one
      vec4 knn_link = texture(starting_knn_tex,vec2(col,row));
      //return the index
      return int(knn_link.r);
    }

    int neighborOfANeighbor(int my_id, uint seed) {
      //float random0 = hash1(seed);
      float random0 = random(float(seed));
      // random0 = random0*random0;
      // random0 = 1. - random0;

      //float random1 = hash1(seed*1798191U);
      float random1 = random(float(seed+7U));
      // random1 = random1*random1;
      // random1 = 1. - random1;

      //fetch a neighbor from the heap
      int neighbor = knn_heap[int(random0*NUM_NEIGHBORS_FLOAT)];
      //if it is not a valid pick a random point
      if(neighbor < 0) {
        return int(random(float(seed))*num_points);
      }

      //if it is valid I fetch from the knn graph texture one of its neighbors
      int neighbor2ndDegree = fetchNeighborIdFromKNNTexture(
                                    neighbor,int(random1*NUM_NEIGHBORS_FLOAT));
      //if it is not a valid pick a random point
      if(neighbor2ndDegree < 0) {
        return int(random(float(seed))*num_points);
      }
      return neighbor2ndDegree;
    }

    void main() {
      //Getting the id of the point and the line id (0/1)
      point_id = int((vertex_id/2.0)+0.1);
      int line_id = int(mod(vertex_id+0.1,2.));
      if(float(point_id) >= num_points) {
        return;
      }
      ${vertexPositionSource}

      //////////////////////////////////
      //KNN computation
      initializeDistances(point_id);
      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {
        //BAD SEED
        //uint seed
        //= uint(float(point_id) + float(NEIGH_PER_ITER)*iteration + float(i));
        //GOOD SEED
        uint seed
              = uint(float(point_id) + float(num_points)*iteration + float(i));
        int j0 = neighborOfANeighbor(point_id,seed);
        int j1 = neighborOfANeighbor(point_id,seed+1U);
        int j2 = neighborOfANeighbor(point_id,seed+2U);
        int j3 = neighborOfANeighbor(point_id,seed+3U);

        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j0,j1,j2,j3);
        if(j0!=point_id)insertInKNN(dist_squared.r, j0);
        if(j1!=point_id)insertInKNN(dist_squared.g, j1);
        if(j2!=point_id)insertInKNN(dist_squared.b, j2);
        if(j3!=point_id)insertInKNN(dist_squared.a, j3);
      }

      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {
        knn[n].r = float(knn_heap[n*4]);
        knn[n].g = float(knn_heap[n*4+1]);
        knn[n].b = float(knn_heap[n*4+2]);
        knn[n].a = float(knn_heap[n*4+3]);
      }
      neighbor_id = NUM_NEIGHBORS_FLOAT;
    }
  `;

  const knnFragmentShaderSource =
      generateFragmentShaderSource(distanceComputationSource, numNeighbors);

  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export interface RearrangedData {
  numPoints: number;
  pointsPerRow: number;
  pixelsPerPoint: number;
  numRows: number;
}
export function executeKNNProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, dataTex: WebGLTexture,
    startingKNNTex: WebGLTexture, iteration: number, knnShape: RearrangedData,
    vertexIdBuffer: WebGLBuffer, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  const oldProgram: WebGLProgram = gpgpu.program;
  const oldLineWidth: number = gl.getParameter(gl.LINE_WIDTH);

  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, knnShape.numRows,
        knnShape.pointsPerRow * knnShape.pixelsPerPoint);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  if (oldLineWidth !== 1) {
    gl.lineWidth(1);
  }
  gpgpu.setProgram(program);

  gl.clearColor(0., 0., 0., 0.);
  gl.clear(gl.COLOR_BUFFER_BIT);

  tf.webgl.webgl_util.callAndCheck(
      gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdBuffer));
  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
      gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);

  const dataTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'data_tex');
  gpgpu.setInputMatrixTexture(dataTex, dataTexLoc, 0);

  const startingKNNTexLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'starting_knn_tex');
  gpgpu.setInputMatrixTexture(startingKNNTex, startingKNNTexLoc, 1);

  const iterationLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'iteration');
  gl.uniform1f(iterationLoc, iteration);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, knnShape.numPoints);

  const pntsPerRowKNNLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row_knn');
  gl.uniform1f(pntsPerRowKNNLoc, knnShape.pointsPerRow);

  const numRowsKNNLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows_knn');
  gl.uniform1f(numRowsKNNLoc, knnShape.numRows);

  tf.webgl.webgl_util.callAndCheck(
      gl, () => gl.drawArrays(gl.LINES, 0, knnShape.numPoints * 2));

  // Restore the old program and its vertex buffers
  // TOCHECK if it can be improved
  if (oldProgram != null) {
    gpgpu.setProgram(oldProgram);
    tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(
        gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
  }
  if (oldLineWidth !== 1) {
    gl.lineWidth(oldLineWidth);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createCopyDistancesProgram(gpgpu: tf.webgl.GPGPUContext):
    WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D knn_tex;
    uniform float width;
    uniform float height;

    void main() {
      vec2 coordinates = gl_FragCoord.xy / vec2(width,height);
      float distance = texture2D(knn_tex,coordinates).g;
      gl_FragColor = vec4(distance,0,0,1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeCopyDistancesProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, knnTex: WebGLTexture,
    knnShape: RearrangedData, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, knnShape.numRows,
        knnShape.pointsPerRow * knnShape.pixelsPerPoint);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const knnLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'knn_tex');
  gpgpu.setInputMatrixTexture(knnTex, knnLoc, 0);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'width');
  gl.uniform1f(pntsPerRowLoc, knnShape.pointsPerRow * knnShape.pixelsPerPoint);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'height');
  gl.uniform1f(numRowsLoc, knnShape.numRows);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createCopyIndicesProgram(gpgpu: tf.webgl.GPGPUContext):
    WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D knn_tex;
    uniform float width;
    uniform float height;

    void main() {
      vec2 coordinates = gl_FragCoord.xy / vec2(width,height);
      float id = texture2D(knn_tex,coordinates).r;
      gl_FragColor = vec4(id,0,0,1);

      if(id < 0.) {
        gl_FragColor.b = 1.;
      }
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeCopyIndicesProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, knnTex: WebGLTexture,
    knnShape: RearrangedData, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, knnShape.numRows,
        knnShape.pointsPerRow * knnShape.pixelsPerPoint);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const knnLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'knn_tex');
  gpgpu.setInputMatrixTexture(knnTex, knnLoc, 0);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'width');
  gl.uniform1f(pntsPerRowLoc, knnShape.pointsPerRow * knnShape.pixelsPerPoint);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'height');
  gl.uniform1f(numRowsLoc, knnShape.numRows);

  gpgpu.executeProgram();
}
