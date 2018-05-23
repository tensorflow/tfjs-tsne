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

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createEmbeddingSplatterProgram(gpgpu: tf.webgl.GPGPUContext):
    WebGLProgram {
  const vertexShaderSource = `#version 300 es
    precision highp float;
    in float vertex_id;

    uniform sampler2D embedding_tex;
    uniform vec2 minV;
    uniform vec2 maxV;
    uniform float kernel_support;
    uniform float points_per_row;
    uniform float num_rows;

    out vec2 kernel_coords;

    void main() {
      //TODO Clean up and check performance loss due to the conversions
      uint pnt_id = uint((vertex_id / 4.0) + 0.1);
      uint quad_id = uint(mod(vertex_id + 0.1,4.));

      uint row    = uint((float(pnt_id) + 0.1)/points_per_row);
      uint column = uint(float(pnt_id) - float(row) * points_per_row);

      float width = (points_per_row * 2.0);
      float row_tex = (float(row) + 0.5) / num_rows;
      vec2 tex_coords_x = vec2((float(column) * 2. + 0.5) / width, row_tex);
      vec2 tex_coords_y = vec2((float(column) * 2. + 1.5) / width, row_tex);

      float x_pnt = texture(embedding_tex,tex_coords_x).r;
      float y_pnt = texture(embedding_tex,tex_coords_y).r;
      vec2 vertex_coords = vec2(x_pnt,y_pnt);

      if(quad_id == uint(0)) {kernel_coords = vec2(-1,-1);}
      else if(quad_id == uint(1)) {kernel_coords = vec2(1,-1);}
      else if(quad_id == uint(2)) {kernel_coords = vec2(1,1);}
      else if(quad_id == uint(3)) {kernel_coords = vec2(-1,1);}

      vertex_coords += kernel_coords * kernel_support;      // embedding space
      vertex_coords = (vertex_coords - minV) / (maxV-minV); //  0:1 space
      vertex_coords = vertex_coords * 2.0 - 1.0;            // -1:1 space

      gl_Position = vec4(vertex_coords,0,1);
    }
  `;
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    uniform sampler2D kernel_tex;
    in vec2 kernel_coords;
    out vec4 fragmentColor;

    void main() {
      fragmentColor = texture(kernel_tex,(kernel_coords + 1.) / 2.0);
    }
  `;
  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, fragmentShaderSource);
}

export function executeEmbeddingSplatterProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram,
    targetTex: WebGLTexture, embeddingTex: WebGLTexture,
    kernelTex: WebGLTexture, targetTexDiameter: number, numPoints: number,
    minX: number, minY: number, maxX: number, maxY: number,
    kernelSupport: number, pntsPerRow: number, numRows: number,
    vertexIdBuffer: WebGLBuffer) {
  const gl = gpgpu.gl;
  const oldProgram: WebGLProgram = gpgpu.program;

  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, targetTexDiameter, targetTexDiameter);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  gl.clearColor(0., 0., 0., 0.);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);

  tf.webgl.webgl_util.callAndCheck(
      gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdBuffer));

  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
      gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);

  const embeddingLocation =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);

  const kernelLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'kernel_tex');
  gpgpu.setInputMatrixTexture(kernelTex, kernelLocation, 1);

  const kernelSupportLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'kernel_support');
  gl.uniform1f(kernelSupportLoc, kernelSupport);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const minLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
  gl.uniform2f(minLoc, minX, minY);

  const maxLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
  gl.uniform2f(maxLoc, maxX, maxY);

  tf.webgl.webgl_util.callAndCheck(
      gl, () => gl.drawArrays(gl.TRIANGLES, 0, numPoints * 2 * 3));

  gl.disable(gl.BLEND);

  // Restore the old program and its vertex buffers
  // TOCHECK if it can be improved
  if (oldProgram != null) {
    gpgpu.setProgram(oldProgram);
    tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(
        gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createQInterpolatorProgram(gpgpu: tf.webgl.GPGPUContext):
    WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D embedding_tex;
    uniform sampler2D splat_tex;
    uniform vec2 minV;
    uniform vec2 maxV;
    uniform float points_per_row;
    uniform float num_rows;
    uniform float num_points;

    void main() {
      vec2 pnt_location = gl_FragCoord.xy - vec2(0.5,0.5);

      if(pnt_location.y * points_per_row + pnt_location.x >= num_points) {
        gl_FragColor = vec4(0,0,0,0);
        return;
      }

      float emb_width = (points_per_row * 2.0);
      float emb_row_coord = (pnt_location.y + 0.5) / num_rows;
      vec2 emb_coords_x
              = vec2((pnt_location.x * 2.+0.5) / emb_width, emb_row_coord);
      vec2 emb_coords_y
              = vec2((pnt_location.x * 2. + 1.5) / emb_width, emb_row_coord);

      float x_pnt = texture2D(embedding_tex,emb_coords_x).r;
      float y_pnt = texture2D(embedding_tex,emb_coords_y).r;

      vec2 splat_coords = vec2(x_pnt,y_pnt);
      splat_coords = (splat_coords - minV) / (maxV - minV); //  0:1 space

      float q = (texture2D(splat_tex,splat_coords).r - 1.);

      gl_FragColor = vec4(q, 0, 0, 1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeQInterpolatorProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, splatTex: WebGLTexture,
    embeddingTex: WebGLTexture, numPoints: number, minX: number, minY: number,
    maxX: number, maxY: number, pntsPerRow: number, numRows: number,
    targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const embeddingLocation =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);

  const splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'splat_tex');
  gpgpu.setInputMatrixTexture(splatTex, splatLocation, 1);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, numPoints);

  const minLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
  gl.uniform2f(minLoc, minX, minY);

  const maxLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
  gl.uniform2f(maxLoc, maxX, maxY);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createXYInterpolatorProgram(gpgpu: tf.webgl.GPGPUContext):
    WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D embedding_tex;
    uniform sampler2D splat_tex;
    uniform vec2 minV;
    uniform vec2 maxV;
    uniform float points_per_row;
    uniform float num_rows;
    uniform float num_points;
    uniform float eta;

    void main() {
      vec2 pnt_location = gl_FragCoord.xy - vec2(0.5,0.5);
      pnt_location.x = floor(pnt_location.x/2.+0.1);

      if(pnt_location.y*points_per_row + pnt_location.x >= num_points) {
        gl_FragColor = vec4(0,0,0,0);
        return;
      }

      float emb_width = (points_per_row * 2.0);
      float emb_row_coord = (pnt_location.y + 0.5) / num_rows;
      vec2 emb_coords_x
              = vec2((pnt_location.x * 2. + 0.5) / emb_width, emb_row_coord);
      vec2 emb_coords_y
              = vec2((pnt_location.x * 2. + 1.5) / emb_width, emb_row_coord);

      float x_pnt = texture2D(embedding_tex,emb_coords_x).r;
      float y_pnt = texture2D(embedding_tex,emb_coords_y).r;

      vec2 splat_coords = vec2(x_pnt,y_pnt);
      splat_coords = (splat_coords - minV) / (maxV - minV); //  0:1 space

      float q = 0.;
      if(mod(gl_FragCoord.x - 0.5,2.) < 0.5 ) {
        q = texture2D(splat_tex,splat_coords).g * eta * 2.;
      }else{
        q = texture2D(splat_tex,splat_coords).b * eta * 2.;
      }

      gl_FragColor = vec4(q,0.0,0.0,1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeXYInterpolatorProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, splatTex: WebGLTexture,
    embeddingTex: WebGLTexture, targetTex: WebGLTexture, numPoints: number,
    minX: number, minY: number, maxX: number, maxY: number, pntsPerRow: number,
    numRows: number, eta: number) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const embeddingLocation =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);

  const splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'splat_tex');
  gpgpu.setInputMatrixTexture(splatTex, splatLocation, 1);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, numPoints);

  const etaLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'eta');
  gl.uniform1f(etaLoc, eta);

  const minLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
  gl.uniform2f(minLoc, minX, minY);

  const maxLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
  gl.uniform2f(maxLoc, maxX, maxY);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createAttractiveForcesComputationProgram(
    gpgpu: tf.webgl.GPGPUContext): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;

    uniform sampler2D embedding_tex;
    uniform sampler2D offset_tex;
    uniform sampler2D neigh_id_tex;
    uniform sampler2D neigh_prob_tex;

    uniform float points_per_row;
    uniform float num_rows;
    uniform float num_points;
    uniform float num_neighs_per_row;
    uniform float eta;

    void main() {
      //add for nearest pixel interpolation
      vec2 half_pxl = vec2(0.5,0.5);

      // Dimension of the fragment
      // 0 -> x :1 -> y
      float dimension = mod(gl_FragCoord.x - 0.4,2.);

      //Point location in the [points_per_row,num_rows] space
      vec2 i_location = gl_FragCoord.xy - half_pxl;
      i_location.x = floor(i_location.x / 2. + 0.1);

      //just an extra fragment -> return
      if(i_location.y*points_per_row + i_location.x >= num_points) {
        gl_FragColor = vec4(0,0,0,0);
        return;
      }

      //Offset coordinates for the point
      vec2 offset_coord = (i_location + half_pxl) /
                                              vec2(points_per_row,num_rows);
      //Offset information ...
      vec4 offset_info  = texture2D(offset_tex,offset_coord);
      //... contains the number of neighbors for the point ...
      float num_neighs  = offset_info.z;
      //... and the coordinates of the firts neigh in the neigh textures
      vec2 offset_neigh = offset_info.xy;

      //Computing the coordinates of the point in the texture
      //_i represent the point to move, _j the neighbors
      float emb_width = (points_per_row * 2.0);
      float emb_row_i = (i_location.y + 0.5) / num_rows;
      vec2 x_i_coord = vec2((i_location.x * 2. + 0.5) / emb_width, emb_row_i);
      vec2 y_i_coord = vec2((i_location.x * 2. + 1.5) / emb_width, emb_row_i);
      //getting the coordinates in the embedding
      float x_i = texture2D(embedding_tex,x_i_coord).r;
      float y_i = texture2D(embedding_tex,y_i_coord).r;

      //Sum of all attractive forces
      float sum_pos = 0.;

      //Can't be higher than 1000 (perplexity is usually around 30)
      //and a 'while' can't be used
      for(int n = 0; n < 2000; ++n) {
        //Actual check on number of neighbors
        if(float(n) >= num_neighs) {
          break;
        }

        //Get the id and the probability for the neighbor
        float pij = texture2D(neigh_prob_tex,
                              (offset_neigh + half_pxl) / num_neighs_per_row
                             ).r;
        float neigh_id = texture2D(neigh_id_tex,
                                  (offset_neigh + half_pxl) / num_neighs_per_row
                                  ).r;

        //Getting the coordinates of the neighbor
        vec2 j_location = vec2(mod(neigh_id + 0.1, points_per_row),
                               floor(neigh_id / points_per_row + 0.1));
        float emb_row_j = (j_location.y + 0.5) / num_rows;
        vec2 x_j_coord = vec2((j_location.x * 2. + 0.5) / emb_width, emb_row_j);
        vec2 y_j_coord = vec2((j_location.x * 2. + 1.5) / emb_width, emb_row_j);
        float x_j = texture2D(embedding_tex,x_j_coord).r;
        float y_j = texture2D(embedding_tex,y_j_coord).r;

        //Actual computation of the attractive forces
        float dist_x    = (x_i - x_j);
        float dist_y    = (y_i - y_j);
        float qij       = 1. / (1. + dist_x * dist_x + dist_y * dist_y);
        //the update depends on the dimension that this fragment represents
        if(dimension < 0.5) {
          // * 4 / (num_points*2) -> * 2 / num_points
          sum_pos += eta * 2. * pij * qij * dist_x / (num_points);
        }else{
          sum_pos += eta * 2. * pij * qij * dist_y / (num_points);
        }

        //Increase the coordinate of the neigh in the neigh_id texture
        offset_neigh.x += 1.;
        //check if the new neigh is in the next row
        if(offset_neigh.x + 0.2 > num_neighs_per_row) {
          //in that case reset the column and increase the row
          offset_neigh.x = 0.1;
          offset_neigh.y += 1.0;
        }
      }

      //The output is the sum of the attractive forces
      gl_FragColor = vec4(sum_pos,0,0,0);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeAttractiveForcesComputationProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram,
    embeddingTex: WebGLTexture, offsetTex: WebGLTexture,
    neighIdTex: WebGLTexture,  // Float for now...
    // better to use an integer texture
    neighProbTex: WebGLTexture, numPoints: number, neighsPerRow: number,
    pntsPerRow: number, numRows: number, eta: number,
    targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const embeddingLocation =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 3);

  const offsetLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'offset_tex');
  gpgpu.setInputMatrixTexture(offsetTex, offsetLocation, 2);

  const neighIdLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'neigh_id_tex');
  gpgpu.setInputMatrixTexture(neighIdTex, neighIdLocation, 1);

  const neighProbLocation =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'neigh_prob_tex');
  gpgpu.setInputMatrixTexture(neighProbTex, neighProbLocation, 0);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const etaLoc =
      tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'eta');
  gl.uniform1f(etaLoc, eta);

  const neighsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_neighs_per_row');
  gl.uniform1f(neighsPerRowLoc, neighsPerRow);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, numPoints);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createEmbeddingInitializationProgram(
    gpgpu: tf.webgl.GPGPUContext): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;

    uniform sampler2D random_tex;
    uniform float points_per_row;
    uniform float num_rows;
    uniform float num_points;

    void main() {
      //add for nearest pixel interpolation
      vec2 half_pxl = vec2(0.5,0.5);

      // Dimension of the fragment
      // 0 -> x :1 -> y
      float dimension = mod(gl_FragCoord.x - 0.4,2.);
      vec2 pnt_location = gl_FragCoord.xy - half_pxl;
      pnt_location.x = floor(pnt_location.x / 2.);

      //just an extra fragment -> return
      if(pnt_location.y*points_per_row + pnt_location.x >= num_points) {
        gl_FragColor = vec4(0,0,0,1);
        return;
      }

      float width = (points_per_row * 2.0);
      float row_coord = (pnt_location.y + 0.5)/num_rows;
      vec2 rad_coord = vec2((pnt_location.x * 2. + 0.5) / width, row_coord);
      vec2 ang_coord = vec2((pnt_location.x * 2. + 1.5) / width, row_coord);

      float rad = texture2D(random_tex,rad_coord).r * 3.;
      float ang = texture2D(random_tex,ang_coord).r * 3.1415 * 2.;

      gl_FragColor = vec4(rad,ang,0,1);

      if(dimension < 0.5) {
        gl_FragColor = vec4(cos(ang) * rad,0,0,0);
      }else{
        gl_FragColor = vec4(sin(ang) * rad,0,0,0);
      }
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeEmbeddingInitializationProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram,
    randomTex: WebGLTexture, numPoints: number, pntsPerRow: number,
    numRows: number, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const randomLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'random_tex');
  gpgpu.setInputMatrixTexture(randomTex, randomLoc, 3);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, numPoints);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createDistributionParametersComputationProgram(
    gpgpu: tf.webgl.GPGPUContext): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;

    #define MAX_NEIGHBORS 128
    #define MAX_ITERATIONS 500
    #define FLOAT_MAX 10e30
    #define TOLERANCE 1e-5

    uniform sampler2D knn_graph_tex;
    uniform float points_per_row;
    uniform float num_rows;
    uniform float num_points;
    uniform float num_neighs;
    uniform float perplexity;

    vec2 half_pixel = vec2(0.5,0.5);
    float distances_squared[MAX_NEIGHBORS];

    void readDistances(vec2 point_location) {
      for(int n = 0; n < MAX_NEIGHBORS; ++n ) {
        if(float(n) >= num_neighs-0.1) {
          break;
        }
        vec2 knn_coordinates = vec2(
            (point_location.x * num_neighs + float(n) + half_pixel.x)
                                        /(points_per_row * num_neighs),
            (point_location.y + half_pixel.y) / num_rows
        );
        distances_squared[n] = texture2D(knn_graph_tex,knn_coordinates).g;
      }
    }

    void main() {
      vec2 point_location = gl_FragCoord.xy - half_pixel;
      //invalid points
      if(point_location.y*points_per_row + point_location.x >= num_points) {
        gl_FragColor = vec4(0,0,0,0);
        return;
      }
      readDistances(point_location);

      //Beta computation
      float beta = 1.;
      float max_beta = FLOAT_MAX;
      float min_beta = -FLOAT_MAX;
      //To avoid computing the log at every iteration
      float log_perplexity = log(perplexity);
      float entropy_diff = 0.;
      float entropy = 0.;
      float sum_probabilities = 0.;

      //Binary search for a maximum of MAX_ITERATIONS
      for(int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        //At every iteration I compute the
        //entropy enforced by the current beta
        sum_probabilities = 0.;
        entropy = 0.;
        for(int n = 0; n < MAX_NEIGHBORS; ++n ) {
          if(float(n) >= num_neighs-0.1) {
            break;
          }
          float neigh_probability = exp(-beta * distances_squared[n]);
          sum_probabilities += neigh_probability;
          entropy += beta * distances_squared[n] * neigh_probability;
        }

        entropy = entropy / sum_probabilities + log(sum_probabilities);
        entropy_diff = entropy - log_perplexity;

        //the current beta is good enough!
        if(entropy_diff < TOLERANCE && -entropy_diff < TOLERANCE) {
          break;
        }

        if(entropy_diff > 0.) {
          min_beta = beta;
          if(max_beta == FLOAT_MAX || max_beta == -FLOAT_MAX) {
            beta *= 2.;
          }else{
            beta = (beta + max_beta) / 2.;
          }
        }else{
          max_beta = beta;
          if(min_beta == -FLOAT_MAX || min_beta == FLOAT_MAX) {
            beta /= 2.;
          }else{
            beta = (beta + min_beta) / 2.;
          }
        }
      }
      gl_FragColor = vec4(beta,sum_probabilities,0,1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeDistributionParametersComputationProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, knnGraph: WebGLTexture,
    numPoints: number, numNeighs: number, pntsPerRow: number, numRows: number,
    perplexity: number, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const knnGraphLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'knn_graph_tex');
  gpgpu.setInputMatrixTexture(knnGraph, knnGraphLoc, 0);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, numPoints);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numNeighsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_neighs');
  gl.uniform1f(numNeighsLoc, numNeighs);

  const perplexityLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'perplexity');
  // TODO PASS AS A PARAMETER
  gl.uniform1f(perplexityLoc, perplexity);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createGaussiaDistributionsFromDistancesProgram(
    gpgpu: tf.webgl.GPGPUContext): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D knn_graph_tex;
    uniform sampler2D parameters_tex;
    uniform float points_per_row;
    uniform float num_rows;
    uniform float num_points;
    uniform float num_neighs;

    vec2 half_pixel = vec2(0.5,0.5);

    void main() {
      vec2 point_location = gl_FragCoord.xy - half_pixel;
      point_location.x = floor(point_location.x / num_neighs);

      //invalid points
      if(point_location.y*points_per_row + point_location.x >= num_points) {
        gl_FragColor = vec4(0,0,0,0);
        return;
      }
      float distance_squared
            = texture2D(knn_graph_tex,
                        gl_FragCoord.xy /
                        vec2(points_per_row*num_neighs,num_rows)
                      ).g;
      vec2 parameters
            = texture2D(parameters_tex,
                        (point_location.xy + half_pixel)/
                        vec2(points_per_row,num_rows)
                      ).rg;
      float beta = parameters.r;
      float normalization = parameters.g;

      float probability = exp(-beta * distance_squared) / normalization;
      //check for NaN for degenerated knn (d = 0 for every point)
      if (!(probability < 0.0 || 0.0 < probability || probability == 0.0)) {
        probability = 0.;
      }

      gl_FragColor = vec4(probability,0,0,1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeGaussiaDistributionsFromDistancesProgram(
    gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, knnGraph: WebGLTexture,
    parameters: WebGLTexture, numPoints: number, numNeighs: number,
    pntsPerRow: number, numRows: number, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * numNeighs);
  } else {
    tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const knnGraphLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'knn_graph_tex');
  gpgpu.setInputMatrixTexture(knnGraph, knnGraphLoc, 0);

  const parametersLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'parameters_tex');
  gpgpu.setInputMatrixTexture(parameters, parametersLoc, 1);

  const numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_points');
  gl.uniform1f(numPointsLoc, numPoints);

  const pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'points_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numNeighsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_neighs');
  gl.uniform1f(numNeighsLoc, numNeighs);

  gpgpu.executeProgram();
}
