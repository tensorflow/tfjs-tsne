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
import * as drawer_util from './drawer_util';
import * as gl_util from './gl_util';

export interface DrawableEmbedding {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  embeddingTexture: WebGLTexture;
  pntsPerRow: number;
  numRows: number;
  numPoints: number;
}

export class EmbeddingDrawer {
  private numPoints: number;
  private gpgpu: tf.webgl.GPGPUContext;

  private simpleEmbeddingDrawerProgram: WebGLProgram;
  private coloredEmbeddingDrawerProgram: WebGLProgram;
  private splatTextureDrawerProgram: WebGLProgram;
  private texturedPointsDrawerProgram: WebGLProgram;

  private colorScaleTex: WebGLTexture;
  private vertexIdBuffer: WebGLBuffer;

  constructor(numPoints: number) {
    this.numPoints = numPoints;
    const backend = tf.ENV.findBackend('webgl') as tf.webgl.MathBackendWebGL;
    this.gpgpu = backend.getGPGPUContext();
    this.initializePrograms();
  }

  drawPoints(
      embedding: DrawableEmbedding, alpha: number, targetTex?: WebGLTexture) {
    drawer_util.executeSimpleEmbeddingDrawerProgram(
        this.gpgpu, this.simpleEmbeddingDrawerProgram,
        embedding.embeddingTexture, embedding.numPoints, embedding.minX,
        embedding.minY, embedding.maxX, embedding.maxY, embedding.pntsPerRow,
        embedding.numRows, this.vertexIdBuffer, alpha, 800, targetTex);
  }

  drawColoredPoints(
      embedding: DrawableEmbedding, alpha: number, colors: WebGLTexture,
      targetTex?: WebGLTexture) {
    drawer_util.executeColoredEmbeddingDrawerProgram(
        this.gpgpu, this.coloredEmbeddingDrawerProgram,
        embedding.embeddingTexture, embedding.numPoints, embedding.minX,
        embedding.minY, embedding.maxX, embedding.maxY, embedding.pntsPerRow,
        embedding.numRows, this.vertexIdBuffer, alpha, 800, colors, targetTex);
  }

  drawTexturedPoints(
      embedding: DrawableEmbedding, alpha: number, pointsTexture: WebGLTexture,
      pointTextureDiameter: number, colorsTexture: WebGLTexture,
      targetTex?: WebGLTexture) {
    drawer_util.executeTexturedPointsDrawerProgram(
        this.gpgpu, this.texturedPointsDrawerProgram,
        embedding.embeddingTexture, embedding.numPoints, embedding.minX,
        embedding.minY, embedding.maxX, embedding.maxY, embedding.pntsPerRow,
        embedding.numRows, this.vertexIdBuffer, alpha, 800, pointsTexture,
        pointTextureDiameter, colorsTexture, targetTex);
  }

  drawPointsAndSplatTexture(
      embedding: DrawableEmbedding, splatTexture: WebGLTexture,
      drawnEmbeddingTex: WebGLTexture, textureNormalization: number) {
    drawer_util.executeSplatTextureDrawerProgram(
        this.gpgpu, this.splatTextureDrawerProgram, splatTexture,
        this.colorScaleTex, drawnEmbeddingTex, textureNormalization, 800);
  }

  ////////////////////////////////
  ///// PRIVATE FUNCTIONS  ///////
  ////////////////////////////////

  private initializePrograms() {
    this.simpleEmbeddingDrawerProgram =
        drawer_util.createSimpleEmbeddingDrawerProgram(this.gpgpu);

    this.coloredEmbeddingDrawerProgram =
        drawer_util.createColoredEmbeddingDrawerProgram(this.gpgpu);

    const vertexId = new Float32Array(this.numPoints);

    let i = 0;
    for (i = 0; i < this.numPoints; ++i) {
      vertexId[i] = i;
    }

    this.vertexIdBuffer =
        tf.webgl.webgl_util.createStaticVertexBuffer(this.gpgpu.gl, vertexId);

    this.splatTextureDrawerProgram =
        drawer_util.createSplatTextureDrawerProgram(this.gpgpu);

    this.texturedPointsDrawerProgram =
        drawer_util.createTexturedPointsDrawerProgram(this.gpgpu);

    // Red to blue color scale (obtained from colorbrewer)
    const colors = new Float32Array([
      178, 24,  43,  214, 96,  77,  244, 165, 130, 253, 219, 199, 255, 255,
      255, 209, 229, 240, 146, 197, 222, 67,  147, 195, 33,  102, 172
    ]);

    this.colorScaleTex = gl_util.createAndConfigureInterpolatedTexture(
        this.gpgpu.gl, 9, 1, 3, colors);
  }
}
