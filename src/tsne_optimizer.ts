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
import {RearrangedData} from './interfaces';
import * as knn_util from './knn_util';
import * as tsne_util from './tsne_optimizer_util';

export class TSNEOptimizer {
  // Interactive parameters
  private _eta: number; // used as uniform in the shaders
  private _spacePerPixel: number;
  private _minimumGain: number;
  private _momentum: tf.Scalar;
  //@ts-ignore
  private _finalMomentum: tf.Scalar;
  private _applyGain: boolean;
  //@ts-ignore
  private _momSwitchIter: number;
  private _exaggeration: tf.Scalar;
  private _exaggerationNumber: number;

  private rawExaggeration: number|Array<{iteration : number, value: number}>;

  private verbose: boolean;

  private numPoints: number;
  private pointsPerRow: number;
  private numRows: number;
  private splatTextureDiameter: number;
  private kernelTextureDiameter: number;
  private kernelSupport: number;

  private embedding: tf.Tensor;
  private gradient: tf.Tensor;
  private gain: tf.Tensor;
  private minGain: tf.Tensor;
  private backend: tf.webgl.MathBackendWebGL;
  private gpgpu: tf.webgl.GPGPUContext;

  private embeddingInitializationProgram: WebGLProgram;
  private embeddingSplatterProgram: WebGLProgram;
  private qInterpolatorProgram: WebGLProgram;
  private xyInterpolatorProgram: WebGLProgram;
  private attractiveForcesProgram: WebGLProgram;
  private distributionParameterssComputationProgram: WebGLProgram;
  private gaussiaDistributionsFromDistancesProgram: WebGLProgram;

  // Repulsive forces computation
  private _splatTexture: WebGLTexture;
  private kernelTexture: WebGLTexture;
  private splatVertexIdBuffer: WebGLBuffer;

  // Attractive forces computation
  private probOffsetTexture: WebGLTexture;
  private probNeighIdTexture: WebGLTexture;
  private probTexture: WebGLTexture;
  private numNeighPerRow: number;

  private _minX: number;
  private _maxX: number;
  private _minY: number;
  private _maxY: number;
  private _normQ: number;
  private _iteration: number;

  ////////////////////////////////

  // getters
  get minX(): number { return this._minX; }
  get maxX(): number { return this._maxX; }
  get minY(): number { return this._minY; }
  get maxY(): number { return this._maxY; }
  get iteration(): number { return this._iteration; }
  get numberOfPoints() { return this.numPoints; }
  get numberOfPointsPerRow() { return this.pointsPerRow; }
  get numberOfRows() { return this.numRows; }
  get embeddingCoordinates() { return this.embedding; }

  get embedding2D(): tf.Tensor {
    const result = tf.tidy(() => {
      // 2d tensor with some extra points
      const reshaped =
          this.embedding.reshape([ this.numRows * this.pointsPerRow, 2 ])
              .slice([ 0, 0 ], [ this.numPoints, 2 ]);
      return reshaped;
    });
    return result;
  }

  get embeddingTexture() {
    return this.backend.getTexture(this.embedding.dataId);
  }
  get splatTexture() { return this._splatTexture; }
  get normalizationQ() { return this._normQ; }

  // getters and settters for tSNE GD parameters
  get exaggerationAtCurrentIteration(): number {
    return this._exaggeration.get();
  }
  get exaggeration(): number|Array<{iteration : number, value: number}> {
    return this.rawExaggeration;
  }
  set exaggeration(ex: number|Array<{iteration : number, value: number}>) {
    this.rawExaggeration = ex;
    if (typeof ex === 'number') {
      // Number
      if (ex < 1) {
        throw Error('Exaggeration must be greater then or equal to one');
      }
    } else {
      // Polyline
      for (let i = 0; i < ex.length; ++i) {
        if (ex[i].value < 1) {
          throw Error('Exaggeration must be greater then or equal to one');
        }
        if (ex[i].iteration < 0) {
          throw Error('Piecewise linear exaggeration function \
                                        must have poistive iteration values');
        }
      }
      for (let i = 0; i < ex.length - 1; ++i) {
        if (ex[i].iteration >= ex[i + 1].iteration) {
          throw Error('Piecewise linear exaggeration function \
                                      must have increasing iteration values');
        }
      }
      // If I get only one value I save it as a number
      if (ex.length === 1) {
        this.exaggeration = ex[0].value;
      }
    }
    this.updateExaggeration();
  }

  get momentum(): number { return this._momentum.get(); }
  set momentum(mom: number) {
    if (mom < 0 || mom > 1) {
      throw Error('Momentum must be in the [0,1] range');
    }
    this._momentum.dispose();
    this._momentum = tf.scalar(mom);
  }

  get applyGain(): boolean { return this._applyGain; }
  set applyGain(applyGain: boolean) {
    this._applyGain = applyGain;
  }

  get eta(): number { return this._eta; }
  set eta(eta: number) {
    if (eta <= 0) {
      throw Error('ETA must be greater then zero');
    }
    this._eta = eta;
  }

  get spacePerPixel(): number {return this._spacePerPixel;}
  set spacePerPixel(spacePerPixel: number) {
      if (spacePerPixel <= 0) {
          throw Error('Space Per Pixel must be greater then zero');
      }
      this._spacePerPixel = spacePerPixel;
  }

  ////////////////////////////////
  ///// PUBLIC INTERFACE  ////////
  ////////////////////////////////

  constructor(numPoints: number, verbose?: boolean,
              splatTextureDiameter?: number, kernelTextureRadius?: number) {
    if (verbose != null) {
      this.verbose = verbose;
    } else {
      this.verbose = false;
    }

    this.log('Initializing the tSNE gradient descent computation...');
    this.numPoints = numPoints;
    this._iteration = 0;

    // WebGL version 2 is required
    const webglVersion = tf.ENV.get('WEBGL_VERSION');
    if (webglVersion === 1) {
      throw Error('WebGL version 1 is not supported by tfjs-tsne');
    }
    // Saving the GPGPU context
    this.backend = tf.ENV.findBackend('webgl') as tf.webgl.MathBackendWebGL;
    if (this.backend === null) {
      throw Error('WebGL backend is not available');
    }
    this.gpgpu = this.backend.getGPGPUContext();

    // Check for the float interpolation extension
    tf.webgl.webgl_util.getExtensionOrThrow(this.gpgpu.gl,
                                            'OES_texture_float_linear');

    // The points are organized as xyxyxy... with a pixel per dimension
    // The resulting texture are almost square
    // to avoid precision problems in the shaders
    this.pointsPerRow = Math.ceil(Math.sqrt(numPoints * 2));
    if (this.pointsPerRow % 2 === 1) {
      ++this.pointsPerRow;
    }
    this.pointsPerRow /= 2;
    this.numRows = Math.ceil(numPoints / this.pointsPerRow);
    this.log('\t# points per row', this.pointsPerRow);
    this.log('\t# rows', this.numRows);

    // Default values for the gradient descent parameters
    this._eta = 200;
    this._spacePerPixel = 0.35;
    this._minimumGain = 0.1;
    this._momentum = tf.scalar(0.5);
    this._finalMomentum = tf.scalar(0.5);
    this._momSwitchIter = 250;
    this.rawExaggeration =
        [ {iteration : 200, value : 4}, {iteration : 600, value : 1} ];
    this.updateExaggeration();

    // Initialization of the splat textures used for the computation
    // of the scalar fields used to approximate the repulsive forces
    if (splatTextureDiameter == null) {
      splatTextureDiameter = 5;
    }
    this.splatTextureDiameter = splatTextureDiameter;
    if (kernelTextureRadius == null) {
      kernelTextureRadius = 50;
    }
    this.kernelTextureDiameter = kernelTextureRadius * 2 + 1;
    this.initializeRepulsiveForceTextures();
    this.log('\tSplat texture diameter', this.splatTextureDiameter);
    this.log('\tKernel texture diameter', this.kernelTextureDiameter);

    // Initialize the WebGL custom programs
    this.initilizeCustomWebGLPrograms();

    // Initialize the initial position of the points and the gradients
    this.initializeEmbedding();
    this.log('\tEmbedding', this.embedding);
    this.log('\tGradient', this.gradient);
  }

  dispose() {
    // Tensors
    this.embedding.dispose();
    this.gradient.dispose();
    this.gain.dispose();
    this.minGain.dispose();
    this._momentum.dispose();
    this._exaggeration.dispose();

    // Textures
    this.gpgpu.gl.deleteTexture(this._splatTexture);
    this.gpgpu.gl.deleteTexture(this.kernelTexture);
    if (this.kernelTexture != null) {
      this.gpgpu.gl.deleteTexture(this.probOffsetTexture);
    }
    if (this.kernelTexture != null) {
      this.gpgpu.gl.deleteTexture(this.probNeighIdTexture);
    }
    if (this.kernelTexture != null) {
      this.gpgpu.gl.deleteTexture(this.probTexture);
    }

    // Buffers
    this.gpgpu.gl.deleteBuffer(this.splatVertexIdBuffer);

    // Programs
    this.gpgpu.gl.deleteProgram(this.embeddingInitializationProgram);
    this.gpgpu.gl.deleteProgram(this.embeddingSplatterProgram);
    this.gpgpu.gl.deleteProgram(this.qInterpolatorProgram);
    this.gpgpu.gl.deleteProgram(this.xyInterpolatorProgram);
    this.gpgpu.gl.deleteProgram(this.attractiveForcesProgram);
    this.gpgpu.gl.deleteProgram(this.distributionParameterssComputationProgram);
    this.gpgpu.gl.deleteProgram(this.gaussiaDistributionsFromDistancesProgram);
  }

  // Randomly initialize the position of the
  initializeEmbedding() {
    if (this.embedding != null) {
      this.embedding.dispose();
    }
    if (this.gradient != null) {
      this.gradient.dispose();
    }
    if (this.gain != null) {
      this.gain.dispose();
    }
    if (this.minGain != null) {
      this.minGain.dispose();
    }

    // Previous gradients are set to zero
    this.gradient = tf.zeros([ this.numRows, this.pointsPerRow * 2 ]);
    this.gain = tf.ones([ this.numRows, this.pointsPerRow * 2 ]);
    // Set a lower limit on the gain
    this.minGain =
      tf.ones([this.numRows, this.pointsPerRow * 2]).
        mul(tf.scalar(this._minimumGain));

    this.embedding = tf.tidy(() => {
      const randomData =
          tf.randomUniform([ this.numRows, this.pointsPerRow * 2 ]);
      const embedding = tf.zeros([ this.numRows, this.pointsPerRow * 2 ]);
      // <bvl2 - In the C++ version there is a multiplier here 0.1>
      this.initializeEmbeddingPositions(embedding, randomData);
      return embedding.mul(tf.scalar(0.1));
    });
    /*this.embedding = this.initializeFixedXorEmbeddingPositions(
      this.numRows, this.pointsPerRow, this.numPoints
    );*/

    // Setting embedding boundaries
    const maxEmbeddingAbsCoordinate = 3;
    this._minX = -maxEmbeddingAbsCoordinate;
    this._minY = -maxEmbeddingAbsCoordinate;
    this._maxX = maxEmbeddingAbsCoordinate;
    this._maxY = maxEmbeddingAbsCoordinate;
    this.log('\tmin X', this._minX);
    this.log('\tmax X', this._maxX);
    this.log('\tmin Y', this._minY);
    this.log('\tmax Y', this._maxY);
    this._iteration = 0;
  }

  // Defines the neighborhood relationships between the points
  initializeNeighbors(numNeighPerRow: number, offsets: WebGLTexture,
                      probabilities: WebGLTexture, neighIds: WebGLTexture) {
    this.numNeighPerRow = numNeighPerRow;
    this.probOffsetTexture = offsets;
    this.probTexture = probabilities;
    this.probNeighIdTexture = neighIds;
  }

  async initializeNeighborsFromKNNGraph(numPoints: number, numNeighbors: number,
                                        distances: Float32Array,
                                        indices: Uint32Array,
                                        perplexity: number): Promise<void> {
    // Computing the shape of the knnGraphTexture
    const pointsPerRow =
        Math.max(1,
          Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors));
    const numRows = Math.ceil(numPoints / pointsPerRow);
    const dataShape =
        {numPoints, pixelsPerPoint : numNeighbors, numRows, pointsPerRow};

    // Indices and distances are packed in a 2-channel texture
    const textureValues =
        new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
    for (let i = 0; i < numPoints; ++i) {
      for (let n = 0; n < numNeighbors; ++n) {
        const id = (i * numNeighbors + n);
        textureValues[id * 2] = indices[id];
        textureValues[id * 2 + 1] = distances[id];
      }
    }

    // Texture generation
    const knnGraphTexture = gl_util.createAndConfigureTexture(
        this.gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);

    // Initializing the P matrix
    await this.initializeNeighborsFromKNNTexture(
      dataShape, knnGraphTexture, perplexity);

    // Deleting the knn texture
    this.gpgpu.gl.deleteTexture(knnGraphTexture);
  }

  // The Cantor pair function used to generate efficient map keys.
  // Given that Number.MAX_SAFE_INTEGER is
  // 9,007,199,254,740,991 this can handle pairs of
  // numbers up to just over 67,000,000 - more than we need.
  pairFunction(k1: number, k2: number): number {
    return (k1 + k2)*(k1 + k2 + 1)/2;
  }
  // Defines the neighborhood relationships between the points
  async initializeNeighborsFromKNNTexture(shape: RearrangedData,
                                          knnGraph: WebGLTexture,
                                          perplexity: number):
      Promise<void> {
    this.log('Asymmetric neighborhood initialization...');
    if (shape.numPoints !== this.numPoints) {
      throw new Error(`KNN size and number of points must agree` +
        `(${shape.numPoints},${this.numPoints})`);
    }
    this.log(`Shape info: pixelsPerPoint ${shape.pixelsPerPoint}`);
    this.log(`Create distribution params texture: 
        pointsPerRow: ${shape.pointsPerRow} x numRows: ${shape.numRows}`);
    // contains the beta and the sum of the gaussian weighted vector
    // used to compute the probability distributions
    const distParamTexture = gl_util.createAndConfigureTexture(
      this.gpgpu.gl, shape.pointsPerRow, shape.numRows, 2);

    this.log('Create zeroed distribution tensor');
    // contains the per-point probability vectors
    const gaussianDistributions =
      tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);

    // Computation of the per-point probability vectors
    this.gpgpu.enableAutomaticDebugValidation(true);
    this.log('Computing distribution params');

    this.computeDistributionParameters(distParamTexture, shape,
      perplexity, knnGraph);

    console.log(`Perplexity: ${perplexity}`);
    //this.logDistributionParamsTexture(
    //  distParamTexture,
    //  shape.pointsPerRow,
    //  shape.numRows);
    this.log('Computing Gaussian distn');

    this.computeGaussianDistributions(gaussianDistributions,
      distParamTexture, shape, knnGraph);
    this.log('Retrieve Gaussian distn');
    const gaussianDistributionsData = gaussianDistributions.dataSync();

    // Contains the per-point probability vectors
    const knnIndices =
      tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);
    // Computation of the per-point probability vectors
    this.log('Create copy indices program', knnIndices.shape);
    const copyIndicesProgram = knn_util.createCopyIndicesProgram(this.gpgpu);
    this.log('Execute copy indices program', knnIndices.shape);
    knn_util.executeCopyIndicesProgram(
      this.gpgpu, copyIndicesProgram, knnGraph, shape,
      this.backend.getTexture(knnIndices.dataId));

    const knnDistances =
      tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);
    const copyDistancesProgram =
      knn_util.createCopyDistancesProgram(this.gpgpu);
    knn_util.executeCopyDistancesProgram(
      this.gpgpu, copyDistancesProgram, knnGraph, shape,
      this.backend.getTexture(knnDistances.dataId));

    const knnIndicesData = await knnIndices.data();
    const knnDistanceData = await knnDistances.data();

    this.log(`Type of knnIndicesData ${knnIndicesData.constructor.name}`);
    this.log('knn Indices', knnIndicesData);
    this.log('knn Distances', knnDistanceData);

    // Neighborhood indices
    const asymNeighIds =
      new Uint32Array(shape.numPoints * shape.pixelsPerPoint);
    for (let i = 0; i < this.numPoints; ++i) {
      for (let d = 0; d < shape.pixelsPerPoint; ++d) {
        const linearId = i * shape.pixelsPerPoint + d;
        asymNeighIds[i * shape.pixelsPerPoint + d] = knnIndicesData[linearId];
      }
    }

    const neighLists = [];
    for (let i = 0; i < shape.numPoints; ++i) {
      const offset = i*shape.pixelsPerPoint;
      neighLists.push(
        [...asymNeighIds.slice(offset, offset+shape.pixelsPerPoint)]);
    }
    // Gaussian probs from neighbor to point.
    // for calculation of symmetric Pij
    const n2pProbMap = [];
    for (let i = 0; i < shape.numPoints; ++i) {
      for (let n = 0; n < shape.pixelsPerPoint; ++n) {
        const offset = i*shape.pixelsPerPoint + n;
        const neighId = asymNeighIds[offset];
        const iOff = neighLists[neighId].indexOf(i);
        if(iOff > -1) {
          const gaussOff = neighId * shape.pixelsPerPoint + iOff;
          n2pProbMap[this.pairFunction(neighId, i)] =
            gaussianDistributionsData[gaussOff];
        }
      }
    }
    this.log(`Neig - point prob`);
    // contains the total number of indirect neighbors per point
    const neighborCounter = new Uint32Array(this.numPoints);
    const neighborLinearOffset = new Uint32Array(this.numPoints);
    // count how many time each point appears as a neighbor
    for (let i = 0; i < shape.numPoints * shape.pixelsPerPoint; ++i) {
      ++neighborCounter[asymNeighIds[i]];
    }

    for (let i = 1; i < shape.numPoints; ++i) {
      neighborLinearOffset[i] = neighborLinearOffset[i - 1] +
        shape.pixelsPerPoint + neighborCounter[i - 1];
    }

    this.log('Counter', neighborCounter);
    this.log('Linear offset', neighborLinearOffset);

    // check
    let check = 0;
    let maxValue = 0;
    let maxId = 0;
    for (let i = 0; i < neighborCounter.length; ++i) {
      check += neighborCounter[i];
      if (neighborCounter[i] > maxValue) {
        maxValue = neighborCounter[i];
        maxId = i;
      }
    }
    this.log('Number of indirect links', check);
    this.log('Most central point', maxId);
    this.log('Number of indirect links for the central point', maxValue);

    // twice as many neighbors to make it symmetric (allows for duplicates)
    this.numNeighPerRow =
        Math.ceil(Math.sqrt(shape.numPoints * shape.pixelsPerPoint * 2));
    this.log('numNeighPerRow', this.numNeighPerRow);

    // Offsets
    {
      const offsets = new Float32Array(this.pointsPerRow * this.numRows * 3);
      let pointOffset = 0;
      for (let i = 0; i < this.numPoints; ++i) {
        // Direct + Indirect neighbors
        const totalNeighbors = shape.pixelsPerPoint + neighborCounter[i];
        offsets[i * 3 + 0] = (pointOffset) % (this.numNeighPerRow);
        offsets[i * 3 + 1] = Math.floor((pointOffset) / (this.numNeighPerRow));
        offsets[i * 3 + 2] = totalNeighbors;
        pointOffset += totalNeighbors;
      }
      this.log('Offsets', offsets);
      this.probOffsetTexture = gl_util.createAndConfigureTexture(
          this.gpgpu.gl, this.pointsPerRow, this.numRows, 3, offsets);
    }

    // Probabilities && Indices
    {
      const probabilities =
          new Float32Array(this.numNeighPerRow * this.numNeighPerRow);
      const neighIds =
          new Float32Array(this.numNeighPerRow * this.numNeighPerRow);
      const assignedNeighborCounter = new Uint32Array(this.numPoints);
      // Direct pass: probabilities are copied from the knn Graph
      for (let i = 0; i < this.numPoints; ++i) {
        // for all its neighbors
        for (let n = 0; n < shape.pixelsPerPoint; ++n) {
          const linearId = i * shape.pixelsPerPoint + n;
          // p_{i|n}
          // pointId is the neighbor id
          const pointId = knnIndicesData[linearId];
          // probability p_{i|n}
          const probability = gaussianDistributionsData[linearId];

          let npProb = n2pProbMap[this.pairFunction(pointId, i)];
          if (!npProb) {
            npProb = 0;
          }
          const pij = (npProb + probability)/2;
          const symMatrixDirectId = neighborLinearOffset[i] + n;
          const symMatrixIndirectId =
              neighborLinearOffset[pointId] +   // offset
              shape.pixelsPerPoint +            // num of direct neighbors
              assignedNeighborCounter[pointId]; // num of indirect

          probabilities[symMatrixDirectId] = pij;
          probabilities[symMatrixIndirectId] = pij;
          neighIds[symMatrixDirectId] = pointId;
          neighIds[symMatrixIndirectId] = i;
          ++assignedNeighborCounter[pointId];
        }
      }

      this.probTexture = gl_util.createAndConfigureTexture(
          this.gpgpu.gl, this.numNeighPerRow, this.numNeighPerRow, 1,
          probabilities);
      this.probNeighIdTexture = gl_util.createAndConfigureTexture(
          this.gpgpu.gl, this.numNeighPerRow, this.numNeighPerRow, 1, neighIds);
    }

    gaussianDistributions.dispose();
    knnIndices.dispose();

    this.log('...done!');
  }

  initializedNeighborhoods(): boolean {
    return this.probNeighIdTexture != null;
  }

  updateExaggeration() {
    if (this._exaggeration !== undefined) {
      this._exaggeration.dispose();
    }
    // Exaggeration is a number
    if (typeof this.rawExaggeration === 'number') {
      this._exaggeration = tf.scalar(this.rawExaggeration);
      return;
    }

    // Edge cases (before first element)
    if (this._iteration <= this.rawExaggeration[0].iteration) {
      this._exaggeration = tf.scalar(this.rawExaggeration[0].value);
      this.log(`Exaggeration val: ${this.rawExaggeration[0].value}`);
      return;
    }
    // Edge cases (after last element)
    if (this._iteration >=
        this.rawExaggeration[this.rawExaggeration.length - 1].iteration) {
      this._exaggeration = tf.scalar(
          this.rawExaggeration[this.rawExaggeration.length - 1].value);
      this.log(`Exaggeration val: 
        ${this.rawExaggeration[this.rawExaggeration.length - 1].value}`);
      return;
    }

    // Interpolation
    let i = 0;
    while (i < this.rawExaggeration.length &&
           this._iteration < this.rawExaggeration[i].iteration) {
      ++i;
    }
    const it0 = this.rawExaggeration[i].iteration;
    const it1 = this.rawExaggeration[i + 1].iteration;
    const v0 = this.rawExaggeration[i].value;
    const v1 = this.rawExaggeration[i + 1].value;
    const f = (it1 - this._iteration) / (it1 - it0);
    const v = v0 * f + v1 * (1 - f);
    this._exaggeration = tf.scalar(v);
    this._exaggerationNumber = v;
    this.log(`Exaggeration val: ${v}`);
  }
  //@ts-ignore
  private async logForces(attr:tf.Tensor, rep:tf.Tensor) {
    const attrData = await attr.data();
    const repData = await rep.data();
    console.log('Repulsive forces');
    for (let i = 0; i < repData.length; i++) {
      console.log(`R${i}:${repData[i]}`);
    }
    console.log('Attractive forces');
    for (let i = 0; i < attrData.length; i++) {
      console.log(`A${i}:${attrData[i]}`);
    }
  }

  // Texture tSNE
  async iterate(): Promise<void> {
    if (!this.initializedNeighborhoods()) {
      throw new Error('No neighborhoods defined. You may want to call\
                    initializeNeighbors or initializeNeighborsFromKNNGraph');
    }

    // check if the current splat texture is of the right size
    this.updateSplatTextureDiameter();
    this.updateExaggeration();

    let normQ: tf.Tensor;
    let gradientIter: tf.Tensor;
    [gradientIter, normQ] = tf.tidy(() => {
      // computing the gradient
      // 1) splat the points
      this.splatPoints();
      // 2) compute interpolation of the scalar fields
      const interpQ = tf.zeros([ this.numRows, this.pointsPerRow ]);
      const interpXY = tf.zeros([ this.numRows, this.pointsPerRow * 2 ]);
      // calculate S(y_l) - 1
      this.computeInterpolatedQ(interpQ);
      // calculate V(y_i)
      this.computeInterpolatedXY(interpXY);

      // 3) compute the normalization term - Z_hat <= Sigma(S(y_l) - 1)
      const normQ = interpQ.sum();
      // 4) compute the repulsive forces
      // break on normQ == 0?
      const repulsiveForces = interpXY.div(normQ.where(
        normQ.greater(tf.scalar(0)), tf.scalar(1)));
      // 5) compute the attractive forces
      const attractiveForces =
          tf.zeros([ this.numRows, this.pointsPerRow * 2 ]);
      this.computeAttractiveForces(attractiveForces);
      // 6) compute the gradient
      const gradientIter = attractiveForces.mul(this._exaggeration).
        sub(repulsiveForces).mul(tf.scalar(4));
      return [gradientIter, normQ];
    });
    this._normQ = (await normQ.data())[0];
    this.log(`Normalization: ${this._normQ}`);
    this.log(`Iteration ${this._iteration}`);
    normQ.dispose();
    // 7) update the embedding
    [this.gradient, this.gain, this.embedding] =
      await this.updateEmbedding(gradientIter, this._iteration);
    // 8) update the bounding box
    await this.computeBoundaries();

    // Increase the iteration counter
    ++this._iteration;
  }

  ////////////////////////////////
  ///// PRIVATE FUNCTIONS  ///////
  ////////////////////////////////
  /**
   * updateEmbedding uses an altered gradient iteration step to
   * update the embedding value.
   *
   * The gain tensor is optionally introduced to alleviate
   * points getting stuck in local minima something that
   * typically occurs at small numbers of points (< 500).
   * It acts a multiplier for the gradient.
   *
   * At a local minimum point gradients will
   * rapidly flip direction as they try to get out of it.
   * Therefore, if the gradient sign flips we add a little extra
   * force to the gradient (the gain is *1.25). If it still keep flipping,
   * we add a little more force to it.
   *
   * At some point the force behind this is big enough to push the
   * point out of the local minima and we can gradually
   * reduce the force with which we push it (gain is * 0.8).
   */
  private async updateEmbedding(gradIter: tf.Tensor, iterCount: number) {
    // The momentum used switches at a set iteration.
    const curMomentum = (iterCount < this._momSwitchIter) ?
      this._momentum : this._finalMomentum;
    // find points with flipped gradient
    console.log(`Iteration: ${this._iteration} 
      Exaggeration: ${this._exaggerationNumber}
      Momentum: ${curMomentum}`);
    return tf.tidy(() => {
      let gradient: tf.Tensor;
      let gain: tf.Tensor = null;
      if (this.applyGain) {
        //const gradFlip = gradIter.sign().notEqual(this.gradient.sign());
        // Calculate the new gain for the point based on gradient flip
        gain = this.gain.add(tf.scalar(0.2)).where(
          gradIter.sign().notEqual(this.gradient.sign()),
          this.gain.mul(tf.scalar(0.8)));
        //@ts-ignore
        const limitedGain = gain.where(
          gain.greaterEqual(tf.scalar(this._minimumGain)), this.minGain);
        // Now calculate the new gradient and update the embedding
        this.log(`Eta: ${this._eta}`);
        gradient =
          this.gradient.mul(curMomentum).sub(
            gradIter.mul(limitedGain.mul(tf.scalar(this._eta))));
      }
      else {
        gradient =
          this.gradient.mul(curMomentum).sub(
            gradIter.mul(tf.scalar(this._eta)));
      }

      const embedding = this.embedding.add(gradient);
      this.gradient.dispose();
      this.embedding.dispose();
      if (this.gain) {
        this.gain.dispose();
      }
      return [gradient, gain, embedding];
    });
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

  // Initialize the custom defined textures
  private initializeRepulsiveForceTextures() {
    // The splat texture holds the scalar fields used
    // for computing the gradient.
    this._splatTexture = gl_util.createAndConfigureInterpolatedTexture(
      this.gpgpu.gl, this.splatTextureDiameter,
      this.splatTextureDiameter, 4,
      null);

    // Computation of the kernel to splat.
    // First channel  - 1/(1+d^2)
    // Second channel - (1/(1+d^2))^2*d[x]
    // Third channel  - (1/(1+d^2))^2*d[y]
    // Fourth channel - 1 (extra for counting)
    this.kernelSupport = 2.5;

    this.log(`Kernel support: ${this.kernelSupport} 
      Kernel diameter: ${this.kernelTextureDiameter}`);
    const kernel = new Float32Array(this.kernelTextureDiameter *
                                    this.kernelTextureDiameter * 4);

    // Computation of the tSNE splat kernel
    const kernelRadius = Math.floor(this.kernelTextureDiameter / 2);
    for (let j = 0; j < this.kernelTextureDiameter; ++j) {
      for (let i = 0; i < this.kernelTextureDiameter; ++i) {
        const x = ((i - kernelRadius) / kernelRadius) * this.kernelSupport;
        const y = ((j - kernelRadius) / kernelRadius) * this.kernelSupport;
        const euclSquared = x * x + y * y;
        const tStudent = 1. / (1. + euclSquared);
        const id = (j * this.kernelTextureDiameter + i) * 4;
        kernel[id + 0] = tStudent;
        kernel[id + 1] = tStudent * tStudent * x;
        kernel[id + 2] = tStudent * tStudent * y;
        kernel[id + 3] = 1;
      }
    }
    // Texture creation
    this.kernelTexture = gl_util.createAndConfigureInterpolatedTexture(
        this.gpgpu.gl, this.kernelTextureDiameter, this.kernelTextureDiameter,
        4, kernel);
  }

  // Initialize the WebGL programs used by the algorithm
  private initilizeCustomWebGLPrograms() {
    this.log('\tCreating custom programs...');
    this.embeddingInitializationProgram =
        tsne_util.createEmbeddingInitializationProgram(this.gpgpu);

    this.embeddingSplatterProgram =
        tsne_util.createEmbeddingSplatterProgramOrig(this.gpgpu);

    const splatVertexId = new Float32Array(this.numPoints * 6);

    {
      let id = 0;
      for (let i = 0; i < this.numPoints; ++i) {
        // Mapping from id => vertex coord
        // is in the createEmbeddingSplatterProgram
        // vertex shader source.
        // 0 -> -1,-1
        // 1 ->  1,-1
        // 2 ->  1, 1
        // 3 -> -1, 1
        // graphically:
        // 3 --- 2
        // |  /  |
        // 0 --- 1
        id = i * 6;
        splatVertexId[id + 0] = 0 + i * 4;
        splatVertexId[id + 1] = 1 + i * 4;
        splatVertexId[id + 2] = 2 + i * 4;
        splatVertexId[id + 3] = 0 + i * 4;
        splatVertexId[id + 4] = 2 + i * 4;
        splatVertexId[id + 5] = 3 + i * 4;
      }
    }

    this.splatVertexIdBuffer = tf.webgl.webgl_util.createStaticVertexBuffer(
        this.gpgpu.gl, splatVertexId);

    // Compute the interpolated value of Q (first channel)
    this.qInterpolatorProgram =
        tsne_util.createQInterpolatorProgram(this.gpgpu);
    this.xyInterpolatorProgram =
        tsne_util.createXYInterpolatorProgram(this.gpgpu);

    // Computation of the attracive forces
    this.attractiveForcesProgram =
        tsne_util.createAttractiveForcesComputationProgram(this.gpgpu);

    // Computation of the joint probability distribution
    this.distributionParameterssComputationProgram =
        tsne_util.createDistributionParametersComputationProgram(this.gpgpu);
    this.gaussiaDistributionsFromDistancesProgram =
        tsne_util.createGaussiaDistributionsFromDistancesProgram(this.gpgpu);
  }

  // Compute the boundaries of the embedding for defining the splat area
  // and the visualization boundaries
  // Use squared limits - i.e. find the largest range and
  // use that on all dimensions
  private async computeBoundaries(): Promise<void> {
    const [min, max] = tf.tidy(() => {
      // 2d tensor with some extra points
      const embedding2D =
          this.embedding.reshape([ this.numRows * this.pointsPerRow, 2 ])
              .slice([ 0, 0 ], [ this.numPoints, 2 ]);

      const minn = embedding2D.min(0);
      const maxx = embedding2D.max(0);
      return [ minn, maxx ];
    });

    const minData = await min.data();
    const maxData = await max.data();

    const percentageOffset = 0.05;

    const offsetX = (maxData[0] - minData[0]) * percentageOffset;
    this._minX = minData[0] - offsetX;
    this._maxX = maxData[0] + offsetX;

    const offsetY = (maxData[1] - minData[1]) * percentageOffset;
    this._minY = minData[1] - offsetY;
    this._maxY = maxData[1] + offsetY;

    min.dispose();
    max.dispose();
  }

  private calculateTexSize(range: number) {
    const minSplatDiameter = 5;
    const maxSplatDiameter = 5000; //5000
    return Math.min(maxSplatDiameter,
      Math.max(Math.floor(range/this._spacePerPixel), minSplatDiameter));
  }

  private updateSplatTextureDiameter() {
    const rangeX = this._maxX - this._minX;
    const rangeY = this._maxY - this._minY;
    const maxRange = Math.max(rangeX, rangeY);
    this.log(`Embedding range: ${maxRange}`);

    const textureDiam = this.calculateTexSize(maxRange);

    // differing width/height doesnt work with the shaders
    if (Math.abs((textureDiam - this.splatTextureDiameter)/
      this.splatTextureDiameter) >= 0.2) {
      this.log(`Change splat size: ${this._iteration}`);
      this.gpgpu.gl.deleteTexture(this._splatTexture);
      this.splatTextureDiameter = Math.ceil(textureDiam);
      this.log(`Splat diameter: ${this.splatTextureDiameter}`);
      this._splatTexture = gl_util.createAndConfigureInterpolatedTexture(
          this.gpgpu.gl,
          this.splatTextureDiameter,
          this.splatTextureDiameter,
          4, null);
    }
    else {
      this.log(`Splat diameter: ${this.splatTextureDiameter}`);
    }
  }

  // @ts-ignore
  private initializeEmbeddingPositions(embedding: tf.Tensor,
                                       random: tf.Tensor) {
    tsne_util.executeEmbeddingInitializationProgram(
        this.gpgpu, this.embeddingInitializationProgram,
        this.backend.getTexture(random.dataId), this.numPoints,
        this.pointsPerRow, this.numRows,
        this.backend.getTexture(embedding.dataId));
  }

  private splatPoints() {
    tsne_util.executeEmbeddingSplatterProgram(
        this.gpgpu, this.embeddingSplatterProgram, this._splatTexture,
        this.backend.getTexture(this.embedding.dataId), this.kernelTexture,
        this.splatTextureDiameter, this.splatTextureDiameter,
        this.numPoints, this._minX, this._minY,
        this._maxX, this._maxY, this.kernelSupport,
        this.pointsPerRow,
        this.numRows, this.splatVertexIdBuffer);
  }

  private computeInterpolatedQ(interpolatedQ: tf.Tensor) {
    tsne_util.executeQInterpolatorProgram(
        this.gpgpu, this.qInterpolatorProgram, this._splatTexture,
        this.backend.getTexture(this.embedding.dataId), this.numPoints,
        this._minX, this._minY, this._maxX, this._maxY, this.pointsPerRow,
        this.numRows, this.backend.getTexture(interpolatedQ.dataId));
  }

  private computeInterpolatedXY(interpolatedXY: tf.Tensor) {
    tsne_util.executeXYInterpolatorProgram(
        this.gpgpu, this.xyInterpolatorProgram, this._splatTexture,
        this.backend.getTexture(this.embedding.dataId),
        this.backend.getTexture(interpolatedXY.dataId), this.numPoints,
        this._minX, this._minY, this._maxX, this._maxY, this.pointsPerRow,
        this.numRows, this._eta);
  }

  private computeAttractiveForces(attractiveForces: tf.Tensor) {
    tsne_util.executeAttractiveForcesComputationProgram(
        this.gpgpu, this.attractiveForcesProgram,
        this.backend.getTexture(this.embedding.dataId), this.probOffsetTexture,
        this.probNeighIdTexture, this.probTexture, this.numPoints,
        this.numNeighPerRow, this.pointsPerRow, this.numRows, this._eta,
        this.backend.getTexture(attractiveForces.dataId));
  }

  private computeDistributionParameters(distributionParameters: WebGLTexture,
                                        shape: RearrangedData,
                                        perplexity: number,
                                        knnGraph: WebGLTexture) {
    tsne_util.executeDistributionParametersComputationProgram(
      this.gpgpu, this.distributionParameterssComputationProgram, knnGraph,
      shape.numPoints, shape.pixelsPerPoint, shape.pointsPerRow,
      shape.numRows, perplexity, distributionParameters);
  }

  private computeGaussianDistributions(distributions: tf.Tensor,
                                       distributionParameters: WebGLTexture,
                                       shape: RearrangedData,
                                       knnGraph: WebGLTexture) {
    const distTexture = this.backend.getTexture(distributions.dataId);
    tsne_util.executeGaussiaDistributionsFromDistancesProgram(
      this.gpgpu, this.gaussiaDistributionsFromDistancesProgram, knnGraph,
      distributionParameters, shape.numPoints, shape.pixelsPerPoint,
      shape.pointsPerRow, shape.numRows,
      distTexture);
  }
}
