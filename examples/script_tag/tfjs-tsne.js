// @tensorflow/tfjs-tsne Copyright 2018 Google
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core'], factory) :
  (factory((global.tfjs_tsne = {}),global.tf));
}(this, (function (exports,tf) { 'use strict';

  function createVertexProgram(gl, vertexShaderSource, fragmentShaderSource) {
      var vertexShader = tf.webgl.webgl_util.createVertexShader(gl, vertexShaderSource);
      var fragmentShader = tf.webgl.webgl_util.createFragmentShader(gl, fragmentShaderSource);
      var program = tf.webgl.webgl_util.createProgram(gl);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, vertexShader); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, fragmentShader); });
      tf.webgl.webgl_util.linkProgram(gl, program);
      tf.webgl.webgl_util.validateProgram(gl, program);
      return program;
  }
  function createAndConfigureInterpolatedTexture(gl, width, height, numChannels, pixels) {
      tf.webgl.webgl_util.validateTextureSize(gl, width, height);
      var texture = tf.webgl.webgl_util.createTexture(gl);
      var tex2d = gl.TEXTURE_2D;
      var internalFormat = getTextureInternalFormat(gl, numChannels);
      var format = getTextureFormat(gl, numChannels);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.LINEAR); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.LINEAR); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, getTextureType(gl), pixels); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
      return texture;
  }
  function createAndConfigureTexture(gl, width, height, numChannels, pixels) {
      tf.webgl.webgl_util.validateTextureSize(gl, width, height);
      var texture = tf.webgl.webgl_util.createTexture(gl);
      var tex2d = gl.TEXTURE_2D;
      var internalFormat = getTextureInternalFormat(gl, numChannels);
      var format = getTextureFormat(gl, numChannels);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, getTextureType(gl), pixels); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
      return texture;
  }
  function createAndConfigureUByteTexture(gl, width, height, numChannels, pixels) {
      tf.webgl.webgl_util.validateTextureSize(gl, width, height);
      var texture = tf.webgl.webgl_util.createTexture(gl);
      var tex2d = gl.TEXTURE_2D;
      var internalFormat = getTextureInternalUByteFormat(gl, numChannels);
      var format = getTextureFormat(gl, numChannels);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, getTextureTypeUByte(gl), pixels); });
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
      return texture;
  }
  function getTextureInternalFormat(gl, numChannels) {
      if (numChannels === 4) {
          return gl.RGBA32F;
      }
      else if (numChannels === 3) {
          return gl.RGB32F;
      }
      else if (numChannels === 2) {
          return gl.RG32F;
      }
      return gl.R32F;
  }
  function getTextureInternalUByteFormat(gl, numChannels) {
      if (numChannels === 4) {
          return gl.RGBA8;
      }
      else if (numChannels === 3) {
          return gl.RGB8;
      }
      else if (numChannels === 2) {
          return gl.RG8;
      }
      return gl.R8;
  }
  function getTextureFormat(gl, numChannels) {
      if (numChannels === 4) {
          return gl.RGBA;
      }
      else if (numChannels === 3) {
          return gl.RGB;
      }
      else if (numChannels === 2) {
          return gl.RG;
      }
      return gl.RED;
  }
  function getTextureType(gl) {
      return gl.FLOAT;
  }
  function getTextureTypeUByte(gl) {
      return gl.UNSIGNED_BYTE;
  }

  var gl_util = /*#__PURE__*/Object.freeze({
    createVertexProgram: createVertexProgram,
    createAndConfigureInterpolatedTexture: createAndConfigureInterpolatedTexture,
    createAndConfigureTexture: createAndConfigureTexture,
    createAndConfigureUByteTexture: createAndConfigureUByteTexture
  });

  function generateDistanceComputationSource(format) {
      var source = "\n    #define DATA_NUM_PACKED_DIMENSIONS " + format.pixelsPerPoint + ".\n    #define DATA_POINTS_PER_ROW " + format.pointsPerRow + ".\n    #define DATA_NUM_ROWS " + format.numRows + ".\n    #define TEXTURE_WIDTH " + format.pointsPerRow * format.pixelsPerPoint + ".\n\n    //returns the texture coordinate for point/dimension\n    vec2 dataTexCoordinates(int id, int dimension) {\n      float id_f = float(id);\n      float row = (floor(id_f/DATA_POINTS_PER_ROW)+0.5) / DATA_NUM_ROWS;\n      float col = ((mod(id_f,DATA_POINTS_PER_ROW)*(DATA_NUM_PACKED_DIMENSIONS)\n                  + float(dimension)) + 0.5) / (TEXTURE_WIDTH);\n      return vec2(col,row);\n    }\n\n    //compute the euclidean squared distances between two points i and j\n    float pointDistanceSquared(int i, int j) {\n      vec4 result = vec4(0,0,0,0);\n      int num_iter = int(DATA_NUM_PACKED_DIMENSIONS);\n      for(int d = 0; d < num_iter; ++d) {\n        vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n        vec4 vj = texture(data_tex,dataTexCoordinates(j,d));\n        result += (vi-vj)*(vi-vj);\n      }\n      return (result.r+result.g+result.b+result.a);\n    }\n\n    //compute the euclidean squared distances between two points i and j\n    vec4 pointDistanceSquaredBatch(int i, int j0, int j1, int j2, int j3) {\n      vec4 result = vec4(0,0,0,0);\n      int num_iter = int(DATA_NUM_PACKED_DIMENSIONS);\n      for(int d = 0; d < num_iter; ++d) {\n        vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n        vec4 vj0 = texture(data_tex,dataTexCoordinates(j0,d));\n        vec4 vj1 = texture(data_tex,dataTexCoordinates(j1,d));\n        vec4 vj2 = texture(data_tex,dataTexCoordinates(j2,d));\n        vec4 vj3 = texture(data_tex,dataTexCoordinates(j3,d));\n        vj0 = (vi-vj0); vj0 *= vj0;\n        vj1 = (vi-vj1); vj1 *= vj1;\n        vj2 = (vi-vj2); vj2 *= vj2;\n        vj3 = (vi-vj3); vj3 *= vj3;\n        result.r += (vj0.r+vj0.g+vj0.b+vj0.a);\n        result.g += (vj1.r+vj1.g+vj1.b+vj1.a);\n        result.b += (vj2.r+vj2.g+vj2.b+vj2.a);\n        result.a += (vj3.r+vj3.g+vj3.b+vj3.a);\n      }\n      return result;\n    }\n    ";
      return source;
  }
  function generateMNISTDistanceComputationSource() {
      var source = "\n  #define POINTS_PER_ROW 250.\n  #define NUM_ROWS 240.\n  #define TEXTURE_WIDTH 3500.\n  #define TEXTURE_HEIGHT 3360.\n  #define DIGIT_WIDTH 14.\n  #define NUM_PACKED_DIMENSIONS 196\n\n  //returns the texture coordinate for point/dimension\n  vec2 dataTexCoordinates(int id, int dimension) {\n    float id_f = float(id);\n    float dimension_f = float(dimension);\n    float col = ((mod(id_f,POINTS_PER_ROW)*DIGIT_WIDTH));\n    float row = (floor(id_f/POINTS_PER_ROW)*DIGIT_WIDTH);\n\n    return (vec2(col,row)+\n            vec2(mod(dimension_f,DIGIT_WIDTH),floor(dimension_f/DIGIT_WIDTH))+\n            vec2(0.5,0.5)\n            )/\n            vec2(TEXTURE_WIDTH,TEXTURE_HEIGHT);\n  }\n\n  //compute the euclidean squared distances between two points i and j\n  float pointDistanceSquared(int i, int j) {\n    vec4 result = vec4(0,0,0,0);\n    for(int d = 0; d < NUM_PACKED_DIMENSIONS; d+=1) {\n      vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n      vec4 vj = texture(data_tex,dataTexCoordinates(j,d));\n      result += (vi-vj)*(vi-vj);\n    }\n    return (result.r+result.g+result.b+result.a);\n  }\n\n  //compute the euclidean squared distances between two points i and j\n  vec4 pointDistanceSquaredBatch(int i, int j0, int j1, int j2, int j3) {\n    vec4 result = vec4(0,0,0,0);\n    for(int d = 0; d < NUM_PACKED_DIMENSIONS; d+=1) {\n      vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n      vec4 vj0 = texture(data_tex,dataTexCoordinates(j0,d));\n      vec4 vj1 = texture(data_tex,dataTexCoordinates(j1,d));\n      vec4 vj2 = texture(data_tex,dataTexCoordinates(j2,d));\n      vec4 vj3 = texture(data_tex,dataTexCoordinates(j3,d));\n      vj0 = (vi-vj0); vj0 *= vj0;\n      vj1 = (vi-vj1); vj1 *= vj1;\n      vj2 = (vi-vj2); vj2 *= vj2;\n      vj3 = (vi-vj3); vj3 *= vj3;\n      result.r += (vj0.r+vj0.g+vj0.b+vj0.a);\n      result.g += (vj1.r+vj1.g+vj1.b+vj1.a);\n      result.b += (vj2.r+vj2.g+vj2.b+vj2.a);\n      result.a += (vj3.r+vj3.g+vj3.b+vj3.a);\n    }\n    return result;\n  }\n  ";
      return source;
  }
  function generateKNNClusterTexture(numPoints, numClusters, numNeighbors) {
      var pointsPerRow = Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors);
      var numRows = Math.ceil(numPoints / pointsPerRow);
      var dataShape = { numPoints: numPoints, pixelsPerPoint: numNeighbors, numRows: numRows, pointsPerRow: pointsPerRow };
      var pointsPerCluster = Math.ceil(numPoints / numClusters);
      var textureValues = new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
      for (var i = 0; i < numPoints; ++i) {
          var clusterId = Math.floor(i / pointsPerCluster);
          for (var n = 0; n < numNeighbors; ++n) {
              var id = (i * numNeighbors + n) * 2;
              textureValues[id] = Math.floor(Math.random() * pointsPerCluster) +
                  clusterId * pointsPerCluster;
              textureValues[id + 1] = Math.random();
          }
      }
      var backend = tf.ENV.findBackend('webgl');
      if (backend === null) {
          throw Error('WebGL backend is not available');
      }
      var gpgpu = backend.getGPGPUContext();
      var knnGraph = createAndConfigureTexture(gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);
      return { knnGraph: knnGraph, dataShape: dataShape };
  }
  function generateKNNLineTexture(numPoints, numNeighbors) {
      var pointsPerRow = Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors);
      var numRows = Math.ceil(numPoints / pointsPerRow);
      var dataShape = { numPoints: numPoints, pixelsPerPoint: numNeighbors, numRows: numRows, pointsPerRow: pointsPerRow };
      var textureValues = new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
      for (var i = 0; i < numPoints; ++i) {
          for (var n = 0; n < numNeighbors; ++n) {
              var id = (i * numNeighbors + n) * 2;
              textureValues[id] =
                  Math.floor(i + n - (numNeighbors / 2) + numPoints) % numPoints;
              textureValues[id + 1] = 1;
          }
      }
      var backend = tf.ENV.findBackend('webgl');
      if (backend === null) {
          throw Error('WebGL backend is not available');
      }
      var gpgpu = backend.getGPGPUContext();
      var knnGraph = createAndConfigureTexture(gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);
      return { knnGraph: knnGraph, dataShape: dataShape };
  }
  function generateKNNClusterData(numPoints, numClusters, numNeighbors) {
      var pointsPerCluster = Math.ceil(numPoints / numClusters);
      var distances = new Float32Array(numPoints * numNeighbors);
      var indices = new Uint32Array(numPoints * numNeighbors);
      for (var i = 0; i < numPoints; ++i) {
          var clusterId = Math.floor(i / pointsPerCluster);
          for (var n = 0; n < numNeighbors; ++n) {
              var id = (i * numNeighbors + n);
              distances[id] = Math.random();
              indices[id] = Math.floor(Math.random() * pointsPerCluster) +
                  clusterId * pointsPerCluster;
          }
      }
      return { distances: distances, indices: indices };
  }
  function generateKNNLineData(numPoints, numNeighbors) {
      var distances = new Float32Array(numPoints * numNeighbors);
      var indices = new Uint32Array(numPoints * numNeighbors);
      for (var i = 0; i < numPoints; ++i) {
          for (var n = 0; n < numNeighbors; ++n) {
              var id = (i * numNeighbors + n);
              distances[id] = 1;
              indices[id] =
                  Math.floor(i + n - (numNeighbors / 2) + numPoints) % numPoints;
          }
      }
      return { distances: distances, indices: indices };
  }

  var dataset_util = /*#__PURE__*/Object.freeze({
    generateDistanceComputationSource: generateDistanceComputationSource,
    generateMNISTDistanceComputationSource: generateMNISTDistanceComputationSource,
    generateKNNClusterTexture: generateKNNClusterTexture,
    generateKNNLineTexture: generateKNNLineTexture,
    generateKNNClusterData: generateKNNClusterData,
    generateKNNLineData: generateKNNLineData
  });

  function createSplatTextureDrawerProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D splat_tex;\n    uniform sampler2D color_scale_tex;\n    uniform sampler2D drawn_embedding_tex;\n\n    uniform float width;\n    uniform float height;\n    uniform float normalization;\n\n    void main() {\n      float value = 0.;\n      vec2 texture_pos = vec2(-1,-1);\n\n      if(gl_FragCoord.x < width/2. && gl_FragCoord.y >= height/2.) {\n        vec2 texture_pos = (gl_FragCoord.xy-vec2(0.,height/2.))\n                              / vec2(width/2.,height/2.);\n        vec4 values = texture2D(drawn_embedding_tex,texture_pos);\n        gl_FragColor = values;\n        return;\n\n      }else if(gl_FragCoord.x >= width/2. && gl_FragCoord.y >= height/2.) {\n        vec2 texture_pos = (gl_FragCoord.xy-vec2(width/2.,height/2.))\n                              / vec2(width/2.,height/2.);\n        vec4 values = texture2D(splat_tex,texture_pos)/normalization;\n        value = values.x;\n\n      }else if(gl_FragCoord.x < width/2. && gl_FragCoord.y < height/2.) {\n        vec2 texture_pos = gl_FragCoord.xy / vec2(width/2.,height/2.);\n        vec4 values = texture2D(splat_tex,texture_pos)/normalization*2.;\n        value = values.y;\n\n      }else if(gl_FragCoord.x >= width/2. && gl_FragCoord.y < height/2.) {\n        vec2 texture_pos = (gl_FragCoord.xy-vec2(width/2.,0))\n                              / vec2(width/2.,height/2.);\n        vec4 values = texture2D(splat_tex,texture_pos)/normalization*2.;\n        value = values.z;\n      }\n\n\n      vec2 color_scale_pos  = vec2(-1.*value+0.5,0.5);\n      vec4 color = texture2D(color_scale_tex,color_scale_pos)/255.;\n\n      gl_FragColor = vec4(color.xyz,1);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeSplatTextureDrawerProgram(gpgpu, program, splatTex, colorScaleTex, drawnEmbeddingTex, normalization, textureDiameter, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, textureDiameter, textureDiameter);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'splat_tex');
      gpgpu.setInputMatrixTexture(splatTex, splatLocation, 0);
      var colorScaleLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'color_scale_tex');
      gpgpu.setInputMatrixTexture(colorScaleTex, colorScaleLocation, 1);
      var drawnEmbeddingLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'drawn_embedding_tex');
      gpgpu.setInputMatrixTexture(drawnEmbeddingTex, drawnEmbeddingLoc, 2);
      var widthLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'width');
      gl.uniform1f(widthLoc, textureDiameter);
      var heightLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'height');
      gl.uniform1f(heightLoc, textureDiameter);
      var normLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'normalization');
      gl.uniform1f(normLoc, normalization);
      gpgpu.executeProgram();
  }
  function createSimpleEmbeddingDrawerProgram(gpgpu) {
      var vertexShaderSource = "\n    precision highp float;\n    attribute float vertex_id;\n\n    uniform sampler2D embedding_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float pnts_per_row;\n    uniform float num_rows;\n\n    void main() {\n      int row    = int(vertex_id/pnts_per_row);\n      int column = int(mod(vertex_id,pnts_per_row));\n      float width = (pnts_per_row*2.0);\n      float row_tex = (float(row)+0.5)/num_rows;\n      vec2 tex_coords_x = vec2((float(column)*2.+0.5)/width, row_tex);\n      vec2 tex_coords_y = vec2((float(column)*2.+1.+0.5)/width, row_tex);\n\n      float x_pnt = texture2D(embedding_tex,tex_coords_x).r;\n      float y_pnt = texture2D(embedding_tex,tex_coords_y).r;\n      vec2 vertex_coords = vec2(x_pnt,y_pnt);\n\n      vertex_coords = (vertex_coords-minV)/(maxV-minV); //  0:1 space\n      vertex_coords = vertex_coords*2.0 - 1.0;          // -1:1 space\n\n      gl_Position = vec4(vertex_coords,0,1);\n      gl_PointSize = 4.;\n    }\n  ";
      var fragmentShaderSource = "\n    precision highp float;\n    uniform float alpha;\n\n    void main() {\n      float r = 0.0, delta = 0.0;\n      vec2 cxy = 2.0 * gl_PointCoord - 1.0;\n      r = dot(cxy, cxy);\n      if (r > 1.0) {\n          discard;\n      }\n      gl_FragColor = vec4(0,0.6,1,alpha);\n    }\n  ";
      return createVertexProgram(gpgpu.gl, vertexShaderSource, fragmentShaderSource);
  }
  function executeSimpleEmbeddingDrawerProgram(gpgpu, program, embeddingTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, pointIdBuffer, alpha, targetTexDiameter, targetTex) {
      var gl = gpgpu.gl;
      var oldProgram = gpgpu.program;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, targetTexDiameter, targetTexDiameter);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gl.clearColor(1, 1, 1, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gpgpu.setProgram(program);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer); });
      tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', pointIdBuffer, 1, 0, 0);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'pnts_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var alphaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'alpha');
      gl.uniform1f(alphaLoc, alpha);
      if (maxX - minX > maxY - minY) {
          maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
          minY = (maxY + minY) / 2 - (maxX - minX) / 2;
      }
      else {
          maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
          minX = (maxX + minX) / 2 - (maxY - minY) / 2;
      }
      var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
      gl.uniform2f(minLoc, minX, minY);
      var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
      gl.uniform2f(maxLoc, maxX, maxY);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.POINTS, 0, numPoints); });
      gl.disable(gl.BLEND);
      if (oldProgram != null) {
          gpgpu.setProgram(oldProgram);
          tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
      }
  }
  function createColoredEmbeddingDrawerProgram(gpgpu) {
      var vertexShaderSource = "\n    precision highp float;\n    attribute float vertex_id;\n\n    uniform sampler2D embedding_tex;\n    uniform sampler2D color_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float pnts_per_row;\n    uniform float num_rows;\n\n    varying vec4 color;\n\n    void main() {\n      int row    = int(vertex_id/pnts_per_row);\n      int column = int(mod(vertex_id,pnts_per_row));\n      float width = (pnts_per_row*2.0);\n      float row_tex = (float(row)+0.5)/num_rows;\n      vec2 tex_coords_x = vec2((float(column)*2.+0.5)/width, row_tex);\n      vec2 tex_coords_y = vec2((float(column)*2.+1.+0.5)/width, row_tex);\n\n      vec2 color_coords = vec2((float(column)+0.5)/pnts_per_row, row_tex);\n      color = texture2D(color_tex,color_coords);\n\n      float x_pnt = texture2D(embedding_tex,tex_coords_x).r;\n      float y_pnt = texture2D(embedding_tex,tex_coords_y).r;\n      vec2 vertex_coords = vec2(x_pnt,y_pnt);\n\n      vertex_coords = (vertex_coords-minV)/(maxV-minV); //  0:1 space\n      vertex_coords = vertex_coords*2.0 - 1.0;          // -1:1 space\n\n      gl_Position = vec4(vertex_coords,0,1);\n      gl_PointSize = 4.;\n    }\n  ";
      var fragmentShaderSource = "\n    precision highp float;\n    uniform float alpha;\n    varying vec4 color;\n\n    void main() {\n      //vec4 color = vec4(0.1,0.4,0.9,alpha);\n      float r = 0.0, delta = 0.0;\n      vec2 cxy = 2.0 * gl_PointCoord - 1.0;\n      r = dot(cxy, cxy);\n      if (r > 1.0) {\n          discard;\n      }\n      gl_FragColor = color;\n      gl_FragColor.a = alpha;\n    }\n  ";
      return createVertexProgram(gpgpu.gl, vertexShaderSource, fragmentShaderSource);
  }
  function executeColoredEmbeddingDrawerProgram(gpgpu, program, embeddingTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, pointIdBuffer, alpha, targetTexDiameter, colorsTex, targetTex) {
      var gl = gpgpu.gl;
      var oldProgram = gpgpu.program;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, targetTexDiameter, targetTexDiameter);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gl.clearColor(1, 1, 1, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gpgpu.setProgram(program);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer); });
      tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', pointIdBuffer, 1, 0, 0);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
      var colorLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'color_tex');
      gpgpu.setInputMatrixTexture(colorsTex, colorLocation, 1);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'pnts_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var alphaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'alpha');
      gl.uniform1f(alphaLoc, alpha);
      if (maxX - minX > maxY - minY) {
          maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
          minY = (maxY + minY) / 2 - (maxX - minX) / 2;
      }
      else {
          maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
          minX = (maxX + minX) / 2 - (maxY - minY) / 2;
      }
      var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
      gl.uniform2f(minLoc, minX, minY);
      var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
      gl.uniform2f(maxLoc, maxX, maxY);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.POINTS, 0, numPoints); });
      gl.disable(gl.BLEND);
      if (oldProgram != null) {
          gpgpu.setProgram(oldProgram);
          tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
      }
  }
  function createTexturedPointsDrawerProgram(gpgpu) {
      var vertexShaderSource = "#version 300 es\n    precision highp float;\n    in float vertex_id;\n\n    uniform sampler2D embedding_tex;\n    uniform sampler2D color_tex;\n\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float pnts_per_row;\n    uniform float num_rows;\n\n    out float image_id;\n    out vec4 label_color;\n\n\n    void main() {\n      int row    = int(vertex_id/pnts_per_row);\n      int column = int(mod(vertex_id,pnts_per_row));\n\n      float width = (pnts_per_row*2.0);\n      float row_tex = (float(row)+0.5)/num_rows;\n      vec2 tex_coords_x = vec2((float(column)*2.+0.5)/width, row_tex);\n      vec2 tex_coords_y = vec2((float(column)*2.+1.+0.5)/width, row_tex);\n\n      vec2 color_coords = vec2((float(column)+0.5)/pnts_per_row, row_tex);\n      label_color = texture(color_tex,color_coords);\n\n      float x_pnt = texture(embedding_tex,tex_coords_x).r;\n      float y_pnt = texture(embedding_tex,tex_coords_y).r;\n      vec2 vertex_coords = vec2(x_pnt,y_pnt);\n\n      vertex_coords = (vertex_coords-minV)/(maxV-minV); //  0:1 space\n      vertex_coords = vertex_coords*2.0 - 1.0;          // -1:1 space\n\n      gl_Position = vec4(vertex_coords,0,1);\n      gl_PointSize = 14.;\n      image_id = vertex_id;\n    }\n  ";
      var fragmentShaderSource = "#version 300 es\n    precision highp float;\n    uniform float alpha;\n    uniform float pnts_per_row;\n    uniform float num_rows;\n    uniform float point_texture_diameter;\n\n    in float image_id;\n    in vec4 label_color;\n\n    out vec4 fragment_color;\n\n    uniform sampler2D point_tex;\n\n    //Random function developed by Inigo Quilez\n    //https://www.shadertoy.com/view/llGSzw\n    float hash1( uint n ) {\n        // integer hash copied from Hugo Elias\n    \t  n = (n << 13U) ^ n;\n        n = n * (n * n * 15731U + 789221U) + 1376312589U;\n        return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);\n    }\n\n    void main() {\n      vec2 cxy = gl_PointCoord*point_texture_diameter;\n\n      float random = hash1(uint(image_id));\n\n      int row    = int(image_id/250.);\n      int col = int(mod(image_id,250.));\n\n\n      float col_tex = (float(col)*point_texture_diameter+0.5+cxy.x)/3500.;\n      float row_tex = (float(row)*point_texture_diameter+0.5+cxy.y)/3360.;\n\n      vec2 tex_coords = vec2(col_tex, row_tex);\n      vec4 texture_value = texture(point_tex,tex_coords);\n      float average_value = (texture_value.r,texture_value.g,\n                              texture_value.b,texture_value.a)/4.;\n\n\n      fragment_color = label_color;\n      fragment_color.a = average_value*1.5;\n\n      float fade_in = 0.05;\n      if(random - alpha < fade_in) {\n        fragment_color.a *= 1. - (random - alpha)/fade_in;\n      }else if(random > alpha) {\n        fragment_color.a = 0.;\n      }\n    }\n  ";
      return createVertexProgram(gpgpu.gl, vertexShaderSource, fragmentShaderSource);
  }
  function executeTexturedPointsDrawerProgram(gpgpu, program, embeddingTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, pointIdBuffer, alpha, targetTexDiameter, pointsTex, pointTextureDiameter, colorsTex, targetTex) {
      var gl = gpgpu.gl;
      var oldProgram = gpgpu.program;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, targetTexDiameter, targetTexDiameter);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gl.clearColor(1, 1, 1, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gpgpu.setProgram(program);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer); });
      tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', pointIdBuffer, 1, 0, 0);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
      var pointsTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'point_tex');
      gpgpu.setInputMatrixTexture(pointsTex, pointsTexLoc, 1);
      var colorsTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'color_tex');
      gpgpu.setInputMatrixTexture(colorsTex, colorsTexLoc, 2);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'pnts_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var pointTextureDiameterLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'point_texture_diameter');
      gl.uniform1f(pointTextureDiameterLoc, pointTextureDiameter);
      var alphaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'alpha');
      gl.uniform1f(alphaLoc, alpha);
      if (maxX - minX > maxY - minY) {
          maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
          minY = (maxY + minY) / 2 - (maxX - minX) / 2;
      }
      else {
          maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
          minX = (maxX + minX) / 2 - (maxY - minY) / 2;
      }
      var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
      gl.uniform2f(minLoc, minX, minY);
      var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
      gl.uniform2f(maxLoc, maxX, maxY);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.POINTS, 0, numPoints); });
      gl.disable(gl.BLEND);
      if (oldProgram != null) {
          gpgpu.setProgram(oldProgram);
          tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
      }
  }

  var EmbeddingDrawer = (function () {
      function EmbeddingDrawer(numPoints) {
          this.numPoints = numPoints;
          var backend = tf.ENV.findBackend('webgl');
          this.gpgpu = backend.getGPGPUContext();
          this.initializePrograms();
      }
      EmbeddingDrawer.prototype.drawPoints = function (embedding, alpha, targetTex) {
          executeSimpleEmbeddingDrawerProgram(this.gpgpu, this.simpleEmbeddingDrawerProgram, embedding.embeddingTexture, embedding.numPoints, embedding.minX, embedding.minY, embedding.maxX, embedding.maxY, embedding.pntsPerRow, embedding.numRows, this.vertexIdBuffer, alpha, 800, targetTex);
      };
      EmbeddingDrawer.prototype.drawColoredPoints = function (embedding, alpha, colors, targetTex) {
          executeColoredEmbeddingDrawerProgram(this.gpgpu, this.coloredEmbeddingDrawerProgram, embedding.embeddingTexture, embedding.numPoints, embedding.minX, embedding.minY, embedding.maxX, embedding.maxY, embedding.pntsPerRow, embedding.numRows, this.vertexIdBuffer, alpha, 800, colors, targetTex);
      };
      EmbeddingDrawer.prototype.drawTexturedPoints = function (embedding, alpha, pointsTexture, pointTextureDiameter, colorsTexture, targetTex) {
          executeTexturedPointsDrawerProgram(this.gpgpu, this.texturedPointsDrawerProgram, embedding.embeddingTexture, embedding.numPoints, embedding.minX, embedding.minY, embedding.maxX, embedding.maxY, embedding.pntsPerRow, embedding.numRows, this.vertexIdBuffer, alpha, 800, pointsTexture, pointTextureDiameter, colorsTexture, targetTex);
      };
      EmbeddingDrawer.prototype.drawPointsAndSplatTexture = function (embedding, splatTexture, drawnEmbeddingTex, textureNormalization) {
          executeSplatTextureDrawerProgram(this.gpgpu, this.splatTextureDrawerProgram, splatTexture, this.colorScaleTex, drawnEmbeddingTex, textureNormalization, 800);
      };
      EmbeddingDrawer.prototype.initializePrograms = function () {
          this.simpleEmbeddingDrawerProgram =
              createSimpleEmbeddingDrawerProgram(this.gpgpu);
          this.coloredEmbeddingDrawerProgram =
              createColoredEmbeddingDrawerProgram(this.gpgpu);
          var vertexId = new Float32Array(this.numPoints);
          var i = 0;
          for (i = 0; i < this.numPoints; ++i) {
              vertexId[i] = i;
          }
          this.vertexIdBuffer =
              tf.webgl.webgl_util.createStaticVertexBuffer(this.gpgpu.gl, vertexId);
          this.splatTextureDrawerProgram =
              createSplatTextureDrawerProgram(this.gpgpu);
          this.texturedPointsDrawerProgram =
              createTexturedPointsDrawerProgram(this.gpgpu);
          var colors = new Float32Array([
              178, 24, 43, 214, 96, 77, 244, 165, 130, 253, 219, 199, 255, 255,
              255, 209, 229, 240, 146, 197, 222, 67, 147, 195, 33, 102, 172
          ]);
          this.colorScaleTex = createAndConfigureInterpolatedTexture(this.gpgpu.gl, 9, 1, 3, colors);
      };
      return EmbeddingDrawer;
  }());

  function generateFragmentShaderSource(distanceComputationSource, numNeighbors) {
      var source = "#version 300 es\n    precision highp float;\n    uniform sampler2D data_tex;\n    uniform float num_points;\n    uniform float points_per_row_knn;\n    uniform float num_rows_knn;\n    uniform float num_neighs;\n    uniform float iteration;\n\n    #define NUM_PACKED_NEIGHBORS " + numNeighbors / 4 + "\n\n    flat in vec4 knn[NUM_PACKED_NEIGHBORS];\n    flat in int point_id;\n    in float neighbor_id;\n\n    const float MAX_DIST = 10e30;\n\n    " + distanceComputationSource + "\n\n    out vec4 fragmentColor;\n    void main() {\n      int id = int(neighbor_id/4.);\n      int channel = int(mod(neighbor_id,4.)+0.1);\n\n      if(channel == 0) {\n        fragmentColor = vec4(knn[id].r,0,0,1);\n      }else if(channel == 1) {\n        fragmentColor = vec4(knn[id].g,0,0,1);\n      }else if(channel == 2) {\n        fragmentColor = vec4(knn[id].b,0,0,1);\n      }else if(channel == 3) {\n        fragmentColor = vec4(knn[id].a,0,0,1);\n      }\n\n      //If the neighbor has a valid id i compute the distance squared\n      //otherwise I set it to invalid\n      if(fragmentColor.r >= 0.) {\n        fragmentColor.g = pointDistanceSquared(int(fragmentColor.r),point_id);\n      }else{\n        fragmentColor.g = MAX_DIST;\n      }\n    }\n  ";
      return source;
  }
  function generateVariablesAndDeclarationsSource(numNeighbors) {
      var source = "\n  precision highp float;\n  #define NEIGH_PER_ITER 20\n  #define NUM_NEIGHBORS " + numNeighbors + "\n  #define NUM_NEIGHBORS_FLOAT " + numNeighbors + ".\n  #define NUM_PACKED_NEIGHBORS " + numNeighbors / 4 + "\n  #define MAX_DIST 10e30\n\n  //attributes\n  in float vertex_id;\n  //uniforms\n  uniform sampler2D data_tex;\n  uniform sampler2D starting_knn_tex;\n  uniform float num_points;\n  uniform float points_per_row_knn;\n  uniform float num_rows_knn;\n  uniform float num_neighs;\n  uniform float iteration;\n\n  //output\n  //the indices are packed in varying vectors\n  flat out vec4 knn[NUM_PACKED_NEIGHBORS];\n  //used to recover the neighbor id in the fragment shader\n  out float neighbor_id;\n  //used to recover the point id in the fragment shader\n  //(for recomputing distances)\n  flat out int point_id;\n\n  float distances_heap[NUM_NEIGHBORS];\n  int knn_heap[NUM_NEIGHBORS];\n  ";
      return source;
  }
  var randomGeneratorSource = "\n//Random function developed by Inigo Quilez\n//https://www.shadertoy.com/view/llGSzw\nfloat hash1( uint n ) {\n    // integer hash copied from Hugo Elias\n\t  n = (n << 13U) ^ n;\n    n = n * (n * n * 15731U + 789221U) + 1376312589U;\n    return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);\n}\n\nuint hash( uint x ) {\n    x += ( x << 10u );\n    x ^= ( x >>  6u );\n    x += ( x <<  3u );\n    x ^= ( x >> 11u );\n    x += ( x << 15u );\n    return x;\n}\nfloat random( float f ) {\n    const uint mantissaMask = 0x007FFFFFu;\n    const uint one          = 0x3F800000u;\n\n    uint h = hash( floatBitsToUint( f ) );\n    h &= mantissaMask;\n    h |= one;\n\n    float  r2 = uintBitsToFloat( h );\n    return r2 - 1.0;\n}\n\n\n// #define HASHSCALE1 .1031\n// float random(float p) {\n// \tvec3 p3  = fract(vec3(p) * HASHSCALE1);\n//   p3 += dot(p3, p3.yzx + 19.19);\n//   return fract((p3.x + p3.y) * p3.z);\n// }\n\n// const vec2 randomConst = vec2(\n//   23.14069263277926, // e^pi (Gelfond's constant)\n//    2.665144142690225 // 2^sqrt(2) (Gelfond\u2013Schneider constant)\n// );\n// float random(float seed) {\n//     return fract(cos(dot(vec2(seed,seed), randomConst)) * 12345.6789);\n// }\n\n";
  var distancesInitializationSource = "\n//Reads the distances and indices from the knn texture\nvoid initializeDistances(int pnt_id) {\n  //row coordinate in the texture\n  float row = (floor(float(pnt_id)/points_per_row_knn)+0.5)/num_rows_knn;\n  //column of the first neighbor\n  float start_col = mod(float(pnt_id),points_per_row_knn)*NUM_NEIGHBORS_FLOAT;\n  for(int n = 0; n < NUM_NEIGHBORS; n++) {\n    float col = (start_col+float(n)+0.5);\n    //normalized by the width of the texture\n    col /= (points_per_row_knn*NUM_NEIGHBORS_FLOAT);\n    //reads the index in the red channel and the distances in the green one\n    vec4 init = texture(starting_knn_tex,vec2(col,row));\n\n    knn_heap[n] = int(init.r);\n    distances_heap[n] = init.g;\n  }\n}\n";
  var knnHeapSource = "\n//Swaps two points in the knn-heap\nvoid swap(int i, int j) {\n  float swap_value = distances_heap[i];\n  distances_heap[i] = distances_heap[j];\n  distances_heap[j] = swap_value;\n  int swap_id = knn_heap[i];\n  knn_heap[i] = knn_heap[j];\n  knn_heap[j] = swap_id;\n}\n\n//I can make use of the heap property but\n//I have to implement a recursive function\nbool inTheHeap(float dist_sq, int id) {\n  for(int i = 0; i < NUM_NEIGHBORS; ++i) {\n    if(knn_heap[i] == id) {\n      return true;\n    }\n  }\n  return false;\n}\n\nvoid insertInKNN(float dist_sq, int j) {\n  //not in the KNN\n  if(dist_sq >= distances_heap[0]) {\n    return;\n  }\n\n  //the point is already in the KNN\n  if(inTheHeap(dist_sq,j)) {\n    return;\n  }\n\n  //Insert in the new point in the root\n  distances_heap[0] = dist_sq;\n  knn_heap[0] = j;\n  //Sink procedure\n  int swap_id = 0;\n  while(swap_id*2+1 < NUM_NEIGHBORS) {\n    int left_id = swap_id*2+1;\n    int right_id = swap_id*2+2;\n    if(distances_heap[left_id] > distances_heap[swap_id] ||\n        (right_id < NUM_NEIGHBORS &&\n                            distances_heap[right_id] > distances_heap[swap_id])\n      ) {\n      if(distances_heap[left_id] > distances_heap[right_id]\n         || right_id >= NUM_NEIGHBORS) {\n        swap(swap_id,left_id);\n        swap_id = left_id;\n      }else{\n        swap(swap_id,right_id);\n        swap_id = right_id;\n      }\n    }else{\n      break;\n    }\n  }\n}\n";
  var vertexPositionSource = "\n  //Line positions\n  float row = (floor(float(point_id)/points_per_row_knn)+0.5)/num_rows_knn;\n  row = row*2.0-1.0;\n  if(line_id < int(1)) {\n    //for the first vertex only the position is important\n    float col = (mod(float(point_id),points_per_row_knn))/(points_per_row_knn);\n    col = col*2.0-1.0;\n    gl_Position = vec4(col,row,0,1);\n    neighbor_id = 0.;\n    return;\n  }\n  //The computation of the KNN happens only for the second vertex\n  float col = (mod(float(point_id),points_per_row_knn)+1.)/(points_per_row_knn);\n  col = col*2.0-1.0;\n  gl_Position = vec4(col,row,0,1);\n";
  function createBruteForceKNNProgram(gpgpu, numNeighbors, distanceComputationSource) {
      var vertexShaderSource = "#version 300 es\n    " +
          generateVariablesAndDeclarationsSource(numNeighbors) +
          distancesInitializationSource + distanceComputationSource +
          knnHeapSource + ("\n    void main() {\n      //Getting the id of the point and the line id (0/1)\n      point_id = int((vertex_id / 2.0) + 0.1);\n      int line_id = int(mod(vertex_id + 0.1, 2.));\n      if(float(point_id) >= num_points) {\n        return;\n      }\n\n      " + vertexPositionSource + "\n\n      //////////////////////////////////\n      //KNN computation\n      initializeDistances(point_id);\n      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {\n        //TODO make it more readable\n\n        int j = int(mod(\n                    float(point_id + i) //point id + current offset\n                    + iteration * float(NEIGH_PER_ITER) //iteration offset\n                    + 1.25,// +1 for avoid checking the point itself,\n                           // +0.25 for error compensation\n                    num_points\n                  ));\n        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j,j+1,j+2,j+3);\n        insertInKNN(dist_squared.r, j);\n        insertInKNN(dist_squared.g, j+1);\n        insertInKNN(dist_squared.b, j+2);\n        insertInKNN(dist_squared.a, j+3);\n      }\n\n      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {\n        knn[n].r = float(knn_heap[n*4]);\n        knn[n].g = float(knn_heap[n*4+1]);\n        knn[n].b = float(knn_heap[n*4+2]);\n        knn[n].a = float(knn_heap[n*4+3]);\n      }\n\n      neighbor_id = NUM_NEIGHBORS_FLOAT;\n    }\n  ");
      var knnFragmentShaderSource = generateFragmentShaderSource(distanceComputationSource, numNeighbors);
      return createVertexProgram(gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
  }
  function createRandomSamplingKNNProgram(gpgpu, numNeighbors, distanceComputationSource) {
      var vertexShaderSource = "#version 300 es\n    " +
          generateVariablesAndDeclarationsSource(numNeighbors) +
          distancesInitializationSource + randomGeneratorSource +
          distanceComputationSource + knnHeapSource + ("\n    void main() {\n      //Getting the id of the point and the line id (0/1)\n      point_id = int((vertex_id/2.0)+0.1);\n      int line_id = int(mod(vertex_id+0.1,2.));\n      if(float(point_id) >= num_points) {\n        return;\n      }\n\n      " + vertexPositionSource + "\n\n      //////////////////////////////////\n      //KNN computation\n\n      initializeDistances(point_id);\n      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {\n        //BAD SEED\n        //uint seed\n        //= uint(float(point_id) + float(NEIGH_PER_ITER)*iteration + float(i));\n        //GOOD SEED\n        //uint seed\n        //= uint(float(point_id) + float(num_points)*iteration + float(i));\n\n        float seed\n            = float(float(point_id) + float(num_points)*iteration + float(i));\n        int j0 = int(random(seed)*num_points);\n        int j1 = int(random(seed+1.)*num_points);\n        int j2 = int(random(seed+2.)*num_points);\n        int j3 = int(random(seed+3.)*num_points);\n\n        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j0,j1,j2,j3);\n        if(j0!=point_id)insertInKNN(dist_squared.r, j0);\n        if(j1!=point_id)insertInKNN(dist_squared.g, j1);\n        if(j2!=point_id)insertInKNN(dist_squared.b, j2);\n        if(j3!=point_id)insertInKNN(dist_squared.a, j3);\n      }\n\n      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {\n        knn[n].r = float(knn_heap[n*4]);\n        knn[n].g = float(knn_heap[n*4+1]);\n        knn[n].b = float(knn_heap[n*4+2]);\n        knn[n].a = float(knn_heap[n*4+3]);\n      }\n      neighbor_id = NUM_NEIGHBORS_FLOAT;\n    }\n  ");
      var knnFragmentShaderSource = generateFragmentShaderSource(distanceComputationSource, numNeighbors);
      return createVertexProgram(gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
  }
  function createKNNDescentProgram(gpgpu, numNeighbors, distanceComputationSource) {
      var vertexShaderSource = "#version 300 es\n    " +
          generateVariablesAndDeclarationsSource(numNeighbors) +
          distancesInitializationSource + randomGeneratorSource +
          distanceComputationSource + knnHeapSource + ("\n    int fetchNeighborIdFromKNNTexture(int id, int neighbor_id) {\n      //row coordinate in the texture\n      float row = (floor(float(id)/points_per_row_knn)+0.5)/num_rows_knn;\n      //column of the first neighbor\n      float start_col = mod(float(id),points_per_row_knn)*NUM_NEIGHBORS_FLOAT;\n      //column of the neighbor of interest\n      float col = (start_col+float(neighbor_id)+0.5);\n      //normalized by the width of the texture\n      col /= (points_per_row_knn*NUM_NEIGHBORS_FLOAT);\n      //reads the index in the red channel and the distances in the green one\n      vec4 knn_link = texture(starting_knn_tex,vec2(col,row));\n      //return the index\n      return int(knn_link.r);\n    }\n\n    int neighborOfANeighbor(int my_id, uint seed) {\n      //float random0 = hash1(seed);\n      float random0 = random(float(seed));\n      // random0 = random0*random0;\n      // random0 = 1. - random0;\n\n      //float random1 = hash1(seed*1798191U);\n      float random1 = random(float(seed+7U));\n      // random1 = random1*random1;\n      // random1 = 1. - random1;\n\n      //fetch a neighbor from the heap\n      int neighbor = knn_heap[int(random0*NUM_NEIGHBORS_FLOAT)];\n      //if it is not a valid pick a random point\n      if(neighbor < 0) {\n        return int(random(float(seed))*num_points);\n      }\n\n      //if it is valid I fetch from the knn graph texture one of its neighbors\n      int neighbor2ndDegree = fetchNeighborIdFromKNNTexture(\n                                    neighbor,int(random1*NUM_NEIGHBORS_FLOAT));\n      //if it is not a valid pick a random point\n      if(neighbor2ndDegree < 0) {\n        return int(random(float(seed))*num_points);\n      }\n      return neighbor2ndDegree;\n    }\n\n    void main() {\n      //Getting the id of the point and the line id (0/1)\n      point_id = int((vertex_id/2.0)+0.1);\n      int line_id = int(mod(vertex_id+0.1,2.));\n      if(float(point_id) >= num_points) {\n        return;\n      }\n      " + vertexPositionSource + "\n\n      //////////////////////////////////\n      //KNN computation\n      initializeDistances(point_id);\n      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {\n        //BAD SEED\n        //uint seed\n        //= uint(float(point_id) + float(NEIGH_PER_ITER)*iteration + float(i));\n        //GOOD SEED\n        uint seed\n              = uint(float(point_id) + float(num_points)*iteration + float(i));\n        int j0 = neighborOfANeighbor(point_id,seed);\n        int j1 = neighborOfANeighbor(point_id,seed+1U);\n        int j2 = neighborOfANeighbor(point_id,seed+2U);\n        int j3 = neighborOfANeighbor(point_id,seed+3U);\n\n        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j0,j1,j2,j3);\n        if(j0!=point_id)insertInKNN(dist_squared.r, j0);\n        if(j1!=point_id)insertInKNN(dist_squared.g, j1);\n        if(j2!=point_id)insertInKNN(dist_squared.b, j2);\n        if(j3!=point_id)insertInKNN(dist_squared.a, j3);\n      }\n\n      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {\n        knn[n].r = float(knn_heap[n*4]);\n        knn[n].g = float(knn_heap[n*4+1]);\n        knn[n].b = float(knn_heap[n*4+2]);\n        knn[n].a = float(knn_heap[n*4+3]);\n      }\n      neighbor_id = NUM_NEIGHBORS_FLOAT;\n    }\n  ");
      var knnFragmentShaderSource = generateFragmentShaderSource(distanceComputationSource, numNeighbors);
      return createVertexProgram(gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
  }
  function executeKNNProgram(gpgpu, program, dataTex, startingKNNTex, iteration, knnShape, vertexIdBuffer, targetTex) {
      var gl = gpgpu.gl;
      var oldProgram = gpgpu.program;
      var oldLineWidth = gl.getParameter(gl.LINE_WIDTH);
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, knnShape.numRows, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      if (oldLineWidth !== 1) {
          gl.lineWidth(1);
      }
      gpgpu.setProgram(program);
      gl.clearColor(0., 0., 0., 0.);
      gl.clear(gl.COLOR_BUFFER_BIT);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdBuffer); });
      tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);
      var dataTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'data_tex');
      gpgpu.setInputMatrixTexture(dataTex, dataTexLoc, 0);
      var startingKNNTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'starting_knn_tex');
      gpgpu.setInputMatrixTexture(startingKNNTex, startingKNNTexLoc, 1);
      var iterationLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'iteration');
      gl.uniform1f(iterationLoc, iteration);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, knnShape.numPoints);
      var pntsPerRowKNNLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row_knn');
      gl.uniform1f(pntsPerRowKNNLoc, knnShape.pointsPerRow);
      var numRowsKNNLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows_knn');
      gl.uniform1f(numRowsKNNLoc, knnShape.numRows);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.LINES, 0, knnShape.numPoints * 2); });
      if (oldProgram != null) {
          gpgpu.setProgram(oldProgram);
          tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
      }
      if (oldLineWidth !== 1) {
          gl.lineWidth(oldLineWidth);
      }
  }
  function createCopyDistancesProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D knn_tex;\n    uniform float width;\n    uniform float height;\n\n    void main() {\n      vec2 coordinates = gl_FragCoord.xy / vec2(width,height);\n      float distance = texture2D(knn_tex,coordinates).g;\n      gl_FragColor = vec4(distance,0,0,1);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeCopyDistancesProgram(gpgpu, program, knnTex, knnShape, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, knnShape.numRows, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var knnLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_tex');
      gpgpu.setInputMatrixTexture(knnTex, knnLoc, 0);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'width');
      gl.uniform1f(pntsPerRowLoc, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'height');
      gl.uniform1f(numRowsLoc, knnShape.numRows);
      gpgpu.executeProgram();
  }
  function createCopyIndicesProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D knn_tex;\n    uniform float width;\n    uniform float height;\n\n    void main() {\n      vec2 coordinates = gl_FragCoord.xy / vec2(width,height);\n      float id = texture2D(knn_tex,coordinates).r;\n      gl_FragColor = vec4(id,0,0,1);\n\n      if(id < 0.) {\n        gl_FragColor.b = 1.;\n      }\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeCopyIndicesProgram(gpgpu, program, knnTex, knnShape, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, knnShape.numRows, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var knnLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_tex');
      gpgpu.setInputMatrixTexture(knnTex, knnLoc, 0);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'width');
      gl.uniform1f(pntsPerRowLoc, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'height');
      gl.uniform1f(numRowsLoc, knnShape.numRows);
      gpgpu.executeProgram();
  }

  function instanceOfRearrangedData(object) {
      return 'numPoints' in object && 'pointsPerRow' in object &&
          'pixelsPerPoint' in object && 'numRows' in object;
  }
  function instanceOfCustomDataDefinition(object) {
      return 'distanceComputationCode' in object;
  }
  var KNNEstimator = (function () {
      function KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, numNeighs, verbose) {
          if (verbose != null) {
              this.verbose = verbose;
          }
          else {
              verbose = false;
          }
          this.backend = tf.ENV.findBackend('webgl');
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
          var knnPointsPerRow = Math.ceil(Math.sqrt(numNeighs * numPoints) / numNeighs);
          this.knnDataShape = {
              numPoints: numPoints,
              pixelsPerPoint: numNeighs,
              pointsPerRow: knnPointsPerRow,
              numRows: Math.ceil(numPoints / knnPointsPerRow)
          };
          this.log('knn-pntsPerRow', this.knnDataShape.pointsPerRow);
          this.log('knn-numRows', this.knnDataShape.numRows);
          this.log('knn-pixelsPerPoint', this.knnDataShape.pixelsPerPoint);
          var distanceComputationSource;
          if (instanceOfRearrangedData(dataFormat)) {
              var rearrangedData = dataFormat;
              distanceComputationSource =
                  generateDistanceComputationSource(rearrangedData);
          }
          else if (instanceOfCustomDataDefinition(dataFormat)) {
              var customDataDefinition = dataFormat;
              distanceComputationSource = customDataDefinition.distanceComputationCode;
          }
          this.initializeTextures();
          this.initilizeCustomWebGLPrograms(distanceComputationSource);
      }
      Object.defineProperty(KNNEstimator.prototype, "knnShape", {
          get: function () {
              return this.knnDataShape;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(KNNEstimator.prototype, "iteration", {
          get: function () {
              return this._iteration;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(KNNEstimator.prototype, "pointsPerIteration", {
          get: function () {
              return 20;
          },
          enumerable: true,
          configurable: true
      });
      KNNEstimator.prototype.log = function (str, obj) {
          if (this.verbose) {
              if (obj != null) {
                  console.log(str + ": \t" + obj);
              }
              else {
                  console.log(str);
              }
          }
      };
      KNNEstimator.prototype.initializeTextures = function () {
          var initNeigh = new Float32Array(this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint * 2 *
              this.knnDataShape.numRows);
          var numNeighs = this.knnDataShape.pixelsPerPoint;
          for (var i = 0; i < this.knnDataShape.numPoints; ++i) {
              for (var n = 0; n < numNeighs; ++n) {
                  initNeigh[(i * numNeighs + n) * 2] = -1;
                  initNeigh[(i * numNeighs + n) * 2 + 1] = 10e30;
              }
          }
          this.knnTexture0 = createAndConfigureTexture(this.gpgpu.gl, this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint, this.knnDataShape.numRows, 2, initNeigh);
          this.knnTexture1 = createAndConfigureTexture(this.gpgpu.gl, this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint, this.knnDataShape.numRows, 2, initNeigh);
      };
      KNNEstimator.prototype.initilizeCustomWebGLPrograms = function (distanceComputationSource) {
          this.copyDistancesProgram = createCopyDistancesProgram(this.gpgpu);
          this.copyIndicesProgram = createCopyIndicesProgram(this.gpgpu);
          this.bruteForceKNNProgram = createBruteForceKNNProgram(this.gpgpu, this.numNeighs, distanceComputationSource);
          this.randomSamplingKNNProgram = createRandomSamplingKNNProgram(this.gpgpu, this.numNeighs, distanceComputationSource);
          this.kNNDescentProgram = createKNNDescentProgram(this.gpgpu, this.numNeighs, distanceComputationSource);
          var linesVertexId = new Float32Array(this.knnDataShape.numPoints * 2);
          {
              for (var i = 0; i < this.knnDataShape.numPoints * 2; ++i) {
                  linesVertexId[i] = i;
              }
          }
          this.linesVertexIdBuffer = tf.webgl.webgl_util.createStaticVertexBuffer(this.gpgpu.gl, linesVertexId);
      };
      KNNEstimator.prototype.iterateBruteForce = function () {
          if ((this._iteration % 2) === 0) {
              this.iterateGPU(this.dataTexture, this._iteration, this.knnTexture0, this.knnTexture1);
          }
          else {
              this.iterateGPU(this.dataTexture, this._iteration, this.knnTexture1, this.knnTexture0);
          }
          ++this._iteration;
          this.gpgpu.gl.finish();
      };
      KNNEstimator.prototype.iterateRandomSampling = function () {
          if ((this._iteration % 2) === 0) {
              this.iterateRandomSamplingGPU(this.dataTexture, this._iteration, this.knnTexture0, this.knnTexture1);
          }
          else {
              this.iterateRandomSamplingGPU(this.dataTexture, this._iteration, this.knnTexture1, this.knnTexture0);
          }
          ++this._iteration;
          this.gpgpu.gl.finish();
      };
      KNNEstimator.prototype.iterateKNNDescent = function () {
          if ((this._iteration % 2) === 0) {
              this.iterateKNNDescentGPU(this.dataTexture, this._iteration, this.knnTexture0, this.knnTexture1);
          }
          else {
              this.iterateKNNDescentGPU(this.dataTexture, this._iteration, this.knnTexture1, this.knnTexture0);
          }
          ++this._iteration;
          this.gpgpu.gl.finish();
      };
      KNNEstimator.prototype.knn = function () {
          if ((this._iteration % 2) === 0) {
              return this.knnTexture0;
          }
          else {
              return this.knnTexture1;
          }
      };
      KNNEstimator.prototype.distancesTensor = function () {
          var _this = this;
          return tf.tidy(function () {
              var distances = tf.zeros([
                  _this.knnDataShape.numRows,
                  _this.knnDataShape.pointsPerRow * _this.knnDataShape.pixelsPerPoint
              ]);
              var knnTexture = _this.knn();
              executeCopyDistancesProgram(_this.gpgpu, _this.copyDistancesProgram, knnTexture, _this.knnDataShape, _this.backend.getTexture(distances.dataId));
              return distances;
          });
      };
      KNNEstimator.prototype.indicesTensor = function () {
          var _this = this;
          return tf.tidy(function () {
              var indices = tf.zeros([
                  _this.knnDataShape.numRows,
                  _this.knnDataShape.pointsPerRow * _this.knnDataShape.pixelsPerPoint
              ]);
              var knnTexture = _this.knn();
              executeCopyIndicesProgram(_this.gpgpu, _this.copyIndicesProgram, knnTexture, _this.knnDataShape, _this.backend.getTexture(indices.dataId));
              return indices;
          });
      };
      KNNEstimator.prototype.iterateGPU = function (dataTexture, _iteration, startingKNNTexture, targetTexture) {
          executeKNNProgram(this.gpgpu, this.bruteForceKNNProgram, dataTexture, startingKNNTexture, _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
      };
      KNNEstimator.prototype.iterateRandomSamplingGPU = function (dataTexture, _iteration, startingKNNTexture, targetTexture) {
          executeKNNProgram(this.gpgpu, this.randomSamplingKNNProgram, dataTexture, startingKNNTexture, _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
      };
      KNNEstimator.prototype.iterateKNNDescentGPU = function (dataTexture, _iteration, startingKNNTexture, targetTexture) {
          executeKNNProgram(this.gpgpu, this.kNNDescentProgram, dataTexture, startingKNNTexture, _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
      };
      return KNNEstimator;
  }());

  /*! *****************************************************************************
  Copyright (c) Microsoft Corporation. All rights reserved.
  Licensed under the Apache License, Version 2.0 (the "License"); you may not use
  this file except in compliance with the License. You may obtain a copy of the
  License at http://www.apache.org/licenses/LICENSE-2.0

  THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
  WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
  MERCHANTABLITY OR NON-INFRINGEMENT.

  See the Apache Version 2.0 License for specific language governing permissions
  and limitations under the License.
  ***************************************************************************** */

  function __awaiter(thisArg, _arguments, P, generator) {
      return new (P || (P = Promise))(function (resolve, reject) {
          function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
          function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
          function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
          step((generator = generator.apply(thisArg, _arguments || [])).next());
      });
  }

  function __generator(thisArg, body) {
      var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
      return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
      function verb(n) { return function (v) { return step([n, v]); }; }
      function step(op) {
          if (f) throw new TypeError("Generator is already executing.");
          while (_) try {
              if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
              if (y = 0, t) op = [0, t.value];
              switch (op[0]) {
                  case 0: case 1: t = op; break;
                  case 4: _.label++; return { value: op[1], done: false };
                  case 5: _.label++; y = op[1]; op = [0]; continue;
                  case 7: op = _.ops.pop(); _.trys.pop(); continue;
                  default:
                      if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                      if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                      if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                      if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                      if (t[2]) _.ops.pop();
                      _.trys.pop(); continue;
              }
              op = body.call(thisArg, _);
          } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
          if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
      }
  }

  function tensorToDataTexture(tensor) {
      return __awaiter(this, void 0, void 0, function () {
          var inputShape, backend, gpgpu, numPoints, numDimensions, numChannels, pixelsPerPoint, pointsPerRow, numRows, tensorData, textureValues, p, tensorOffset, textureOffset, d, texture, shape;
          return __generator(this, function (_a) {
              inputShape = tensor.shape;
              if (inputShape.length !== 2) {
                  throw Error('tensorToDataTexture: input tensor must be 2-dimensional');
              }
              backend = tf.ENV.findBackend('webgl');
              if (backend === null) {
                  throw Error('WebGL backend is not available');
              }
              gpgpu = backend.getGPGPUContext();
              numPoints = inputShape[0];
              numDimensions = inputShape[1];
              numChannels = 4;
              pixelsPerPoint = Math.ceil(numDimensions / numChannels);
              pointsPerRow = Math.floor(Math.sqrt(numPoints * pixelsPerPoint) / pixelsPerPoint);
              numRows = Math.ceil(numPoints / pointsPerRow);
              tensorData = tensor.dataSync();
              textureValues = new Float32Array(pointsPerRow * pixelsPerPoint * numRows * numChannels);
              for (p = 0; p < numPoints; ++p) {
                  tensorOffset = p * numDimensions;
                  textureOffset = p * pixelsPerPoint * numChannels;
                  for (d = 0; d < numDimensions; ++d) {
                      textureValues[textureOffset + d] = tensorData[tensorOffset + d];
                  }
              }
              texture = createAndConfigureTexture(gpgpu.gl, pointsPerRow * pixelsPerPoint, numRows, 4, textureValues);
              shape = { numPoints: numPoints, pointsPerRow: pointsPerRow, numRows: numRows, pixelsPerPoint: pixelsPerPoint };
              return [2, { shape: shape, texture: texture }];
          });
      });
  }

  function createEmbeddingSplatterProgram(gpgpu) {
      var vertexShaderSource = "#version 300 es\n    precision highp float;\n    in float vertex_id;\n\n    uniform sampler2D embedding_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float kernel_support;\n    uniform float points_per_row;\n    uniform float num_rows;\n\n    out vec2 kernel_coords;\n\n    void main() {\n      //TODO Clean up and check performance loss due to the conversions\n      uint pnt_id = uint((vertex_id / 4.0) + 0.1);\n      uint quad_id = uint(mod(vertex_id + 0.1,4.));\n\n      uint row    = uint((float(pnt_id) + 0.1)/points_per_row);\n      uint column = uint(float(pnt_id) - float(row) * points_per_row);\n\n      float width = (points_per_row * 2.0);\n      float row_tex = (float(row) + 0.5) / num_rows;\n      vec2 tex_coords_x = vec2((float(column) * 2. + 0.5) / width, row_tex);\n      vec2 tex_coords_y = vec2((float(column) * 2. + 1.5) / width, row_tex);\n\n      float x_pnt = texture(embedding_tex,tex_coords_x).r;\n      float y_pnt = texture(embedding_tex,tex_coords_y).r;\n      vec2 vertex_coords = vec2(x_pnt,y_pnt);\n\n      if(quad_id == uint(0)) {kernel_coords = vec2(-1,-1);}\n      else if(quad_id == uint(1)) {kernel_coords = vec2(1,-1);}\n      else if(quad_id == uint(2)) {kernel_coords = vec2(1,1);}\n      else if(quad_id == uint(3)) {kernel_coords = vec2(-1,1);}\n\n      vertex_coords += kernel_coords * kernel_support;      // embedding space\n      vertex_coords = (vertex_coords - minV) / (maxV-minV); //  0:1 space\n      vertex_coords = vertex_coords * 2.0 - 1.0;            // -1:1 space\n\n      gl_Position = vec4(vertex_coords,0,1);\n    }\n  ";
      var fragmentShaderSource = "#version 300 es\n    precision highp float;\n    uniform sampler2D kernel_tex;\n    in vec2 kernel_coords;\n    out vec4 fragmentColor;\n\n    void main() {\n      fragmentColor = texture(kernel_tex,(kernel_coords + 1.) / 2.0);\n    }\n  ";
      return createVertexProgram(gpgpu.gl, vertexShaderSource, fragmentShaderSource);
  }
  function executeEmbeddingSplatterProgram(gpgpu, program, targetTex, embeddingTex, kernelTex, targetTexDiameter, numPoints, minX, minY, maxX, maxY, kernelSupport, pntsPerRow, numRows, vertexIdBuffer) {
      var gl = gpgpu.gl;
      var oldProgram = gpgpu.program;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, targetTexDiameter, targetTexDiameter);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      gl.clearColor(0., 0., 0., 0.);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdBuffer); });
      tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
      var kernelLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'kernel_tex');
      gpgpu.setInputMatrixTexture(kernelTex, kernelLocation, 1);
      var kernelSupportLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'kernel_support');
      gl.uniform1f(kernelSupportLoc, kernelSupport);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
      gl.uniform2f(minLoc, minX, minY);
      var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
      gl.uniform2f(maxLoc, maxX, maxY);
      tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.TRIANGLES, 0, numPoints * 2 * 3); });
      gl.disable(gl.BLEND);
      if (oldProgram != null) {
          gpgpu.setProgram(oldProgram);
          tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
      }
  }
  function createQInterpolatorProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D embedding_tex;\n    uniform sampler2D splat_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n\n    void main() {\n      vec2 pnt_location = gl_FragCoord.xy - vec2(0.5,0.5);\n\n      if(pnt_location.y * points_per_row + pnt_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n\n      float emb_width = (points_per_row * 2.0);\n      float emb_row_coord = (pnt_location.y + 0.5) / num_rows;\n      vec2 emb_coords_x\n              = vec2((pnt_location.x * 2.+0.5) / emb_width, emb_row_coord);\n      vec2 emb_coords_y\n              = vec2((pnt_location.x * 2. + 1.5) / emb_width, emb_row_coord);\n\n      float x_pnt = texture2D(embedding_tex,emb_coords_x).r;\n      float y_pnt = texture2D(embedding_tex,emb_coords_y).r;\n\n      vec2 splat_coords = vec2(x_pnt,y_pnt);\n      splat_coords = (splat_coords - minV) / (maxV - minV); //  0:1 space\n\n      float q = (texture2D(splat_tex,splat_coords).r - 1.);\n\n      gl_FragColor = vec4(q, 0, 0, 1);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeQInterpolatorProgram(gpgpu, program, splatTex, embeddingTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
      var splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'splat_tex');
      gpgpu.setInputMatrixTexture(splatTex, splatLocation, 1);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, numPoints);
      var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
      gl.uniform2f(minLoc, minX, minY);
      var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
      gl.uniform2f(maxLoc, maxX, maxY);
      gpgpu.executeProgram();
  }
  function createXYInterpolatorProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D embedding_tex;\n    uniform sampler2D splat_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float eta;\n\n    void main() {\n      vec2 pnt_location = gl_FragCoord.xy - vec2(0.5,0.5);\n      pnt_location.x = floor(pnt_location.x/2.+0.1);\n\n      if(pnt_location.y*points_per_row + pnt_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n\n      float emb_width = (points_per_row * 2.0);\n      float emb_row_coord = (pnt_location.y + 0.5) / num_rows;\n      vec2 emb_coords_x\n              = vec2((pnt_location.x * 2. + 0.5) / emb_width, emb_row_coord);\n      vec2 emb_coords_y\n              = vec2((pnt_location.x * 2. + 1.5) / emb_width, emb_row_coord);\n\n      float x_pnt = texture2D(embedding_tex,emb_coords_x).r;\n      float y_pnt = texture2D(embedding_tex,emb_coords_y).r;\n\n      vec2 splat_coords = vec2(x_pnt,y_pnt);\n      splat_coords = (splat_coords - minV) / (maxV - minV); //  0:1 space\n\n      float q = 0.;\n      if(mod(gl_FragCoord.x - 0.5,2.) < 0.5 ) {\n        q = texture2D(splat_tex,splat_coords).g * eta * 2.;\n      }else{\n        q = texture2D(splat_tex,splat_coords).b * eta * 2.;\n      }\n\n      gl_FragColor = vec4(q,0.0,0.0,1);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeXYInterpolatorProgram(gpgpu, program, splatTex, embeddingTex, targetTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, eta) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
      var splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'splat_tex');
      gpgpu.setInputMatrixTexture(splatTex, splatLocation, 1);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, numPoints);
      var etaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'eta');
      gl.uniform1f(etaLoc, eta);
      var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
      gl.uniform2f(minLoc, minX, minY);
      var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
      gl.uniform2f(maxLoc, maxX, maxY);
      gpgpu.executeProgram();
  }
  function createAttractiveForcesComputationProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n\n    uniform sampler2D embedding_tex;\n    uniform sampler2D offset_tex;\n    uniform sampler2D neigh_id_tex;\n    uniform sampler2D neigh_prob_tex;\n\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float num_neighs_per_row;\n    uniform float eta;\n\n    void main() {\n      //add for nearest pixel interpolation\n      vec2 half_pxl = vec2(0.5,0.5);\n\n      // Dimension of the fragment\n      // 0 -> x :1 -> y\n      float dimension = mod(gl_FragCoord.x - 0.4,2.);\n\n      //Point location in the [points_per_row,num_rows] space\n      vec2 i_location = gl_FragCoord.xy - half_pxl;\n      i_location.x = floor(i_location.x / 2. + 0.1);\n\n      //just an extra fragment -> return\n      if(i_location.y*points_per_row + i_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n\n      //Offset coordinates for the point\n      vec2 offset_coord = (i_location + half_pxl) /\n                                              vec2(points_per_row,num_rows);\n      //Offset information ...\n      vec4 offset_info  = texture2D(offset_tex,offset_coord);\n      //... contains the number of neighbors for the point ...\n      float num_neighs  = offset_info.z;\n      //... and the coordinates of the firts neigh in the neigh textures\n      vec2 offset_neigh = offset_info.xy;\n\n      //Computing the coordinates of the point in the texture\n      //_i represent the point to move, _j the neighbors\n      float emb_width = (points_per_row * 2.0);\n      float emb_row_i = (i_location.y + 0.5) / num_rows;\n      vec2 x_i_coord = vec2((i_location.x * 2. + 0.5) / emb_width, emb_row_i);\n      vec2 y_i_coord = vec2((i_location.x * 2. + 1.5) / emb_width, emb_row_i);\n      //getting the coordinates in the embedding\n      float x_i = texture2D(embedding_tex,x_i_coord).r;\n      float y_i = texture2D(embedding_tex,y_i_coord).r;\n\n      //Sum of all attractive forces\n      float sum_pos = 0.;\n\n      //Can't be higher than 1000 (perplexity is usually around 30)\n      //and a 'while' can't be used\n      for(int n = 0; n < 2000; ++n) {\n        //Actual check on number of neighbors\n        if(float(n) >= num_neighs) {\n          break;\n        }\n\n        //Get the id and the probability for the neighbor\n        float pij = texture2D(neigh_prob_tex,\n                              (offset_neigh + half_pxl) / num_neighs_per_row\n                             ).r;\n        float neigh_id = texture2D(neigh_id_tex,\n                                  (offset_neigh + half_pxl) / num_neighs_per_row\n                                  ).r;\n\n        //Getting the coordinates of the neighbor\n        vec2 j_location = vec2(mod(neigh_id + 0.1, points_per_row),\n                               floor(neigh_id / points_per_row + 0.1));\n        float emb_row_j = (j_location.y + 0.5) / num_rows;\n        vec2 x_j_coord = vec2((j_location.x * 2. + 0.5) / emb_width, emb_row_j);\n        vec2 y_j_coord = vec2((j_location.x * 2. + 1.5) / emb_width, emb_row_j);\n        float x_j = texture2D(embedding_tex,x_j_coord).r;\n        float y_j = texture2D(embedding_tex,y_j_coord).r;\n\n        //Actual computation of the attractive forces\n        float dist_x    = (x_i - x_j);\n        float dist_y    = (y_i - y_j);\n        float qij       = 1. / (1. + dist_x * dist_x + dist_y * dist_y);\n        //the update depends on the dimension that this fragment represents\n        if(dimension < 0.5) {\n          // * 4 / (num_points*2) -> * 2 / num_points\n          sum_pos += eta * 2. * pij * qij * dist_x / (num_points);\n        }else{\n          sum_pos += eta * 2. * pij * qij * dist_y / (num_points);\n        }\n\n        //Increase the coordinate of the neigh in the neigh_id texture\n        offset_neigh.x += 1.;\n        //check if the new neigh is in the next row\n        if(offset_neigh.x + 0.2 > num_neighs_per_row) {\n          //in that case reset the column and increase the row\n          offset_neigh.x = 0.1;\n          offset_neigh.y += 1.0;\n        }\n      }\n\n      //The output is the sum of the attractive forces\n      gl_FragColor = vec4(sum_pos,0,0,0);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeAttractiveForcesComputationProgram(gpgpu, program, embeddingTex, offsetTex, neighIdTex, neighProbTex, numPoints, neighsPerRow, pntsPerRow, numRows, eta, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
      gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 3);
      var offsetLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'offset_tex');
      gpgpu.setInputMatrixTexture(offsetTex, offsetLocation, 2);
      var neighIdLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'neigh_id_tex');
      gpgpu.setInputMatrixTexture(neighIdTex, neighIdLocation, 1);
      var neighProbLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'neigh_prob_tex');
      gpgpu.setInputMatrixTexture(neighProbTex, neighProbLocation, 0);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var etaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'eta');
      gl.uniform1f(etaLoc, eta);
      var neighsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_neighs_per_row');
      gl.uniform1f(neighsPerRowLoc, neighsPerRow);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, numPoints);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      gpgpu.executeProgram();
  }
  function createEmbeddingInitializationProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n\n    uniform sampler2D random_tex;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n\n    void main() {\n      //add for nearest pixel interpolation\n      vec2 half_pxl = vec2(0.5,0.5);\n\n      // Dimension of the fragment\n      // 0 -> x :1 -> y\n      float dimension = mod(gl_FragCoord.x - 0.4,2.);\n      vec2 pnt_location = gl_FragCoord.xy - half_pxl;\n      pnt_location.x = floor(pnt_location.x / 2.);\n\n      //just an extra fragment -> return\n      if(pnt_location.y*points_per_row + pnt_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,1);\n        return;\n      }\n\n      float width = (points_per_row * 2.0);\n      float row_coord = (pnt_location.y + 0.5)/num_rows;\n      vec2 rad_coord = vec2((pnt_location.x * 2. + 0.5) / width, row_coord);\n      vec2 ang_coord = vec2((pnt_location.x * 2. + 1.5) / width, row_coord);\n\n      float rad = texture2D(random_tex,rad_coord).r * 3.;\n      float ang = texture2D(random_tex,ang_coord).r * 3.1415 * 2.;\n\n      gl_FragColor = vec4(rad,ang,0,1);\n\n      if(dimension < 0.5) {\n        gl_FragColor = vec4(cos(ang) * rad,0,0,0);\n      }else{\n        gl_FragColor = vec4(sin(ang) * rad,0,0,0);\n      }\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeEmbeddingInitializationProgram(gpgpu, program, randomTex, numPoints, pntsPerRow, numRows, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var randomLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'random_tex');
      gpgpu.setInputMatrixTexture(randomTex, randomLoc, 3);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, numPoints);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      gpgpu.executeProgram();
  }
  function createDistributionParametersComputationProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n\n    #define MAX_NEIGHBORS 128\n    #define MAX_ITERATIONS 500\n    #define FLOAT_MAX 10e30\n    #define TOLERANCE 1e-5\n\n    uniform sampler2D knn_graph_tex;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float num_neighs;\n    uniform float perplexity;\n\n    vec2 half_pixel = vec2(0.5,0.5);\n    float distances_squared[MAX_NEIGHBORS];\n\n    void readDistances(vec2 point_location) {\n      for(int n = 0; n < MAX_NEIGHBORS; ++n ) {\n        if(float(n) >= num_neighs-0.1) {\n          break;\n        }\n        vec2 knn_coordinates = vec2(\n            (point_location.x * num_neighs + float(n) + half_pixel.x)\n                                        /(points_per_row * num_neighs),\n            (point_location.y + half_pixel.y) / num_rows\n        );\n        distances_squared[n] = texture2D(knn_graph_tex,knn_coordinates).g;\n      }\n    }\n\n    void main() {\n      vec2 point_location = gl_FragCoord.xy - half_pixel;\n      //invalid points\n      if(point_location.y*points_per_row + point_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n      readDistances(point_location);\n\n      //Beta computation\n      float beta = 1.;\n      float max_beta = FLOAT_MAX;\n      float min_beta = -FLOAT_MAX;\n      //To avoid computing the log at every iteration\n      float log_perplexity = log(perplexity);\n      float entropy_diff = 0.;\n      float entropy = 0.;\n      float sum_probabilities = 0.;\n\n      //Binary search for a maximum of MAX_ITERATIONS\n      for(int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {\n        //At every iteration I compute the\n        //entropy enforced by the current beta\n        sum_probabilities = 0.;\n        entropy = 0.;\n        for(int n = 0; n < MAX_NEIGHBORS; ++n ) {\n          if(float(n) >= num_neighs-0.1) {\n            break;\n          }\n          float neigh_probability = exp(-beta * distances_squared[n]);\n          sum_probabilities += neigh_probability;\n          entropy += beta * distances_squared[n] * neigh_probability;\n        }\n\n        entropy = entropy / sum_probabilities + log(sum_probabilities);\n        entropy_diff = entropy - log_perplexity;\n\n        //the current beta is good enough!\n        if(entropy_diff < TOLERANCE && -entropy_diff < TOLERANCE) {\n          break;\n        }\n\n        if(entropy_diff > 0.) {\n          min_beta = beta;\n          if(max_beta == FLOAT_MAX || max_beta == -FLOAT_MAX) {\n            beta *= 2.;\n          }else{\n            beta = (beta + max_beta) / 2.;\n          }\n        }else{\n          max_beta = beta;\n          if(min_beta == -FLOAT_MAX || min_beta == FLOAT_MAX) {\n            beta /= 2.;\n          }else{\n            beta = (beta + min_beta) / 2.;\n          }\n        }\n      }\n      gl_FragColor = vec4(beta,sum_probabilities,0,1);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeDistributionParametersComputationProgram(gpgpu, program, knnGraph, numPoints, numNeighs, pntsPerRow, numRows, perplexity, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var knnGraphLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_graph_tex');
      gpgpu.setInputMatrixTexture(knnGraph, knnGraphLoc, 0);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, numPoints);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numNeighsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_neighs');
      gl.uniform1f(numNeighsLoc, numNeighs);
      var perplexityLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'perplexity');
      gl.uniform1f(perplexityLoc, perplexity);
      gpgpu.executeProgram();
  }
  function createGaussiaDistributionsFromDistancesProgram(gpgpu) {
      var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D knn_graph_tex;\n    uniform sampler2D parameters_tex;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float num_neighs;\n\n    vec2 half_pixel = vec2(0.5,0.5);\n\n    void main() {\n      vec2 point_location = gl_FragCoord.xy - half_pixel;\n      point_location.x = floor(point_location.x / num_neighs);\n\n      //invalid points\n      if(point_location.y*points_per_row + point_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n      float distance_squared\n            = texture2D(knn_graph_tex,\n                        gl_FragCoord.xy /\n                        vec2(points_per_row*num_neighs,num_rows)\n                      ).g;\n      vec2 parameters\n            = texture2D(parameters_tex,\n                        (point_location.xy + half_pixel)/\n                        vec2(points_per_row,num_rows)\n                      ).rg;\n      float beta = parameters.r;\n      float normalization = parameters.g;\n\n      float probability = exp(-beta * distance_squared) / normalization;\n      //check for NaN for degenerated knn (d = 0 for every point)\n      if (!(probability < 0.0 || 0.0 < probability || probability == 0.0)) {\n        probability = 0.;\n      }\n\n      gl_FragColor = vec4(probability,0,0,1);\n    }\n  ";
      return gpgpu.createProgram(fragmentShaderSource);
  }
  function executeGaussiaDistributionsFromDistancesProgram(gpgpu, program, knnGraph, parameters, numPoints, numNeighs, pntsPerRow, numRows, targetTex) {
      var gl = gpgpu.gl;
      if (targetTex != null) {
          gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * numNeighs);
      }
      else {
          tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
      }
      gpgpu.setProgram(program);
      var knnGraphLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_graph_tex');
      gpgpu.setInputMatrixTexture(knnGraph, knnGraphLoc, 0);
      var parametersLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'parameters_tex');
      gpgpu.setInputMatrixTexture(parameters, parametersLoc, 1);
      var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
      gl.uniform1f(numRowsLoc, numRows);
      var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
      gl.uniform1f(numPointsLoc, numPoints);
      var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
      gl.uniform1f(pntsPerRowLoc, pntsPerRow);
      var numNeighsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_neighs');
      gl.uniform1f(numNeighsLoc, numNeighs);
      gpgpu.executeProgram();
  }

  var TSNEOptimizer = (function () {
      function TSNEOptimizer(numPoints, verbose, splatTextureDiameter, kernelTextureRadius) {
          if (verbose != null) {
              this.verbose = verbose;
          }
          else {
              verbose = false;
          }
          this.log('Initializing the tSNE gradient descent computation...');
          this.numPoints = numPoints;
          this._iteration = 0;
          var webglVersion = tf.ENV.get('WEBGL_VERSION');
          if (webglVersion === 1) {
              throw Error('WebGL version 1 is not supported by tfjs-tsne');
          }
          this.backend = tf.ENV.findBackend('webgl');
          if (this.backend === null) {
              throw Error('WebGL backend is not available');
          }
          this.gpgpu = this.backend.getGPGPUContext();
          tf.webgl.webgl_util.getExtensionOrThrow(this.gpgpu.gl, 'OES_texture_float_linear');
          this.pointsPerRow = Math.ceil(Math.sqrt(numPoints * 2));
          if (this.pointsPerRow % 2 === 1) {
              ++this.pointsPerRow;
          }
          this.pointsPerRow /= 2;
          this.numRows = Math.ceil(numPoints / this.pointsPerRow);
          this.log('\t# points per row', this.pointsPerRow);
          this.log('\t# rows', this.numRows);
          this._eta = 2500;
          this._momentum = tf.scalar(0.8);
          this.rawExaggeration =
              [{ iteration: 200, value: 4 }, { iteration: 600, value: 1 }];
          this.updateExaggeration();
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
          this.initilizeCustomWebGLPrograms();
          this.initializeEmbedding();
          this.log('\tEmbedding', this.embedding);
          this.log('\tGradient', this.gradient);
      }
      Object.defineProperty(TSNEOptimizer.prototype, "minX", {
          get: function () {
              return this._minX;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "maxX", {
          get: function () {
              return this._maxX;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "minY", {
          get: function () {
              return this._minY;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "maxY", {
          get: function () {
              return this._maxY;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "iteration", {
          get: function () {
              return this._iteration;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "numberOfPoints", {
          get: function () {
              return this.numPoints;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "numberOfPointsPerRow", {
          get: function () {
              return this.pointsPerRow;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "numberOfRows", {
          get: function () {
              return this.numRows;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "embeddingCoordinates", {
          get: function () {
              return this.embedding;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "embedding2D", {
          get: function () {
              var _this = this;
              var result = tf.tidy(function () {
                  var reshaped = _this.embedding.reshape([_this.numRows * _this.pointsPerRow, 2])
                      .slice([0, 0], [_this.numPoints, 2]);
                  return reshaped;
              });
              return result;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "embeddingTexture", {
          get: function () {
              return this.backend.getTexture(this.embedding.dataId);
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "splatTexture", {
          get: function () {
              return this._splatTexture;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "normalizationQ", {
          get: function () {
              return this._normQ;
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "exaggerationAtCurrentIteration", {
          get: function () {
              return this._exaggeration.get();
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "exaggeration", {
          get: function () {
              return this.rawExaggeration;
          },
          set: function (ex) {
              this.rawExaggeration = ex;
              if (typeof ex === 'number') {
                  if (ex < 1) {
                      throw Error('Exaggeration must be greater then or equal to one');
                  }
              }
              else {
                  for (var i = 0; i < ex.length; ++i) {
                      if (ex[i].value < 1) {
                          throw Error('Exaggeration must be greater then or equal to one');
                      }
                      if (ex[i].iteration < 0) {
                          throw Error('Piecewise linear exaggeration function \
                                        must have poistive iteration values');
                      }
                  }
                  for (var i = 0; i < ex.length - 1; ++i) {
                      if (ex[i].iteration >= ex[i + 1].iteration) {
                          throw Error('Piecewise linear exaggeration function \
                                      must have increasing iteration values');
                      }
                  }
                  if (ex.length === 1) {
                      this.exaggeration = ex[0].value;
                  }
              }
              this.updateExaggeration();
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "momentum", {
          get: function () {
              return this._momentum.get();
          },
          set: function (mom) {
              if (mom < 0 || mom > 1) {
                  throw Error('Momentum must be in the [0,1] range');
              }
              this._momentum.dispose();
              this._momentum = tf.scalar(mom);
          },
          enumerable: true,
          configurable: true
      });
      Object.defineProperty(TSNEOptimizer.prototype, "eta", {
          get: function () {
              return this._eta;
          },
          set: function (eta) {
              if (eta <= 0) {
                  throw Error('ETA must be greater then zero');
              }
              this._eta = eta;
          },
          enumerable: true,
          configurable: true
      });
      TSNEOptimizer.prototype.dispose = function () {
          this.embedding.dispose();
          this.gradient.dispose();
          this._momentum.dispose();
          this._exaggeration.dispose();
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
          this.gpgpu.gl.deleteBuffer(this.splatVertexIdBuffer);
          this.gpgpu.gl.deleteProgram(this.embeddingInitializationProgram);
          this.gpgpu.gl.deleteProgram(this.embeddingSplatterProgram);
          this.gpgpu.gl.deleteProgram(this.qInterpolatorProgram);
          this.gpgpu.gl.deleteProgram(this.xyInterpolatorProgram);
          this.gpgpu.gl.deleteProgram(this.attractiveForcesProgram);
          this.gpgpu.gl.deleteProgram(this.distributionParameterssComputationProgram);
          this.gpgpu.gl.deleteProgram(this.gaussiaDistributionsFromDistancesProgram);
      };
      TSNEOptimizer.prototype.initializeEmbedding = function () {
          return __awaiter(this, void 0, void 0, function () {
              var _this = this;
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          if (this.embedding != null) {
                              this.embedding.dispose();
                          }
                          if (this.gradient != null) {
                              this.gradient.dispose();
                          }
                          this.gradient = tf.zeros([this.numRows, this.pointsPerRow * 2]);
                          this.embedding = tf.tidy(function () {
                              var randomData = tf.randomUniform([_this.numRows, _this.pointsPerRow * 2]);
                              var embedding = tf.zeros([_this.numRows, _this.pointsPerRow * 2]);
                              _this.initializeEmbeddingPositions(embedding, randomData);
                              return embedding;
                          });
                          return [4, this.computeBoundaries()];
                      case 1:
                          _a.sent();
                          this.log('\tmin X', this._minX);
                          this.log('\tmax X', this._maxX);
                          this.log('\tmin Y', this._minY);
                          this.log('\tmax Y', this._maxY);
                          this._iteration = 0;
                          return [2];
                  }
              });
          });
      };
      TSNEOptimizer.prototype.initializeNeighbors = function (numNeighPerRow, offsets, probabilities, neighIds) {
          this.numNeighPerRow = numNeighPerRow;
          this.probOffsetTexture = offsets;
          this.probTexture = probabilities;
          this.probNeighIdTexture = neighIds;
      };
      TSNEOptimizer.prototype.initializeNeighborsFromKNNGraph = function (numPoints, numNeighbors, distances, indices) {
          return __awaiter(this, void 0, void 0, function () {
              var pointsPerRow, numRows, dataShape, textureValues, i, n, id, knnGraphTexture;
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          pointsPerRow = Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors);
                          numRows = Math.ceil(numPoints / pointsPerRow);
                          dataShape = { numPoints: numPoints, pixelsPerPoint: numNeighbors, numRows: numRows, pointsPerRow: pointsPerRow };
                          textureValues = new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
                          for (i = 0; i < numPoints; ++i) {
                              for (n = 0; n < numNeighbors; ++n) {
                                  id = (i * numNeighbors + n);
                                  textureValues[id * 2] = indices[id];
                                  textureValues[id * 2 + 1] = distances[id];
                              }
                          }
                          knnGraphTexture = createAndConfigureTexture(this.gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);
                          return [4, this.initializeNeighborsFromKNNTexture(dataShape, knnGraphTexture)];
                      case 1:
                          _a.sent();
                          this.gpgpu.gl.deleteTexture(knnGraphTexture);
                          return [2];
                  }
              });
          });
      };
      TSNEOptimizer.prototype.initializeNeighborsFromKNNTexture = function (shape, knnGraph) {
          return __awaiter(this, void 0, void 0, function () {
              var distributionParameters, gaussianDistributions, perplexity, gaussianDistributionsData, knnIndices, copyIndicesProgram, knnIndicesData, asymNeighIds, i, d, linearId, neighborCounter, neighborLinearOffset, i, i, check, maxValue, maxId, i, offsets, pointOffset, i, totalNeighbors, probabilities, neighIds, assignedNeighborCounter, i, n, linearId, pointId, probability, symMatrixDirectId, symMatrixIndirectId;
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          this.log('Asymmetric neighborhood initialization...');
                          if (shape.numPoints !== this.numPoints) {
                              throw new Error("KNN size and number of points must agree" +
                                  ("(" + shape.numPoints + "," + this.numPoints + ")"));
                          }
                          distributionParameters = createAndConfigureTexture(this.gpgpu.gl, shape.pointsPerRow, shape.numRows, 2);
                          gaussianDistributions = tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);
                          perplexity = shape.pixelsPerPoint / 3;
                          this.computeDistributionParameters(distributionParameters, shape, perplexity, knnGraph);
                          this.computeGaussianDistributions(gaussianDistributions, distributionParameters, shape, knnGraph);
                          return [4, gaussianDistributions.data()];
                      case 1:
                          gaussianDistributionsData = _a.sent();
                          this.log('Gaussian distributions', gaussianDistributions);
                          knnIndices = tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);
                          copyIndicesProgram = createCopyIndicesProgram(this.gpgpu);
                          executeCopyIndicesProgram(this.gpgpu, copyIndicesProgram, knnGraph, shape, this.backend.getTexture(knnIndices.dataId));
                          return [4, knnIndices.data()];
                      case 2:
                          knnIndicesData = _a.sent();
                          this.log('knn Indices', knnIndices);
                          asymNeighIds = new Float32Array(shape.numPoints * shape.pixelsPerPoint);
                          for (i = 0; i < this.numPoints; ++i) {
                              for (d = 0; d < shape.pixelsPerPoint; ++d) {
                                  linearId = i * shape.pixelsPerPoint + d;
                                  asymNeighIds[i * shape.pixelsPerPoint + d] = knnIndicesData[linearId];
                              }
                          }
                          this.log('NeighIds', asymNeighIds);
                          neighborCounter = new Uint32Array(this.numPoints);
                          neighborLinearOffset = new Uint32Array(this.numPoints);
                          for (i = 0; i < shape.numPoints * shape.pixelsPerPoint; ++i) {
                              ++neighborCounter[asymNeighIds[i]];
                          }
                          for (i = 1; i < shape.numPoints; ++i) {
                              neighborLinearOffset[i] = neighborLinearOffset[i - 1] +
                                  neighborCounter[i - 1] + shape.pixelsPerPoint;
                          }
                          this.log('Counter', neighborCounter);
                          this.log('Linear offset', neighborLinearOffset);
                          check = 0;
                          maxValue = 0;
                          maxId = 0;
                          for (i = 0; i < neighborCounter.length; ++i) {
                              check += neighborCounter[i];
                              if (neighborCounter[i] > maxValue) {
                                  maxValue = neighborCounter[i];
                                  maxId = i;
                              }
                          }
                          this.log('Number of indirect links', check);
                          this.log('Most central point', maxId);
                          this.log('Number of indirect links for the central point', maxValue);
                          this.numNeighPerRow =
                              Math.ceil(Math.sqrt(shape.numPoints * shape.pixelsPerPoint * 2));
                          this.log('numNeighPerRow', this.numNeighPerRow);
                          {
                              offsets = new Float32Array(this.pointsPerRow * this.numRows * 3);
                              pointOffset = 0;
                              for (i = 0; i < this.numPoints; ++i) {
                                  totalNeighbors = shape.pixelsPerPoint + neighborCounter[i];
                                  offsets[i * 3 + 0] = (pointOffset) % (this.numNeighPerRow);
                                  offsets[i * 3 + 1] = Math.floor((pointOffset) / (this.numNeighPerRow));
                                  offsets[i * 3 + 2] = totalNeighbors;
                                  pointOffset += totalNeighbors;
                              }
                              this.log('Offsets', offsets);
                              this.probOffsetTexture = createAndConfigureTexture(this.gpgpu.gl, this.pointsPerRow, this.numRows, 3, offsets);
                          }
                          {
                              probabilities = new Float32Array(this.numNeighPerRow * this.numNeighPerRow);
                              neighIds = new Float32Array(this.numNeighPerRow * this.numNeighPerRow);
                              assignedNeighborCounter = new Uint32Array(this.numPoints);
                              for (i = 0; i < this.numPoints; ++i) {
                                  for (n = 0; n < shape.pixelsPerPoint; ++n) {
                                      linearId = i * shape.pixelsPerPoint + n;
                                      pointId = knnIndicesData[linearId];
                                      probability = gaussianDistributionsData[linearId];
                                      symMatrixDirectId = neighborLinearOffset[i] + n;
                                      symMatrixIndirectId = neighborLinearOffset[pointId] +
                                          shape.pixelsPerPoint +
                                          assignedNeighborCounter[pointId];
                                      probabilities[symMatrixDirectId] = probability;
                                      probabilities[symMatrixIndirectId] = probability;
                                      neighIds[symMatrixDirectId] = pointId;
                                      neighIds[symMatrixIndirectId] = i;
                                      ++assignedNeighborCounter[pointId];
                                  }
                              }
                              this.log('Probabilities', probabilities);
                              this.log('Neighbors', neighIds);
                              this.probTexture = createAndConfigureTexture(this.gpgpu.gl, this.numNeighPerRow, this.numNeighPerRow, 1, probabilities);
                              this.probNeighIdTexture = createAndConfigureTexture(this.gpgpu.gl, this.numNeighPerRow, this.numNeighPerRow, 1, neighIds);
                          }
                          gaussianDistributions.dispose();
                          knnIndices.dispose();
                          this.log('...done!');
                          return [2];
                  }
              });
          });
      };
      TSNEOptimizer.prototype.initializedNeighborhoods = function () {
          return this.probNeighIdTexture != null;
      };
      TSNEOptimizer.prototype.updateExaggeration = function () {
          if (this._exaggeration !== undefined) {
              this._exaggeration.dispose();
          }
          if (typeof this.rawExaggeration === 'number') {
              this._exaggeration = tf.scalar(this.rawExaggeration);
              return;
          }
          if (this._iteration <= this.rawExaggeration[0].iteration) {
              this._exaggeration = tf.scalar(this.rawExaggeration[0].value);
              return;
          }
          if (this._iteration >=
              this.rawExaggeration[this.rawExaggeration.length - 1].iteration) {
              this._exaggeration = tf.scalar(this.rawExaggeration[this.rawExaggeration.length - 1].value);
              return;
          }
          var i = 0;
          while (i < this.rawExaggeration.length &&
              this._iteration < this.rawExaggeration[i].iteration) {
              ++i;
          }
          var it0 = this.rawExaggeration[i].iteration;
          var it1 = this.rawExaggeration[i + 1].iteration;
          var v0 = this.rawExaggeration[i].value;
          var v1 = this.rawExaggeration[i + 1].value;
          var f = (it1 - this._iteration) / (it1 - it0);
          var v = v0 * f + v1 * (1 - f);
          this._exaggeration = tf.scalar(v);
      };
      TSNEOptimizer.prototype.iterate = function () {
          return __awaiter(this, void 0, void 0, function () {
              var _this = this;
              var normQ, _a, _b;
              return __generator(this, function (_c) {
                  switch (_c.label) {
                      case 0:
                          if (!this.initializedNeighborhoods()) {
                              throw new Error('No neighborhoods defined. You may want to call\
                    initializeNeighbors or initializeNeighborsFromKNNGraph');
                          }
                          this.updateSplatTextureDiameter();
                          this.updateExaggeration();
                          _b = tf.tidy(function () {
                              _this.splatPoints();
                              var interpQ = tf.zeros([_this.numRows, _this.pointsPerRow]);
                              var interpXY = tf.zeros([_this.numRows, _this.pointsPerRow * 2]);
                              _this.computeInterpolatedQ(interpQ);
                              _this.computeInterpolatedXY(interpXY);
                              var normQ = interpQ.sum();
                              var repulsiveForces = interpXY.div(normQ);
                              var attractiveForces = tf.zeros([_this.numRows, _this.pointsPerRow * 2]);
                              _this.computeAttractiveForces(attractiveForces);
                              var gradientIter = attractiveForces.mul(_this._exaggeration).sub(repulsiveForces);
                              var gradient = _this.gradient.mul(_this._momentum).sub(gradientIter);
                              _this.gradient.dispose();
                              return [gradient, normQ];
                          }), this.gradient = _b[0], normQ = _b[1];
                          _a = this;
                          return [4, normQ.data()];
                      case 1:
                          _a._normQ = (_c.sent())[0];
                          normQ.dispose();
                          this.embedding = tf.tidy(function () {
                              var embedding = _this.embedding.add(_this.gradient);
                              _this.embedding.dispose();
                              return embedding;
                          });
                          this.computeBoundaries();
                          ++this._iteration;
                          return [2];
                  }
              });
          });
      };
      TSNEOptimizer.prototype.log = function (str, obj) {
          if (this.verbose) {
              if (obj != null) {
                  console.log(str + ": \t" + obj);
              }
              else {
                  console.log(str);
              }
          }
      };
      TSNEOptimizer.prototype.initializeRepulsiveForceTextures = function () {
          this._splatTexture = createAndConfigureInterpolatedTexture(this.gpgpu.gl, this.splatTextureDiameter, this.splatTextureDiameter, 4, null);
          this.kernelSupport = 2.5;
          var kernel = new Float32Array(this.kernelTextureDiameter * this.kernelTextureDiameter * 4);
          var kernelRadius = Math.floor(this.kernelTextureDiameter / 2);
          var j = 0;
          var i = 0;
          for (j = 0; j < this.kernelTextureDiameter; ++j) {
              for (i = 0; i < this.kernelTextureDiameter; ++i) {
                  var x = (i - kernelRadius) / kernelRadius * this.kernelSupport;
                  var y = (j - kernelRadius) / kernelRadius * this.kernelSupport;
                  var euclSquared = x * x + y * y;
                  var tStudent = 1. / (1. + euclSquared);
                  var id = (j * this.kernelTextureDiameter + i) * 4;
                  kernel[id + 0] = tStudent;
                  kernel[id + 1] = tStudent * tStudent * x;
                  kernel[id + 2] = tStudent * tStudent * y;
                  kernel[id + 3] = 1;
              }
          }
          this.kernelTexture = createAndConfigureInterpolatedTexture(this.gpgpu.gl, this.kernelTextureDiameter, this.kernelTextureDiameter, 4, kernel);
      };
      TSNEOptimizer.prototype.initilizeCustomWebGLPrograms = function () {
          this.log('\tCreating custom programs...');
          this.embeddingInitializationProgram =
              createEmbeddingInitializationProgram(this.gpgpu);
          this.embeddingSplatterProgram =
              createEmbeddingSplatterProgram(this.gpgpu);
          var splatVertexId = new Float32Array(this.numPoints * 6);
          {
              var i = 0;
              var id = 0;
              for (i = 0; i < this.numPoints; ++i) {
                  id = i * 6;
                  splatVertexId[id + 0] = 0 + i * 4;
                  splatVertexId[id + 1] = 1 + i * 4;
                  splatVertexId[id + 2] = 2 + i * 4;
                  splatVertexId[id + 3] = 0 + i * 4;
                  splatVertexId[id + 4] = 2 + i * 4;
                  splatVertexId[id + 5] = 3 + i * 4;
              }
          }
          this.splatVertexIdBuffer = tf.webgl.webgl_util.createStaticVertexBuffer(this.gpgpu.gl, splatVertexId);
          this.qInterpolatorProgram =
              createQInterpolatorProgram(this.gpgpu);
          this.xyInterpolatorProgram =
              createXYInterpolatorProgram(this.gpgpu);
          this.attractiveForcesProgram =
              createAttractiveForcesComputationProgram(this.gpgpu);
          this.distributionParameterssComputationProgram =
              createDistributionParametersComputationProgram(this.gpgpu);
          this.gaussiaDistributionsFromDistancesProgram =
              createGaussiaDistributionsFromDistancesProgram(this.gpgpu);
      };
      TSNEOptimizer.prototype.computeBoundaries = function () {
          return __awaiter(this, void 0, void 0, function () {
              var _this = this;
              var _a, min, max, _b, _c, offsetX, _d, _e, offsetY;
              return __generator(this, function (_f) {
                  switch (_f.label) {
                      case 0:
                          _a = tf.tidy(function () {
                              var embedding2D = _this.embedding.reshape([_this.numRows * _this.pointsPerRow, 2])
                                  .slice([0, 0], [_this.numPoints, 2]);
                              var min = embedding2D.min(0);
                              var max = embedding2D.max(0);
                              return [min, max];
                          }), min = _a[0], max = _a[1];
                          _b = this;
                          return [4, min.data()];
                      case 1:
                          _b._minX = (_f.sent())[0];
                          _c = this;
                          return [4, max.data()];
                      case 2:
                          _c._maxX = (_f.sent())[0];
                          offsetX = (this._maxX - this._minX) * 0.05;
                          this._minX -= offsetX;
                          this._maxX += offsetX;
                          _d = this;
                          return [4, min.data()];
                      case 3:
                          _d._minY = (_f.sent())[1];
                          _e = this;
                          return [4, max.data()];
                      case 4:
                          _e._maxY = (_f.sent())[1];
                          offsetY = (this._maxY - this._minY) * 0.05;
                          this._minY -= offsetY;
                          this._maxY += offsetY;
                          min.dispose();
                          max.dispose();
                          return [2];
                  }
              });
          });
      };
      TSNEOptimizer.prototype.updateSplatTextureDiameter = function () {
          var maxSpace = Math.max(this._maxX - this._minX, this._maxY - this._minY);
          var spacePerPixel = 0.35;
          var textureDiameter = Math.ceil(Math.max(maxSpace / spacePerPixel, 5));
          var percChange = Math.abs(this.splatTextureDiameter - textureDiameter) /
              this.splatTextureDiameter;
          if (percChange >= 0.2) {
              this.log('Updating splat-texture diameter', textureDiameter);
              this.gpgpu.gl.deleteTexture(this._splatTexture);
              this.splatTextureDiameter = textureDiameter;
              this._splatTexture = createAndConfigureInterpolatedTexture(this.gpgpu.gl, this.splatTextureDiameter, this.splatTextureDiameter, 4, null);
          }
      };
      TSNEOptimizer.prototype.initializeEmbeddingPositions = function (embedding, random) {
          executeEmbeddingInitializationProgram(this.gpgpu, this.embeddingInitializationProgram, this.backend.getTexture(random.dataId), this.numPoints, this.pointsPerRow, this.numRows, this.backend.getTexture(embedding.dataId));
      };
      TSNEOptimizer.prototype.splatPoints = function () {
          executeEmbeddingSplatterProgram(this.gpgpu, this.embeddingSplatterProgram, this._splatTexture, this.backend.getTexture(this.embedding.dataId), this.kernelTexture, this.splatTextureDiameter, this.numPoints, this._minX, this._minY, this._maxX, this._maxY, this.kernelSupport, this.pointsPerRow, this.numRows, this.splatVertexIdBuffer);
      };
      TSNEOptimizer.prototype.computeInterpolatedQ = function (interpolatedQ) {
          executeQInterpolatorProgram(this.gpgpu, this.qInterpolatorProgram, this._splatTexture, this.backend.getTexture(this.embedding.dataId), this.numPoints, this._minX, this._minY, this._maxX, this._maxY, this.pointsPerRow, this.numRows, this.backend.getTexture(interpolatedQ.dataId));
      };
      TSNEOptimizer.prototype.computeInterpolatedXY = function (interpolatedXY) {
          executeXYInterpolatorProgram(this.gpgpu, this.xyInterpolatorProgram, this._splatTexture, this.backend.getTexture(this.embedding.dataId), this.backend.getTexture(interpolatedXY.dataId), this.numPoints, this._minX, this._minY, this._maxX, this._maxY, this.pointsPerRow, this.numRows, this._eta);
      };
      TSNEOptimizer.prototype.computeAttractiveForces = function (attractiveForces) {
          executeAttractiveForcesComputationProgram(this.gpgpu, this.attractiveForcesProgram, this.backend.getTexture(this.embedding.dataId), this.probOffsetTexture, this.probNeighIdTexture, this.probTexture, this.numPoints, this.numNeighPerRow, this.pointsPerRow, this.numRows, this._eta, this.backend.getTexture(attractiveForces.dataId));
      };
      TSNEOptimizer.prototype.computeDistributionParameters = function (distributionParameters, shape, perplexity, knnGraph) {
          executeDistributionParametersComputationProgram(this.gpgpu, this.distributionParameterssComputationProgram, knnGraph, shape.numPoints, shape.pixelsPerPoint, shape.pointsPerRow, shape.numRows, perplexity, distributionParameters);
      };
      TSNEOptimizer.prototype.computeGaussianDistributions = function (distributions, distributionParameters, shape, knnGraph) {
          executeGaussiaDistributionsFromDistancesProgram(this.gpgpu, this.gaussiaDistributionsFromDistancesProgram, knnGraph, distributionParameters, shape.numPoints, shape.pixelsPerPoint, shape.pointsPerRow, shape.numRows, this.backend.getTexture(distributions.dataId));
      };
      return TSNEOptimizer;
  }());

  function tsne(data, config) {
      return new TSNE(data, config);
  }
  var TSNE = (function () {
      function TSNE(data, config) {
          this.initialized = false;
          this.probabilitiesInitialized = false;
          this.data = data;
          this.config = config;
          var inputShape = this.data.shape;
          this.numPoints = inputShape[0];
          this.numDimensions = inputShape[1];
          if (inputShape.length !== 2) {
              throw Error('computeTSNE: input tensor must be 2-dimensional');
          }
          console.log(this.knnMode);
      }
      TSNE.prototype.initialize = function () {
          return __awaiter(this, void 0, void 0, function () {
              var perplexity, exaggeration, exaggerationIter, exaggerationDecayIter, momentum, _a, exaggerationPolyline;
              return __generator(this, function (_b) {
                  switch (_b.label) {
                      case 0:
                          perplexity = 30;
                          exaggeration = 4;
                          exaggerationIter = 300;
                          exaggerationDecayIter = 200;
                          momentum = 0.8;
                          this.verbose = false;
                          this.knnMode = 'auto';
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
                          if (perplexity > 42) {
                              throw Error('computeTSNE: perplexity cannot be greater than 42');
                          }
                          this.numNeighbors = Math.floor((perplexity * 3) / 4) * 4;
                          _a = this;
                          return [4, tensorToDataTexture(this.data)];
                      case 1:
                          _a.packedData = _b.sent();
                          if (this.verbose === true) {
                              console.log("Number of points " + this.numPoints);
                              console.log("Number of dimensions " + this.numDimensions);
                              console.log("Number of neighbors " + this.numNeighbors);
                          }
                          this.knnEstimator = new KNNEstimator(this.packedData.texture, this.packedData.shape, this.numPoints, this.numDimensions, this.numNeighbors, false);
                          this.optimizer = new TSNEOptimizer(this.numPoints, false);
                          exaggerationPolyline = [
                              { iteration: exaggerationIter, value: exaggeration },
                              { iteration: exaggerationIter + exaggerationDecayIter, value: 1 }
                          ];
                          if (this.verbose === true) {
                              console.log("Exaggerating for " + exaggerationPolyline[0].iteration + " " +
                                  ("iterations with a value of " + exaggerationPolyline[0].value + ". ") +
                                  ("Exaggeration is removed after " + exaggerationPolyline[1].iteration + "."));
                          }
                          this.optimizer.exaggeration = exaggerationPolyline;
                          this.optimizer.momentum = momentum;
                          return [2];
                  }
              });
          });
      };
      TSNE.prototype.compute = function (iterations) {
          return __awaiter(this, void 0, void 0, function () {
              var knnIter;
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          knnIter = this.knnIterations();
                          if (this.verbose) {
                              console.log("Number of KNN iterations:\t" + knnIter);
                              console.log('Computing the KNN...');
                          }
                          return [4, this.iterateKnn(knnIter)];
                      case 1:
                          _a.sent();
                          if (this.verbose) {
                              console.log('Computing the tSNE embedding...');
                          }
                          return [4, this.iterate(iterations)];
                      case 2:
                          _a.sent();
                          if (this.verbose) {
                              console.log('Done!');
                          }
                          return [2];
                  }
              });
          });
      };
      TSNE.prototype.iterateKnn = function (iterations) {
          return __awaiter(this, void 0, void 0, function () {
              var iter;
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          if (!(this.initialized === false)) return [3, 2];
                          return [4, this.initialize()];
                      case 1:
                          _a.sent();
                          _a.label = 2;
                      case 2:
                          this.probabilitiesInitialized = false;
                          for (iter = 0; iter < iterations; ++iter) {
                              this.knnEstimator.iterateBruteForce();
                              if ((this.knnEstimator.iteration % 100) === 0 && this.verbose === true) {
                                  console.log("Iteration KNN:\t" + this.knnEstimator.iteration);
                              }
                          }
                          return [2, true];
                  }
              });
          });
      };
      TSNE.prototype.iterate = function (iterations) {
          return __awaiter(this, void 0, void 0, function () {
              var iter;
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          if (!(this.probabilitiesInitialized === false)) return [3, 2];
                          return [4, this.initializeProbabilities()];
                      case 1:
                          _a.sent();
                          _a.label = 2;
                      case 2:
                          iter = 0;
                          _a.label = 3;
                      case 3:
                          if (!(iter < iterations)) return [3, 6];
                          return [4, this.optimizer.iterate()];
                      case 4:
                          _a.sent();
                          if ((this.optimizer.iteration % 100) === 0 && this.verbose === true) {
                              console.log("Iteration tSNE:\t" + this.optimizer.iteration);
                          }
                          _a.label = 5;
                      case 5:
                          ++iter;
                          return [3, 3];
                      case 6: return [2];
                  }
              });
          });
      };
      TSNE.prototype.knnIterations = function () {
          return Math.ceil(this.numPoints / 20);
      };
      TSNE.prototype.coordinates = function () {
          return this.optimizer.embedding2D;
      };
      TSNE.prototype.knnDistance = function () {
          return 0;
      };
      TSNE.prototype.initializeProbabilities = function () {
          return __awaiter(this, void 0, void 0, function () {
              return __generator(this, function (_a) {
                  switch (_a.label) {
                      case 0:
                          if (this.verbose === true) {
                              console.log("Initializing probabilities");
                          }
                          return [4, this.optimizer.initializeNeighborsFromKNNTexture(this.knnEstimator.knnShape, this.knnEstimator.knn())];
                      case 1:
                          _a.sent();
                          this.probabilitiesInitialized = true;
                          return [2];
                  }
              });
          });
      };
      return TSNE;
  }());

  exports.EmbeddingDrawer = EmbeddingDrawer;
  exports.KNNEstimator = KNNEstimator;
  exports.tsne = tsne;
  exports.TSNEOptimizer = TSNEOptimizer;
  exports.gl_util = gl_util;
  exports.dataset_util = dataset_util;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
