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
import * as tf_tsne from '../../src/index';
import * as gl_util from "../../src/gl_util";

const pngReader = require('pngjs').PNG;

function sleep(time) {
  return new Promise(resolve => setTimeout(resolve, time));
}

const MNIST_IMAGES_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * Get the program uniforms once only
 * @param gl
 * @param program
 * @returns {{point_tex: WebGLUniformLocation, labels_tex: WebGLUniformLocation, col_array: WebGLUniformLocation}}
 */
function getUniformLocations(gl, program, locationArray) {
  let locationTable = {};
  for (const locName of locationArray) {
    locationTable[locName] = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, locName);
  }
  return locationTable;
}

function executeHitSampleProgram(x, y) {
  if (!offscreen_fbo) {
    return -1;
  }
  const gl = backend.getGPGPUContext().gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, offscreen_fbo);
  // retrieve the id data for hit test.
  const hit_array = new Uint32Array(1);
  const READ_FRAMEBUFFER = 0x8CA8;
  gl.bindFramebuffer(READ_FRAMEBUFFER, offscreen_fbo);
  gl.readBuffer(gl.COLOR_ATTACHMENT0 + 1); // which buffer to read
  gl.readPixels(x, y, 1, 1, gl.RED_INTEGER, gl.UNSIGNED_INT, hit_array); // read it
  if (gl.getError() !== gl.NO_ERROR) {
    console.log('Failed to retrieve hit value');
    return 0;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return hit_array[0]
}

function createTextureToQuadProgram(gl) {
  const vertexShaderSource = `#version 300 es
     in vec4 a_Position;
     in vec2 a_TexCoord;
     out vec2 v_TexCoord;
     void main() {
       gl_Position = a_Position;
       v_TexCoord = a_TexCoord;
     }`;

  const fragmentShaderSource = `#version 300 es
     precision highp float;
     uniform sampler2D u_Sampler;
     in vec2 v_TexCoord;
     out vec4 fragColor;
     void main() {
       fragColor = texture(u_Sampler, v_TexCoord);
     }`;
  return gl_util.createVertexProgram(
    gl, vertexShaderSource, fragmentShaderSource);
}

/**
 * Render an existing texture to a rectangle on the canvas.
 * Any information at that position is overwritten
 * @param gpgpu - the backend gpgpu object
 * @param program - the screen quad drawing program
 * @param uniforms - an object containing the uniform loactions for the program
 * @param texture - the texture to be rendered
 * @param width - the width of canvas rectangle
 * @param height - the height of the canvas rectangle
 * @param left - the left edge of the canvas rectangle
 * @param bottom - the bottom edge of the canvas rectangle
 * @returns {number}
 */
function executeRenderTextureToScreenQuad(
  gpgpu, program, uniforms, texture,
  width, height, left, bottom) {

  const gl = gpgpu.gl;
  //const oldProgram = gpgpu.program;
  gpgpu.setProgram(program);

  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));

  // clear the target with a light grey and blend the render
  gl.enable(gl.SCISSOR_TEST);
  gl.enable(gl.DEPTH_TEST);
  gl.scissor(left, bottom, width, height);
  gl.viewport(left, bottom, width, height);
  gl.clearColor(1.0,1.0,1.0,1.);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // vertex and texture coords in one buffer
  const vertexCoords = new Float32Array([
    -1.0, -1.0,   0.0, 0.0,
     1.0, -1.0,   1.0, 0.0,
     1.0,  1.0,   1.0, 1.0,
    -1.0, -1.0,   0.0, 0.0,
     1.0,  1.0,   1.0, 1.0,
    -1.0,  1.0,   0.0, 1.0
  ]);

  const vertexCoordsBuffer = tf.webgl.webgl_util.createStaticVertexBuffer(
    gl, vertexCoords);

  const FSIZE = vertexCoords.BYTES_PER_ELEMENT;
  // position offset = 0
  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
    gl, program, 'a_Position', vertexCoordsBuffer, 2, FSIZE * 4, 0);
  // tex coord offset = FSIZE * 2
  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
    gl, program, 'a_TexCoord', vertexCoordsBuffer, 2, FSIZE * 4, FSIZE * 2);

  gpgpu.setInputMatrixTexture(texture, uniforms.u_Sampler, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, 6); // Draw the rectangle
  gl.deleteBuffer(vertexCoordsBuffer);
  gl.disable(gl.BLEND);
  gl.disable(gl.SCISSOR_TEST);
  gl.disable(gl.DEPTH_TEST);
}

function createTextureDisplayProgram(gl) {
  var vertexShaderSource = `#version 300 es
  precision highp float;
  in vec4 a_position;
  in vec2 a_texcoord;
  
  out vec2 v_texCoord;
  void main() {
      // Vertex shader output
      gl_Position = a_position;
      v_texCoord = a_texcoord;
  }`;

  const colorTextureShader = `#version 300 es
  precision highp float;
  // Interpolated 0-1 fragment coords from vertex shader
  in vec2 v_texCoord;

  uniform sampler2D u_image;
  uniform int comp_idx;
  uniform float tex_norm;
  uniform float scale_s_field;

  out vec4 fragmentColor;
  
  vec4 scaleRedWhiteBlue(float val) {
      float red = step(-0.05, val); //slight bias in the values
      float blue = 1. - red;
      return mix(
        vec4(1.,1.,1.,1.), 
        vec4(1.,0.,0.,1.) * red + vec4(0.,0.,1.,1.) * blue,
        abs(val)
      ); 
  } 
  
  vec4 scaleRedWhite(float val) {
      return mix(
        vec4(1.,1.,1.,1.), 
        vec4(1.,0.,0.,1.),
        val
      );  
  } 
    
  void main() {
     float fieldVal;
     // Look up a color from the texture.
     switch (comp_idx) {
     case 0:
        fieldVal = clamp(scale_s_field * texture(u_image, v_texCoord).r/(tex_norm/3.), 0., 1.);
        fragmentColor = scaleRedWhite(fieldVal);
        break;
     case 1:
        fieldVal = clamp(texture(u_image, v_texCoord).g/(tex_norm/8.), -1., 1.);
        fragmentColor = scaleRedWhiteBlue(fieldVal);
        break;
     case 2:
        fieldVal = clamp(texture(u_image, v_texCoord).b/(tex_norm/8.), -1., 1.);
        fragmentColor = scaleRedWhiteBlue(fieldVal);
        break;               
     default:
        fragmentColor = vec4(0., 0., 0., 1.);
     }    
  }`;
  return gl_util.createVertexProgram(
    gl, vertexShaderSource, colorTextureShader);
}

/**
 *
 * @param sourceGl - source gl
 * @param texture - the texture to be rendered
 * @param format - pixelData format - eg. gl.RGB, gl.RGBA
 * @param type - pixelData type = eg. gl.FLOAT, gl.UNSIGNED_BYTE
 * @param targetGl - if null the texture data is rendered on the source gl
 *                   otherwise texture data is read and copied to the target.
 */
function executeTextureDisplayProgram(
  gpgpu, program, uniforms, texture, textureNorm, numPoints, index,
  width, height, left, bottom) {

  const gl = gpgpu.gl;
  //const oldProgram = gpgpu.program;
  gpgpu.setProgram(program);

  // this is the backend canvas - clear the display window
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
  gl.enable(gl.SCISSOR_TEST);
  gl.enable(gl.DEPTH_TEST);
  gl.scissor(left, bottom, width, height);
  gl.viewport(left, bottom, width, height);
  gl.clearColor(1., 1., 1., 1.);
  tf.webgl.webgl_util.callAndCheck(
    gl, () =>gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT));

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const quadVertices = new Float32Array([
    -1,-1,
    1,-1,
    1, 1,
    -1,-1,
    1, 1,
    -1, 1]);

  // create and load buffers for the geometry vertices and indices
  const quad_buffer = tf.webgl.webgl_util.createStaticVertexBuffer(gl, quadVertices);
  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
    gl, program, 'a_position', quad_buffer, 2, 0, 0);

  const texCoord = new Float32Array([
    0, 0,
    1, 0,
    1, 1,
    0, 0,
    1, 1,
    0, 1]);
  const texc_buff = tf.webgl.webgl_util.createStaticVertexBuffer(gl, texCoord);
  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
    gl, program, 'a_texcoord', texc_buff, 2, 0, 0);

  gpgpu.setInputMatrixTexture(texture, uniforms.u_image, 0);
  gl.uniform1i(uniforms.comp_idx, index);
  gl.uniform1f(uniforms.tex_norm, textureNorm);
  let scale_s_field = 1;
  if (numPoints < 2000) {
    scale_s_field -= 0.9 * (2000 - numPoints)/2000;
  }
  gl.uniform1f(uniforms.scale_s_field, scale_s_field);
  // Create the buffer object
  // Draw the quad
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  gl.disable(gl.BLEND);
  gl.disable(gl.SCISSOR_TEST);
}

/**
 * Render the points and point ids
 * to two separate render targets
 * @param gpgpu
 * @param numLabels
 * @returns {WebGLProgram}
 */
function createPointsToTexturesProgram(gl) {
  const vertexShaderSource = `#version 300 es
    precision highp float;
    precision highp int;
    in float vertex_id;
    in vec3 label_color;

    uniform sampler2D point_tex;
    uniform float points_per_row;
    uniform vec2 minV;
    uniform vec2 maxV;
    
    out float p_id;
    out vec4 color;
    
    void main() {
     
      int pointNum = int(vertex_id);
      int row = int(floor(vertex_id/points_per_row));
      int col = int(mod(vertex_id, points_per_row));
      float x_pnt = texelFetch(point_tex, ivec2(2 * col + 0, row), 0).r;
      float y_pnt = texelFetch(point_tex, ivec2(2 * col + 1, row), 0).r;

      // point coord from embedding to -1,-1 space
      vec2 point_coords = (vec2(x_pnt, y_pnt) - minV)/(maxV - minV); // 0, 1 space
      point_coords = (point_coords  * 2.0) - 1.0; // -1, -1 space
      
      // color lookup based on point label
      color = vec4(label_color, 1.0);
   
      gl_Position = vec4(point_coords, 0, 1);
      gl_PointSize = 4.;
      p_id = vertex_id + 1.;
   
    }
  `;
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    precision highp int;
    layout(location = 0) out vec4 plot_tex;
    layout(location = 1) out uint id_tex;
    
    in vec4 color;
    in float p_id;

    void main() {
      float r = 0.0, delta = 0.0, alpha = 1.0;
      vec2 cxy = 2.0 * gl_PointCoord - 1.0;
      r = dot(cxy, cxy);
      delta = fwidth(r);
      alpha = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, r);
      plot_tex = vec4(color.rgb, alpha);
      id_tex = uint(p_id);
    }
  `;
  return gl_util.createVertexProgram(
    gl, vertexShaderSource, fragmentShaderSource);
}

/**
 *
 * @param gl
 * @param width
 * @param height
 * @param pixels
 * @returns {WebGLTexture}
 */
function createAndConfigureUint32Texture(gl, width, height, pixels) {
  const texture = tf.webgl.webgl_util.createTexture(gl);
  // begin texture ops
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.bindTexture(gl.TEXTURE_2D, texture));
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE));
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE));
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST));
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST));
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, width, height,
                            0, gl.RED_INTEGER, gl.UNSIGNED_INT, pixels));
  // end texture ops
  tf.webgl.webgl_util.callAndCheck(
    gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
  return texture;
}

let hit_texture = null;
let plot_tex = null;
let offscreen_fbo = null;

function initOffscreenState(gl, width, height) {
  if (offscreen_fbo === null) {
    offscreen_fbo = gl.createFramebuffer();
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, offscreen_fbo);

  if (plot_tex === null) {
    plot_tex = gl_util.createAndConfigureTexture(gl, width, height, 4);
    tf.webgl.webgl_util.callAndCheck(
      gl, () => gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, plot_tex, 0));
  }

  if (hit_texture === null) {
    hit_texture = createAndConfigureUint32Texture(gl, width, height);
    tf.webgl.webgl_util.callAndCheck(
      gl, () => gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + 1, gl.TEXTURE_2D, hit_texture, 0));
  }
}

function clearOffscreenState() {
  hit_texture = null;
  plot_tex = null;
  offscreen_fbo = null;
}

/**
 * Render the embedding to a normaf RGB texture and the
 * @param gpgpu
 * @param program
 * @param uniforms
 * @param pointTex
 * @param width
 * @param height
 * @param left
 * @param bottom
 * @param numPoints
 * @param pointsPerRow
 * @param minX
 * @param minY
 * @param maxX
 * @param maxY
 */
function executeOffscreenPointRender(
  gpgpu, program, uniforms, pointTex,
  width, height, numPoints, pointsPerRow,
  minX, minY, maxX, maxY) {
  const gl = gpgpu.gl;
  // aspect-ratio preserving scaling
  if (maxX - minX > maxY - minY) {
    maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
    minY = (maxY + minY) / 2 - (maxX - minX) / 2;
  }
  else {
    maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
    minX = (maxX + minX) / 2 - (maxY - minY) / 2;
  }

  // set up attributes and uniforms and render
  //const oldProgram = gpgpu.program;
  gpgpu.setProgram(program);

  // create two draw buffer textures
  initOffscreenState(gl, width, height);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  // clear both textures
  let attachBufs = [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT0 + 1];
  tf.webgl.webgl_util.callAndCheck(
    gl, ()=> gl.drawBuffers(attachBufs ));

  tf.webgl.webgl_util.callAndCheck(gl, () =>gl.clearBufferfv(gl.COLOR, 0, [0.0,0.0,0.0,0.0]));
  tf.webgl.webgl_util.callAndCheck(gl, () =>gl.clearBufferuiv(gl.COLOR, 1, [0,0,0,0]));

  gl.viewport(0, 0, width, height);

  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
    gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);

  tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(
    gl, program, 'label_color', labelColorBuffer, 3, 0, 0);

  gpgpu.setInputMatrixTexture(pointTex, uniforms.point_tex, 0);

  gl.uniform1f(uniforms.points_per_row, pointsPerRow);

  gl.uniform2f(uniforms.minV, minX, minY);

  gl.uniform2f(uniforms.maxV, maxX, maxY);
  tf.webgl.webgl_util.callAndCheck(
     gl, () => gl.drawArrays(gl.POINTS, 0, numPoints));
  gl.disable(gl.BLEND);
}

function clearBackground(gl) {
  tf.webgl.webgl_util.bindCanvasToFramebuffer(gl);
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.SCISSOR_TEST);
  gl.clearColor(1.0, 1.0, 1.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
}

let labelColor;
let labelColorBuffer;

let vertexId;
let vertexIdBuffer;
/**
 * Set the fixed vertex buffers for this number of points and label colors
 * @param numPoints
 * @param colors
 */
function initBuffers(numPoints, colors) {

  let gl = backend.getGPGPUContext().gl;

  vertexId = new Float32Array([...Array(numPoints).keys()]);
  vertexIdBuffer =
    tf.webgl.webgl_util.createStaticVertexBuffer(gl, vertexId);

  labelColor = colors;
  labelColorBuffer =
    tf.webgl.webgl_util.createStaticVertexBuffer(gl, labelColor);
}

let pointToTexProgram;
let pointToTexUniforms;

let textureDisplayProgram;
let textureDisplayUniforms;

let textureToQuadProgram;
let textureToQuadUniforms;
/**
 * Set the WebGL environment used for plotting
 *
 * numPoints: number of points in this tsne session
 * gpgpu: the
 */
function initPlotPrograms() {
  let gpgpu = backend.getGPGPUContext();
  let gl = gpgpu.gl;

  pointToTexProgram = createPointsToTexturesProgram(gl);
  const pointToTexUniformList = ['point_tex', 'points_per_row', 'minV', 'maxV'];
  pointToTexUniforms = getUniformLocations(gl, pointToTexProgram, pointToTexUniformList);


  textureDisplayProgram = createTextureDisplayProgram(gl);
  const textureUniformsList = ['u_image', 'comp_idx', 'tex_norm', 'scale_s_field'];
  textureDisplayUniforms = getUniformLocations(gl, textureDisplayProgram, textureUniformsList);

  textureToQuadProgram = createTextureToQuadProgram(gl);
  const genericUniformsList = ['u_Sampler'];
  textureToQuadUniforms = getUniformLocations(gl, textureToQuadProgram, genericUniformsList);

}

// Reduce the MNIST images to newWidth newHeight
// and take the first numImages
function subsampleTensorImages(tensor,
                               oldWidth,
                               oldHeight,
                               newWidth,
                               newHeight,
                               numImages) {
  const subSet = tensor.slice([0,0], [numImages]).
    as4D(numImages, oldHeight, oldWidth, 1);
  return subSet.resizeBilinear([newHeight, newWidth]).
    reshape([numImages, newWidth*newHeight]);
}

let displayObjects = {};

/**
 * Assemble a list of all the elements where the WebGL plots
 * and other data will be placed
 * Plots are placed in divs with a specific width
 * height and position.
 */
function initDisplayObjects() {
  displayObjects = {};
  const textPlots = document.getElementsByClassName('texturePlot');
  for (let element of textPlots) {
    displayObjects[element.id] = {
      element: element,
      uniforms: {},
      data: null
    };
  }
  const scatterPlot = document.getElementById('scatterPlot');
  displayObjects['scatterPlot'] = {
    element: scatterPlot,
    uniforms: {},
    data: null
  };

  displayObjects['knnIter'] = {
    element: document.getElementById('knnIterCount'),
    data: null
  };

  displayObjects['status'] = {
    element: document.getElementById('displayStatus'),
    data: null
  };
  return displayObjects
}

function clearWebglData() {
  const gl = backend.getGPGPUContext().gl;
  if (!gl) {
    return;
  }
  gl.deleteTexture(hit_texture);
  gl.deleteTexture(plot_tex);
  gl.deleteFramebuffer(offscreen_fbo);
}

function initCanvas() {
  initDisplayObjects();
  clearWebglData();
  const digitCanv = document.getElementById('digitCanv');
  const digitCanvCtx = digitCanv.getContext('2d');
  blankCanvas(digitCanvCtx);
  clearBackground(backend.getGPGPUContext().gl);
  return {digitCanvCtx: digitCanvCtx};
}

/**
 * Set a canvas context to white and return the associated
 * imageData and underlying data buffer for further manipulation.
 * @param ctx
 * @returns {{imgData: ImageData, pixArray: Uint8ClampedArray}}
 */
function blankCanvas(ctx) {
  const imgData = ctx.getImageData(0,0,ctx.canvas.width, ctx.canvas.height);
  const pixArray = new Uint8ClampedArray(imgData.data.buffer);
  // zero the buffer for the cumulative plot (black
  const fillArray = new Uint32Array(imgData.data.buffer);
  fillArray.fill(0xFFFFFFFF); //little endian
  ctx.putImageData(imgData, 0, 0);
  return {imgData: imgData, pixArray: pixArray};
}

/**
 * MNIST labels are stored as 65000x10 onehot encoding
 * convert this to label number
 * @param labels
 * @returns {Int32Array}
 */
function oneHotToIndex(labels) {
  return tf.tidy(() => {
    const oneHotTensor = tf.tensor2d(labels, [65000, 10], 'int32');
    const labelPosTensor = tf.tensor1d([0,1,2,3,4,5,6,7,8,9], 'int32');
    const labelsTensor = oneHotTensor.mul(labelPosTensor).sum(1);
    return labelsTensor.dataSync();
  });
}

/**
 * Get a promise that loads the MNIST data.
 * @returns {Promise<*>}
 */
async function loadMnist() {
  //const resp = await fetch('../../images/mnist_images.png');
  const resp = await fetch(MNIST_IMAGES_PATH);
  const imgArray = await resp.arrayBuffer();
  const reader = new pngReader();
  return new Promise ((resolve) => {
    reader.parse(imgArray, (err, png) => {
      // parsed PNG is Uint8 RGBA with range 0-255
      // - convert to RGBA Float32 range 0-1
      const pixels = new Float32Array(png.data.length/4);
      for (let i = 0; i < pixels.length; i++) {
        pixels[i] = png.data[i*4]/255.0;
      }
      resolve(pixels);
    });
  });
}

/**
 * Get a promise that loads the MNIST label data
 * @returns {Promise<ArrayBuffer>}
 */
async function loadMnistLabels() {
  const resp = await fetch(MNIST_LABELS_PATH);
  return resp.arrayBuffer();
}

/**
 * A global to hold the MNIST data
 */
let dataSet;
/**
 * A global to hold the MNIST label data
 */
let labelSet;
let cancel = false;
let enableViz = false;
let backend;
let maxSize = 0;

function clearBackendCanvas() {
  let gl = backend.getGPGPUContext().gl;
  tf.webgl.webgl_util.bindCanvasToFramebuffer(gl);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

let webGlCanvas;
/**
 * Make a page spanning canvas from the backend context.
 */
function initBackend() {
  backend = tf.ENV.findBackend('webgl');
  // inject backend canvas for drawing
  webGlCanvas = backend.getCanvas();
  const bodyList = document.getElementsByTagName('body');
  const body = bodyList[0];
  let offset = 0;
  for (let node of body.childNodes) {
    if (node.id === 'canvasContainer') {
      break;
    }
    offset++;
  }
  body.replaceChild(webGlCanvas, body.childNodes[offset]);
  webGlCanvas.id = "wholePageCanvas";
  webGlCanvas.style = "width:100vw; height:100vh; margin-top: 0 !important; margin-left: 0 !important; position:absolute; top:0; display:block;";
  let gl = backend.getGPGPUContext().gl;
  gl.getExtension('EXT_float_blend');
  maxSize= gl.getParameter(gl.MAX_TEXTURE_SIZE);
  gl.canvas.width = gl.canvas.offsetWidth;
  gl.canvas.height = gl.canvas.offsetHeight;
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

  clearBackendCanvas();
}

/**
 * Resize the canvas if the clientWidth has changed
 * @param gl
 */
function resizeCanvas(gl) {
  // resize the canvas
  const cwidth = gl.canvas.clientWidth;
  const cheight = gl.canvas.clientHeight;
  if (gl.canvas.width != cwidth || gl.canvas.height != cheight) {
    gl.canvas.width = cwidth;
    gl.canvas.height = cheight;
  }
}

function getLimits(element, gl) {
  const rect = element.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top  > gl.canvas.clientHeight ||
    rect.right  < 0 || rect.left > gl.canvas.clientWidth) {
    return [false, -1, -1, -1, -1];  // it's off screen
  }

  const width  = rect.right - rect.left;
  const height = rect.bottom - rect.top;
  const left   = rect.left;
  const bottom = gl.canvas.clientHeight - rect.bottom - 1;
  return [true, width, height, left, bottom];
}

function displayTextures() {
  const gl = backend.getGPGPUContext().gl;
  const textureIds = ['textureR', 'textureG', 'textureB'];
  let count = 0;
  textureIds.forEach((id, idx) => {
    const plotObject = displayObjects[id];
    const data = plotObject.data;
    if (!data) {
      // nothing to render
      return;
    }
    let OK, width, height, left, bottom;
    [OK, width, height, left, bottom] = getLimits(plotObject.element, gl);
    if (!OK) {
      return;
    }
    executeTextureDisplayProgram(
      backend.getGPGPUContext(), textureDisplayProgram, textureDisplayUniforms,
      data.splatTexture, data.normalizeTex, data.numPoints, idx,
      width, height, left, bottom);
    count++;
  });
}

let lastRenderTime = 0;
let lastRenderItern = 0;
/**
 * Display the embedding as a scatter plot.
 *
 * tsneOpt: instance of tf_tsne.tsne containing coordinates to be plotted
 * labelsTensor: 1D tensor containing labels 0 to 9 for each point
 * plotSize: size of (square) plot target
 */
function displayScatterPlot(now) {
  if (!enableViz) {
    return;
  }
  const gpgpu = backend.getGPGPUContext();
  const gl = gpgpu.gl;
  resizeCanvas(gl);

  webGlCanvas.style.transform = `translateY(${window.scrollY}px)`;
  const plotObject = displayObjects['scatterPlot'];
  const data = plotObject.data;
  if (!data) {
    // nothing to render
    return;
  }

  const tsneIterElement = document.getElementById('tsneIterCount');
  tsneIterElement.innerHTML = 'tsne iteration: ' + data.iteration;

  // limit to 5 frames per sec
  if ((now - lastRenderTime)  <  200) {
    return;
  }
  lastRenderTime = now;
  clearBackground(gl);

  let OK, width, height, left, bottom;
  [OK, width, height, left, bottom] = getLimits(plotObject.element, gl);
  if (!OK) {
    return;
  }

  const oldProgram = gpgpu.program;

  if (data.iteration !== lastRenderItern) {
    lastRenderItern = data.iteration;
    // Render the embedding points offscreen along with a hit texture.
    executeOffscreenPointRender(
      backend.getGPGPUContext(), pointToTexProgram, pointToTexUniforms, data.coords,
      width, height, data.numPoints, data.pointsPerRow,
      data.minX, data.minY, data.maxX, data.maxY);
  }

  executeRenderTextureToScreenQuad(
    backend.getGPGPUContext(), textureToQuadProgram, textureToQuadUniforms,
    plot_tex, width, height, left, bottom
  );

  displayTextures();

  if (oldProgram != null) {
    gpgpu.setProgram(oldProgram);
    tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(
      gl, oldProgram, gpgpu.vertexBuffer);
  }
};

/**
 * Array o integer RGB colors and make a Float32Array
 * containing 3 rgb components
 *
 * @param colArray
 */
function colToFloatComp(colArray) {
  const nestedComponents = colArray.map(x =>
    [((x >> 16) & 0xFF)/255, ((x >> 8) & 0xFF)/255, (x & 0xFF)/255]
  );
  // flatten the array of arrays
  return new Float32Array([].concat.apply([], nestedComponents));
}

/**
 * Run tf-tsne on the MNIST and plot the data points
 * in a simple interactive canvas.
 * @returns {Promise<void>}
 */
async function runTsne() {
  cancel = false;
  // The MNIST set is preshuffled just load it.
  const allMnistTensor = tf.tensor(dataSet).reshape([65000, 784]);

  const numberPoints = parseInt(document.getElementById('numPointsSlider').value);

  // subset and downsample the images
  const subTensor = subsampleTensorImages(allMnistTensor,
    28,
    28,
    28,
    28,
    numberPoints);
  allMnistTensor.dispose();

  // match the number of labels to the points subset
  const subLabels = labelSet.slice(0, numberPoints);
  const labelColors = colToFloatComp([0xFF0000,
    0xFF9900,
    0xCCFF00,
    0x33FF00,
    0x00FF66,
    0x00FFFF,
    0x0066FF,
    0x3300FF,
    0xCC00FF,
    0xFF0099]);
  const colors = new Float32Array(numberPoints * 3);
  subLabels.forEach((val, idx) => {
    colors [idx * 3] = labelColors[val * 3];
    colors [idx * 3 + 1] = labelColors[val * 3 + 1];
    colors [idx * 3 + 2] = labelColors[val * 3 + 2];
  });
  initBuffers(numberPoints, colors);

  console.log(`calculating on: ${subTensor.shape}`);

  const perplexity = numberPoints < 240 ? Math.floor(numberPoints/8) : 30;
  const tsneOpt = tf_tsne.tsne(subTensor, {
    perplexity : perplexity,
    verbose : true,
    knnMode : 'auto',
  });

  const maxKnnIters = document.getElementById('kNNSlider').value;
  const knnIterations = Math.min(tsneOpt.knnIterations(), maxKnnIters);
  await runAndDisplayKnn(tsneOpt, knnIterations);
  await runAndDisplayTsne(tsneOpt, 1000, numberPoints);
  cancel = true;
  console.log(`Tsne done`);
  tf.dispose(subTensor);
  tsneOpt.optimizer.dispose();
}

async function runAndDisplayKnn(tsneOpt, nIters)
{
  console.log('started kNN');
  displayObjects['status'].element.innerHTML = '...running kNN';
  await sleep(1);
  for (let iterCount = 0; iterCount < nIters; iterCount++) {
    await tsneOpt.iterateKnn(1);
    displayObjects['knnIter'].element.innerHTML = 'knn iteration: ' + (iterCount + 1);
    if (iterCount === 0) {
      enableViz = true;
    }
    await sleep(1);
  }
  displayObjects['status'].element.innerHTML = '';
}

/**
 * support globals for texture copying
 */
let dstFbo;
let srcFbo;
let splatTexCopy;
let embeddingClone = null;

/**
 * Create the source and destination framebuffer objects once
 * Setc srcFbo and dstFbo
 */
function initTextureCopy() {
  let gl = backend.getGPGPUContext().gl;
  dstFbo = gl.createFramebuffer();
  srcFbo = gl.createFramebuffer();
}

/**
 * Blit copy the source text and return the copy
 * Uses the srcFbo and dstFbo framebuffers created
 * by initTextureCopy
 * @param srcTexture
 * @returns {WebGLTexture}
 */
function makeTextureCopy(srcTexture, width, height) {
  let gl = backend.getGPGPUContext().gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, srcFbo);
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D, srcTexture, 0);

  gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo);
  const dstTexture = gl_util.createAndConfigureTexture(
    gl,
    width,
    height,
    4, null);
  gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D, dstTexture, 0);

  gl.bindFramebuffer ( gl.DRAW_FRAMEBUFFER, dstFbo );
  gl.bindFramebuffer ( gl.READ_FRAMEBUFFER, srcFbo );

  gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height,
    gl.COLOR_BUFFER_BIT, gl.NEAREST);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return dstTexture;
}

/**
 * Wrap tsne in an async iterator to decouple from the display.
 * @param tsneOpt
 * @param nIterations
 * @param numPoints
 * @returns {AsyncIterableIterator<{iteration: number, numPoints: *, coords: WebGLTexture, pointsPerRow: number, minX: number, maxX: number, minY: number, maxY: number, splatTexture: WebGLTexture, splatDiameter}>}
 */
async function* tsneResultGenerator(tsneOpt, nIterations, numPoints) {
  let count = 0;
  displayObjects['status'].element.innerHTML = '...running tSNE';
  while (count < nIterations) {
    await tsneOpt.iterate(1);
    if (count === 0) {
      initTextureCopy();
    }
    // Copy  the splat texture in order to display it asynchronously.
    // It is constantly recreated in the tsne algorithm.
    if (splatTexCopy) {
      const gl = backend.getGPGPUContext().gl;
      gl.deleteTexture(splatTexCopy);
      splatTexCopy = null;
    }
    splatTexCopy = makeTextureCopy(
      tsneOpt.optimizer.splatTexture,
      tsneOpt.optimizer.splatTextureDiameter,
      tsneOpt.optimizer.splatTextureDiameter);
    if (embeddingClone) {
      embeddingClone.dispose();
      embeddingClone = null;
    }
    embeddingClone = tf.clone(tsneOpt.optimizer.embedding);
    lastRenderTime = 0; // force render
    yield {
      iteration: count + 1,
      numPoints: numPoints,
      coords: backend.getTexture(embeddingClone.dataId),
      pointsPerRow: tsneOpt.optimizer.numberOfPointsPerRow,
      minX: tsneOpt.optimizer.minX,
      maxX: tsneOpt.optimizer.maxX,
      minY: tsneOpt.optimizer.minY,
      maxY: tsneOpt.optimizer.maxY,
      splatTexture: splatTexCopy,
      diameter: tsneOpt.optimizer.splatTextureDiameter,
      normalizeTex: Math.sqrt(tsneOpt.optimizer.normalizationQ) * tsneOpt.optimizer.exaggerationAtCurrentIteration
    };
    if (cancel) {
      cancel = false;
      tsneOpt.optimizer.dispose();
      break;
    }
    count++;
  }
  displayObjects['status'].element.innerHTML = '';
}

/**
 * Display iteration results in an animation frame
 */
function displayIterInfo() {
  requestAnimationFrame(displayIterInfo);
  displayScatterPlot();
}

async function runAndDisplayTsne(tsneOpt, nIterations, numPoints)
{
  window.requestAnimationFrame(displayIterInfo);
  console.log('started RAF');
  for await (const iterInfo of tsneResultGenerator(tsneOpt, nIterations, numPoints)) {
    displayObjects['scatterPlot'].data = iterInfo;
    displayObjects['textureR'].data = iterInfo;
    displayObjects['textureG'].data = iterInfo;
    displayObjects['textureB'].data = iterInfo;
  }
}

/**
 * Plot the digit on the canvas
 *
 * @param digitCtx
 * @param digitData
 */
async function digitOnCanvas(digitCtx, digitData) {
  const height = digitCtx.canvas.height;
  const width = digitCtx.canvas.height;
  const dataPix = blankCanvas(digitCtx);
  const imgData = dataPix.imgData;
  const pixArray = dataPix.pixArray;
  // put the digit data in a tensor and resize it
  // to match the canvas
  const imgTensor = tf.tensor4d(digitData, [1, 28, 28, 1]);

  const resizedTensor = imgTensor.resizeNearestNeighbor([height, width]);
  const resizedArray = await resizedTensor.data();
  resizedArray.forEach((val, idx) => {
    const pixOffset = 4 * idx;
    const pixVal = 255 - (255 * val);
    pixArray[pixOffset] = pixVal;
    pixArray[pixOffset + 1] = pixVal;
    pixArray[pixOffset + 2] = pixVal;
  });
  digitCtx.putImageData(imgData, 0, 0);
}

/**
 * Handle the mousemove event to explore the points in the
 * plot canvas.
 * @param plotCanv
 * @param e
 */
function plotExplore(plotCtx, digitCtx, e) {
  const rect = plotCtx.getBoundingClientRect();
  const x  = e.clientX - rect.left;
  const y  = e.clientY - rect.top;
  const id = executeHitSampleProgram(x, 511-y);
  if (id < 1) {
    return;
  }
  const digitData = dataSet.slice((id-1)*784, (id)*784);
  digitOnCanvas(digitCtx, digitData);
}

function restart() {
  cancel = true;
  enableViz = false;
  clearOffscreenState();
  setTimeout(async ()=> {
    initCanvas();
    await runTsne()
  }, 1000)
}

function stop() {
  cancel = true;
}

function updatePoints() {
  const nPoints = parseInt(document.getElementById('numPointsSlider').value);
  document.getElementById('pntSliderVal').innerHTML = 'num MNIST points: ' + nPoints.toString().padStart(6, '\u2002');
}

window.onload = async function() {
  initBackend();
  const contexts = initCanvas();
  const digitCtx = contexts.digitCanvCtx;
  updatePoints();
  initPlotPrograms();
  displayObjects['status'].element.innerHTML = '...downloading MNIST data';
  dataSet = await loadMnist();
  const labelOneHot = new Uint8Array(await loadMnistLabels());
  labelSet = oneHotToIndex(labelOneHot);
  displayObjects['status'].element.innerHTML = '';

  document.getElementById('kNNSlider').oninput = () => {
    document.getElementById('sliderVal').innerHTML = 'max kNN iterations: ' + document.getElementById('kNNSlider').value;
  };

  document.getElementById('numPointsSlider').oninput = updatePoints;
  const plotCtx = document.getElementById('scatterPlot');
  plotCtx.addEventListener('mousemove', plotExplore.bind(null, plotCtx, digitCtx));
  document.getElementById('restartButton').addEventListener('click', restart);
  document.getElementById('stopButton').addEventListener('click', stop);
};
