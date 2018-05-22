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

// TODO revise all these programs ( a bit of a hack right now)
// Directly draw a texture on screen
export function createTextureDrawerProgram(gpgpu: tfc.webgl.GPGPUContext):
    WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D texture;
    uniform float width;
    uniform float height;

    void main() {
      vec2 texture_coordinates = gl_FragCoord.xy/vec2(width,height);
      texture_coordinates.y = 1. - texture_coordinates.y;
      vec4 texture_color = texture2D(texture,texture_coordinates);
      gl_FragColor = texture_color;
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeTextureDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext, program: WebGLProgram, texture: WebGLTexture,
    width: number, height: number, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, height, width);
  } else {
    tfc.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const textureLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'texture');
  gpgpu.setInputMatrixTexture(texture, textureLoc, 0);

  const widthLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'width');
  gl.uniform1f(widthLoc, width);

  const heightLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'height');
  gl.uniform1f(heightLoc, height);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

// Draw a texture on screen by averaging the 4 channels
export function createAvgChannelTextureDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D texture;
    uniform float width;
    uniform float height;

    void main() {
      vec2 texture_coordinates = gl_FragCoord.xy/vec2(width,height);
      texture_coordinates.y = 1. - texture_coordinates.y;
      vec4 texture_color = texture2D(texture,texture_coordinates);
      float grey_scale = (texture_color.r + texture_color.g +
                         texture_color.b + texture_color.a)/4.;
      gl_FragColor = vec4(vec3(grey_scale),1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeAvgChannelTextureDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext, program: WebGLProgram, texture: WebGLTexture,
    width: number, height: number, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, height, width);
  } else {
    tfc.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const textureLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'texture');
  gpgpu.setInputMatrixTexture(texture, textureLoc, 0);

  const widthLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'width');
  gl.uniform1f(widthLoc, width);

  const heightLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'height');
  gl.uniform1f(heightLoc, height);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createSplatTextureDrawerProgram(gpgpu: tfc.webgl.GPGPUContext):
    WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D splat_tex;
    uniform sampler2D color_scale_tex;
    uniform sampler2D drawn_embedding_tex;

    uniform float width;
    uniform float height;
    uniform float normalization;

    void main() {
      float value = 0.;
      vec2 texture_pos = vec2(-1,-1);

      if(gl_FragCoord.x < width/2. && gl_FragCoord.y >= height/2.) {
        vec2 texture_pos = (gl_FragCoord.xy-vec2(0.,height/2.))
                              / vec2(width/2.,height/2.);
        vec4 values = texture2D(drawn_embedding_tex,texture_pos);
        gl_FragColor = values;
        return;

      }else if(gl_FragCoord.x >= width/2. && gl_FragCoord.y >= height/2.) {
        vec2 texture_pos = (gl_FragCoord.xy-vec2(width/2.,height/2.))
                              / vec2(width/2.,height/2.);
        vec4 values = texture2D(splat_tex,texture_pos)/normalization;
        value = values.x;

      }else if(gl_FragCoord.x < width/2. && gl_FragCoord.y < height/2.) {
        vec2 texture_pos = gl_FragCoord.xy / vec2(width/2.,height/2.);
        vec4 values = texture2D(splat_tex,texture_pos)/normalization*2.;
        value = values.y;

      }else if(gl_FragCoord.x >= width/2. && gl_FragCoord.y < height/2.) {
        vec2 texture_pos = (gl_FragCoord.xy-vec2(width/2.,0))
                              / vec2(width/2.,height/2.);
        vec4 values = texture2D(splat_tex,texture_pos)/normalization*2.;
        value = values.z;
      }


      vec2 color_scale_pos  = vec2(-1.*value+0.5,0.5);
      vec4 color = texture2D(color_scale_tex,color_scale_pos)/255.;

      gl_FragColor = vec4(color.xyz,1);
    }
  `;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function executeSplatTextureDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext, program: WebGLProgram,
    splatTex: WebGLTexture, colorScaleTex: WebGLTexture,
    drawnEmbeddingTex: WebGLTexture, normalization: number,
    textureDiameter: number, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(targetTex, textureDiameter, textureDiameter);
  } else {
    tfc.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gpgpu.setProgram(program);

  const splatLocation = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'splat_tex');
  gpgpu.setInputMatrixTexture(splatTex, splatLocation, 0);

  const colorScaleLocation =
      tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'color_scale_tex');
  gpgpu.setInputMatrixTexture(colorScaleTex, colorScaleLocation, 1);

  const drawnEmbeddingLoc =
      tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'drawn_embedding_tex');
  gpgpu.setInputMatrixTexture(drawnEmbeddingTex, drawnEmbeddingLoc, 2);

  const widthLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'width');

  gl.uniform1f(widthLoc, textureDiameter);

  const heightLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'height');
  gl.uniform1f(heightLoc, textureDiameter);

  const normLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'normalization');
  gl.uniform1f(normLoc, normalization);

  gpgpu.executeProgram();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createSimpleEmbeddingDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext): WebGLProgram {
  const vertexShaderSource = `
    precision highp float;
    attribute float vertex_id;

    uniform sampler2D embedding_tex;
    uniform vec2 minV;
    uniform vec2 maxV;
    uniform float pnts_per_row;
    uniform float num_rows;

    void main() {
      int row    = int(vertex_id/pnts_per_row);
      int column = int(mod(vertex_id,pnts_per_row));
      float width = (pnts_per_row*2.0);
      float row_tex = (float(row)+0.5)/num_rows;
      vec2 tex_coords_x = vec2((float(column)*2.+0.5)/width, row_tex);
      vec2 tex_coords_y = vec2((float(column)*2.+1.+0.5)/width, row_tex);

      float x_pnt = texture2D(embedding_tex,tex_coords_x).r;
      float y_pnt = texture2D(embedding_tex,tex_coords_y).r;
      vec2 vertex_coords = vec2(x_pnt,y_pnt);

      vertex_coords = (vertex_coords-minV)/(maxV-minV); //  0:1 space
      vertex_coords = vertex_coords*2.0 - 1.0;          // -1:1 space

      gl_Position = vec4(vertex_coords,0,1);
      gl_PointSize = 4.;
    }
  `;
  const fragmentShaderSource = `
    precision highp float;
    uniform float alpha;

    void main() {
      float r = 0.0, delta = 0.0;
      vec2 cxy = 2.0 * gl_PointCoord - 1.0;
      r = dot(cxy, cxy);
      if (r > 1.0) {
          discard;
      }
      gl_FragColor = vec4(0,0.6,1,alpha);
    }
  `;
  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, fragmentShaderSource);
}

export function executeSimpleEmbeddingDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext, program: WebGLProgram,
    embeddingTex: WebGLTexture, numPoints: number, minX: number, minY: number,
    maxX: number, maxY: number, pntsPerRow: number, numRows: number,
    pointIdBuffer: WebGLBuffer, alpha: number, targetTexDiameter: number,
    targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  const oldProgram: WebGLProgram = gpgpu.program;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, targetTexDiameter, targetTexDiameter);
  } else {
    tfc.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gpgpu.setProgram(program);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  tfc.webgl.webgl_util.callAndCheck(
      gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer));
  tfc.webgl.webgl_util.bindVertexBufferToProgramAttribute(
      gl, program, 'vertex_id', pointIdBuffer, 1, 0, 0);

  const embeddingLocation =
      tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);

  const pntsPerRowLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'pnts_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numRowsLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const alphaLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'alpha');
  gl.uniform1f(alphaLoc, alpha);

  // TODO improve
  if (maxX - minX > maxY - minY) {
    maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
    minY = (maxY + minY) / 2 - (maxX - minX) / 2;
  } else {
    maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
    minX = (maxX + minX) / 2 - (maxY - minY) / 2;
  }

  const minLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'minV');
  gl.uniform2f(minLoc, minX, minY);

  const maxLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'maxV');
  gl.uniform2f(maxLoc, maxX, maxY);

  tfc.webgl.webgl_util.callAndCheck(
      gl, () => gl.drawArrays(gl.POINTS, 0, numPoints));

  gl.disable(gl.BLEND);

  // Restore the old program and its vertex buffers
  // TOCHECK if it can be improved
  if (oldProgram != null) {
    gpgpu.setProgram(oldProgram);
    tfc.webgl.gpgpu_util.bindVertexProgramAttributeStreams(
        gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createColoredEmbeddingDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext): WebGLProgram {
  const vertexShaderSource = `
    precision highp float;
    attribute float vertex_id;

    uniform sampler2D embedding_tex;
    uniform sampler2D color_tex;
    uniform vec2 minV;
    uniform vec2 maxV;
    uniform float pnts_per_row;
    uniform float num_rows;

    varying vec4 color;

    void main() {
      int row    = int(vertex_id/pnts_per_row);
      int column = int(mod(vertex_id,pnts_per_row));
      float width = (pnts_per_row*2.0);
      float row_tex = (float(row)+0.5)/num_rows;
      vec2 tex_coords_x = vec2((float(column)*2.+0.5)/width, row_tex);
      vec2 tex_coords_y = vec2((float(column)*2.+1.+0.5)/width, row_tex);

      vec2 color_coords = vec2((float(column)+0.5)/pnts_per_row, row_tex);
      color = texture2D(color_tex,color_coords);

      float x_pnt = texture2D(embedding_tex,tex_coords_x).r;
      float y_pnt = texture2D(embedding_tex,tex_coords_y).r;
      vec2 vertex_coords = vec2(x_pnt,y_pnt);

      vertex_coords = (vertex_coords-minV)/(maxV-minV); //  0:1 space
      vertex_coords = vertex_coords*2.0 - 1.0;          // -1:1 space

      gl_Position = vec4(vertex_coords,0,1);
      gl_PointSize = 4.;
    }
  `;
  const fragmentShaderSource = `
    precision highp float;
    uniform float alpha;
    varying vec4 color;

    void main() {
      //vec4 color = vec4(0.1,0.4,0.9,alpha);
      float r = 0.0, delta = 0.0;
      vec2 cxy = 2.0 * gl_PointCoord - 1.0;
      r = dot(cxy, cxy);
      if (r > 1.0) {
          discard;
      }
      gl_FragColor = color;
      gl_FragColor.a = alpha;
    }
  `;
  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, fragmentShaderSource);
}

export function executeColoredEmbeddingDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext, program: WebGLProgram,
    embeddingTex: WebGLTexture, numPoints: number, minX: number, minY: number,
    maxX: number, maxY: number, pntsPerRow: number, numRows: number,
    pointIdBuffer: WebGLBuffer, alpha: number, targetTexDiameter: number,
    colorsTex: WebGLTexture, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  const oldProgram: WebGLProgram = gpgpu.program;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, targetTexDiameter, targetTexDiameter);
  } else {
    tfc.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gpgpu.setProgram(program);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  tfc.webgl.webgl_util.callAndCheck(
      gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer));
  tfc.webgl.webgl_util.bindVertexBufferToProgramAttribute(
      gl, program, 'vertex_id', pointIdBuffer, 1, 0, 0);

  const embeddingLocation =
      tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);

  const colorLocation = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'color_tex');
  gpgpu.setInputMatrixTexture(colorsTex, colorLocation, 1);

  const pntsPerRowLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'pnts_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numRowsLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const alphaLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'alpha');
  gl.uniform1f(alphaLoc, alpha);

  // TODO improve
  if (maxX - minX > maxY - minY) {
    maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
    minY = (maxY + minY) / 2 - (maxX - minX) / 2;
  } else {
    maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
    minX = (maxX + minX) / 2 - (maxY - minY) / 2;
  }

  const minLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'minV');
  gl.uniform2f(minLoc, minX, minY);

  const maxLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'maxV');
  gl.uniform2f(maxLoc, maxX, maxY);

  tfc.webgl.webgl_util.callAndCheck(
      gl, () => gl.drawArrays(gl.POINTS, 0, numPoints));

  gl.disable(gl.BLEND);

  // Restore the old program and its vertex buffers
  // TOCHECK if it can be improved
  if (oldProgram != null) {
    gpgpu.setProgram(oldProgram);
    tfc.webgl.gpgpu_util.bindVertexProgramAttributeStreams(
        gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

export function createTexturedPointsDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext): WebGLProgram {
  const vertexShaderSource = `#version 300 es
    precision highp float;
    in float vertex_id;

    uniform sampler2D embedding_tex;
    uniform sampler2D color_tex;

    uniform vec2 minV;
    uniform vec2 maxV;
    uniform float pnts_per_row;
    uniform float num_rows;

    out float image_id;
    out vec4 label_color;


    void main() {
      int row    = int(vertex_id/pnts_per_row);
      int column = int(mod(vertex_id,pnts_per_row));

      float width = (pnts_per_row*2.0);
      float row_tex = (float(row)+0.5)/num_rows;
      vec2 tex_coords_x = vec2((float(column)*2.+0.5)/width, row_tex);
      vec2 tex_coords_y = vec2((float(column)*2.+1.+0.5)/width, row_tex);

      vec2 color_coords = vec2((float(column)+0.5)/pnts_per_row, row_tex);
      label_color = texture(color_tex,color_coords);

      float x_pnt = texture(embedding_tex,tex_coords_x).r;
      float y_pnt = texture(embedding_tex,tex_coords_y).r;
      vec2 vertex_coords = vec2(x_pnt,y_pnt);

      vertex_coords = (vertex_coords-minV)/(maxV-minV); //  0:1 space
      vertex_coords = vertex_coords*2.0 - 1.0;          // -1:1 space

      gl_Position = vec4(vertex_coords,0,1);
      gl_PointSize = 14.;
      image_id = vertex_id;
    }
  `;
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    uniform float alpha;
    uniform float pnts_per_row;
    uniform float num_rows;
    uniform float point_texture_diameter;

    in float image_id;
    in vec4 label_color;

    out vec4 fragment_color;

    uniform sampler2D point_tex;

    //Random function developed by Inigo Quilez
    //https://www.shadertoy.com/view/llGSzw
    float hash1( uint n ) {
        // integer hash copied from Hugo Elias
    	  n = (n << 13U) ^ n;
        n = n * (n * n * 15731U + 789221U) + 1376312589U;
        return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);
    }

    void main() {
      vec2 cxy = gl_PointCoord*point_texture_diameter;

      float random = hash1(uint(image_id));

      int row    = int(image_id/250.);
      int col = int(mod(image_id,250.));


      float col_tex = (float(col)*point_texture_diameter+0.5+cxy.x)/3500.;
      float row_tex = (float(row)*point_texture_diameter+0.5+cxy.y)/3360.;

      vec2 tex_coords = vec2(col_tex, row_tex);
      vec4 texture_value = texture(point_tex,tex_coords);
      float average_value = (texture_value.r,texture_value.g,
                              texture_value.b,texture_value.a)/4.;


      fragment_color = label_color;
      fragment_color.a = average_value*1.5;

      float fade_in = 0.05;
      if(random - alpha < fade_in) {
        fragment_color.a *= 1. - (random - alpha)/fade_in;
      }else if(random > alpha) {
        fragment_color.a = 0.;
      }
    }
  `;
  return gl_util.createVertexProgram(
      gpgpu.gl, vertexShaderSource, fragmentShaderSource);
}

export function executeTexturedPointsDrawerProgram(
    gpgpu: tfc.webgl.GPGPUContext, program: WebGLProgram,
    embeddingTex: WebGLTexture, numPoints: number, minX: number, minY: number,
    maxX: number, maxY: number, pntsPerRow: number, numRows: number,
    pointIdBuffer: WebGLBuffer, alpha: number, targetTexDiameter: number,
    pointsTex: WebGLTexture, pointTextureDiameter: number,
    colorsTex: WebGLTexture, targetTex?: WebGLTexture) {
  const gl = gpgpu.gl;
  const oldProgram: WebGLProgram = gpgpu.program;
  if (targetTex != null) {
    gpgpu.setOutputMatrixTexture(
        targetTex, targetTexDiameter, targetTexDiameter);
  } else {
    tfc.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  }

  // TODO provide external color
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gpgpu.setProgram(program);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  tfc.webgl.webgl_util.callAndCheck(
      gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer));
  tfc.webgl.webgl_util.bindVertexBufferToProgramAttribute(
      gl, program, 'vertex_id', pointIdBuffer, 1, 0, 0);

  const embeddingLocation =
      tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'embedding_tex');
  gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);

  const pointsTexLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'point_tex');
  gpgpu.setInputMatrixTexture(pointsTex, pointsTexLoc, 1);

  const colorsTexLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'color_tex');
  gpgpu.setInputMatrixTexture(colorsTex, colorsTexLoc, 2);

  const pntsPerRowLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'pnts_per_row');
  gl.uniform1f(pntsPerRowLoc, pntsPerRow);

  const numRowsLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'num_rows');
  gl.uniform1f(numRowsLoc, numRows);

  const pointTextureDiameterLoc =
      tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
          gl, program, 'point_texture_diameter');
  gl.uniform1f(pointTextureDiameterLoc, pointTextureDiameter);

  const alphaLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'alpha');
  gl.uniform1f(alphaLoc, alpha);

  // TODO improve
  if (maxX - minX > maxY - minY) {
    maxY = (maxY + minY) / 2 + (maxX - minX) / 2;
    minY = (maxY + minY) / 2 - (maxX - minX) / 2;
  } else {
    maxX = (maxX + minX) / 2 + (maxY - minY) / 2;
    minX = (maxX + minX) / 2 - (maxY - minY) / 2;
  }

  const minLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'minV');
  gl.uniform2f(minLoc, minX, minY);

  const maxLoc = tfc.webgl.webgl_util.getProgramUniformLocationOrThrow(
      gl, program, 'maxV');
  gl.uniform2f(maxLoc, maxX, maxY);

  tfc.webgl.webgl_util.callAndCheck(
      gl, () => gl.drawArrays(gl.POINTS, 0, numPoints));

  gl.disable(gl.BLEND);

  // Restore the old program and its vertex buffers
  // TOCHECK if it can be improved
  if (oldProgram != null) {
    gpgpu.setProgram(oldProgram);
    tfc.webgl.gpgpu_util.bindVertexProgramAttributeStreams(
        gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
  }
}
