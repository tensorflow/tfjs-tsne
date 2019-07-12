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
import Konva from 'konva';
const tsnejs = require('./tsne.js');

const pngReader = require('pngjs').PNG;

const imgSize = 512;
const plotDigitIndex = new Int32Array(imgSize*imgSize);
let subTensor;

let stage;
let layer;
let nodes = [];
let tooltipLayer;
let tooltip;

const MNIST_IMAGES_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const plotIndex = new Uint16Array(imgSize*imgSize);
// Colors for digits 0-9
const c0 = 0xFF0000;
const c1 = 0xFF9900;
const c2 = 0xCCFF00;
const c3 = 0x33FF00;
const c4 = 0x00FF66;
const c5 = 0x00FFFF;
const c6 = 0x0066FF;
const c7 = 0x3300FF;
const c8 = 0xCC00FF;
const c9 = 0xFF0099;
const colArray = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9];

plotDigitIndex.fill(-1);
function sleep(time) {
  return new Promise(resolve => setTimeout(resolve, time));
}

function initKonva() {
  stage = new Konva.Stage({
    container: 'konvaContainer',
    width: 512,
    height: 512
  });
  nodes = [];
  tooltipLayer = null;
  tooltip = null;
  layer = new Konva.Layer();
  stage.add(layer);
}

function setupTooltip(digitCtx) {
  tooltipLayer = new Konva.Layer();
  tooltip = new Konva.Label({
    opacity: 0.75,
    visible: false,
    listening: false
  });
  tooltip.add(new Konva.Tag({
    fill: 'black',
    pointerDirection: 'down',
    pointerWidth: 10,
    pointerHeight: 10
  }));
  tooltip.add(new Konva.Text({
    test: '',
    fontFamily: 'Calibri',
    fontSize: 18,
    padding: 5,
    fill: 'white'
  }));
  tooltipLayer.add(tooltip);
  stage.add(tooltipLayer);
  stage.on('mouseover mousemove', function(evt) {
    let node = evt.target;
    if(node) {
      let mousePos = node.getStage().getPointerPosition();
      tooltip.position({
        x: mousePos.x,
        y: mousePos.y - 5
      });
      tooltip.getText().setText(`${node.getId()}`);
      if (node.getId() != undefined) {
        tooltip.show();
        plotExplore(digitCtx, node.__index);
      } else {
        tooltip.hide();
      }
      tooltipLayer.batchDraw();
    }
  });
  stage.on('mouseout', function(evt) {
    tooltip.hide();
    tooltipLayer.draw();
  });
  let scaleBy = 1.05;
  window.addEventListener('wheel', (e) => {
    e.preventDefault();
    let oldScale = stage.scaleX();

    let mousePointTo = {
      x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
      y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
    };

    let newScale = e.deltaY > 0 ? oldScale * scaleBy : oldScale / scaleBy;
    stage.scale({ x: newScale, y: newScale });

    let newPos = {
      x: -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
      y: -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale
    };
    for (let node of nodes) {
      let rad = node.getRadius();
      if (e.deltaY > 0) {
        node.setRadius(rad/scaleBy);
      }
      else {
        node.setRadius(rad * scaleBy);
      }
    }
    stage.position(newPos);
    stage.batchDraw();
  });
}

function plotCoordsKonva(numberPoints,
                         coordData,
                         labelData,
                         digitCtx) {
  // first time create the points

  let maxX = Number.MIN_VALUE;
  let minX = Number.MAX_VALUE;
  let maxY = Number.MIN_VALUE;
  let minY = Number.MAX_VALUE;
  for (let coord of coordData) {
    if (coord[0] > maxX) {
      maxX = coord[0];
    }
    if (coord[0] < minX) {
      minX = coord[0];
    }
    if (coord[1] > maxY) {
      maxY = coord[1];
    }
    if (coord[1] < minY) {
      minY = coord[1];
    }
  }
  const yRange  = (maxY - minY) * 1.5;
  const xRange = (maxX - minX) * 1.5;
  const kWidth = stage.width();
  const kHeight = stage.height();
  if (nodes.length === 0) {
    setupTooltip(digitCtx);
    for (let i = 0; i < numberPoints; i++) {
      const xcoord = Math.round((coordData[i][0]/xRange) * (kWidth - 1) + 256);
      const ycoord = Math.round((coordData[i][1]/yRange) * (kHeight - 1) + 256);
      const colStr =  colArray[labelData[i]].toString(16).padStart(6,'0');
      let node = new Konva.Circle({
        x: xcoord,
        y: ycoord,
        radius: 2.5,
        fill: '#' + colStr,
        id: labelSet[i],
      });
      node.__index = i;
      layer.add(node);
      nodes.push(node);
    }
  }
  else {
    for (let i = 0; i < numberPoints; i++) {
      const xcoord = Math.round((coordData[i][0]/xRange) * (kWidth - 1) + 256);
      const ycoord = Math.round((coordData[i][1]/yRange) * (kHeight - 1) + 256);
      nodes[i].position({x: xcoord, y: ycoord});
    }
  }
  layer.batchDraw();
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

function initCanvas() {
  const digitCanv = document.getElementById('digitCanv');
  // create drawing canvas of required dimensions
  const digitCanvCtx = digitCanv.getContext('2d');
  blankCanvas(digitCanvCtx);
  return {digitCanvCtx: digitCanvCtx};
}

/**
 * Set a canvas context to black and return the associated
 * imageData and underlying data buffer for further manipulation.
 * @param ctx
 * @returns {{imgData: ImageData, pixArray: Uint8ClampedArray}}
 */
function blankCanvas(ctx) {
  const imgData = ctx.getImageData(0,0,ctx.canvas.width, ctx.canvas.height);
  const pixArray = new Uint8ClampedArray(imgData.data.buffer);
  // zero the buffer for the cumulative plot (black
  const fillArray = new Uint32Array(imgData.data.buffer);
  fillArray.fill(0xFF000000); //little endian
  ctx.putImageData(imgData, 0, 0);
  return {imgData: imgData, pixArray: pixArray};
}

/**
 * MNIST labels are stored as 65000x10 onehot encoding
 * convert this to label number
 * @param labels
 * @returns {Uint8Array}
 */
function oneHotToIndex(labels) {
  const res = new Uint8Array(labels.length/10);
  for(let i =0; i<labels.length; i+=10) {
    for (let j = 0; j < 10; j++) {
      if (labels[i+j] === 1) {
        res[i/10] = j;
        break;
      }
    }
  }
  return res;
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
 * Get a promist that loads the MNIST label data
 * @returns {Promise<ArrayBuffer>}
 */
async function loadMnistLabels() {
  //const resp = await fetch('../../images/mnist_labels_uint8.bin');
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
/**
 * Run tf-tsne on the MNIST and plot the data points
 * in a simple interactive canvas.
 * @returns {Promise<void>}
 */
async function runTsne(digitCtx) {
  cancel = false;

  const numberData = document.getElementById('ndigitSlider').value;
  const selTensor = subTensor.slice([0,0], [numberData, 784])
  let subMnistData = await selTensor.transpose().data();

  console.log(`calculating on: ${selTensor.shape}`);

  let opt = {};
  opt.epsilon = 10; // epsilon is learning rate (10 = default)
  opt.perplexity = numberData < 300 ? Math.floor(numberData/10) : 30;; // roughly how many neighbors each point influences (30 = default)
  opt.dim = 2; // dimensionality of the embedding (2 = default)

  let tsne = new tsnejs.tSNE(opt);
  let points = [];
  for (let p = 0; p<numberData; p++) {
    let row = [];
    for (let d = 0; d<784; d++) {
      row.push(subMnistData[p + d*numberData]);
    }
    points.push(row);
  }
  tsne.initDataRaw(points);
  const numIters = 1000;
  const tsneIterElement = document.getElementById('tsneIterCount');
  // get the image data and access the data buffer to overwrite
  for(let i=0; i<numIters; i+=1) {
    tsne.step();
    const coordData = tsne.getSolution();
    plotCoordsKonva(numberData, coordData, labelSet, digitCtx);
    tsneIterElement.innerHTML = 'tsne iteration: ' + (i + 1);
    if (cancel) {
      cancel = false;
      return;
    }
    await sleep(1);
  }
  console.log(`Tsne done`);
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
    const pixVal = 255 * val;
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
function plotExplore(digitCtx, digitIndex) {
  if (digitIndex >= 1) {
    console.log(`digit idx: ${digitIndex}, label: ${labelSet[digitIndex]}`);
    const digitData = dataSet.slice(digitIndex*784, (digitIndex+1)*784);
    digitOnCanvas(digitCtx, digitData);
  }
}

function restart(digitCtx) {
  cancel = true;
  initKonva();
  blankCanvas(digitCtx);
  setTimeout(async ()=> {
    await runTsne(digitCtx)
  }, 1000)

}

function stop() {
  cancel = true;
}

window.onload = async function() {
  initKonva();
  const contexts = initCanvas();
  const digitCtx = contexts.digitCanvCtx;

  dataSet = await loadMnist();
  const labelOneHot = new Uint8Array(await loadMnistLabels());
  labelSet = oneHotToIndex(labelOneHot);
  // The MNIST set is preshuffled
  const allMnistTensor = tf.tensor(dataSet).
  reshape([65000, 784]);
  // subset and downsample the images
  subTensor = subsampleTensorImages(allMnistTensor,
    28,
    28,
    28,
    28,
    65000);


  document.getElementById('ndigitSlider').oninput = () => {
    document.getElementById('numDigits').innerHTML = 'num digits: ' + document.getElementById('ndigitSlider').value;
  }
  document.getElementById('restartButton').addEventListener('click', restart.bind(null, digitCtx));
  document.getElementById('stopButton').addEventListener('click', stop);
  await runTsne(digitCtx);
}
