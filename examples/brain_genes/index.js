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
import Konva from 'konva';

const colors = [
  0xE8C159, 0xE8C459, 0xE8C659, 0xE8C659, 0xE8CA59, 0xE8CB59,
  0xE8CD59, 0xE8CD59, 0xE8CF59, 0xE8D659, 0xE8D659, 0xE8D659,
  0xE8D959, 0xE8D959, 0xFFF161, 0xEFFF61, 0x7F5B33, 0x7F5B33,
  0x7F6233, 0xFFB566, 0xFFBA66, 0xFFBF66, 0xFFC466, 0xFFC966,
  0xFFD366, 0xBF934C, 0xBF934C, 0xE87751, 0xE87751, 0xE88051,
  0xE88051, 0xE88A51, 0xE88A51, 0xE88F51, 0xFF5D62, 0xFF5D62,
  0xFF5D62, 0xFF5D62, 0xFF6C62, 0xFF6C62, 0xFF7462, 0xFF7462,
  0xFF7B62, 0xFF7B62, 0xFF877F, 0xFF877F, 0xFF8A7F, 0xFF8D7F,
  0xFF8D7F, 0xFF8D7F, 0xFF907F, 0xFF907F, 0xFF937F, 0xFF997F,
  0xFF997F, 0xFF60DE, 0xFF60DE, 0xFF60DE, 0xFF60DE, 0xFF60DE,
  0xFF60DE, 0x9C53E8, 0x895BFF, 0x895BFF, 0x895BFF, 0x895BFF,
  0x895BFF, 0xC0A7FF, 0x7FFF89, 0x7FFF89, 0x7FFF8F, 0x92FF5D,
  0xA9E855, 0xA9E855, 0xA9E855, 0xA9E855, 0xA9E855, 0xA9E855,
  0xA9E855, 0xE2FF5D, 0xE2FF5D, 0x349009, 0x349009, 0x349009,
  0x349009, 0x00D5E8, 0x00D5E8, 0x00D5E8, 0x00D5E8, 0x00D5E8,
  0x00D5E8, 0x00D5E8, 0x00D5E8, 0x06CAFF, 0x00E8BB, 0x00FFAA,
  0x054DB2, 0x0E56BB, 0x1961C6, 0x276FD4, 0x276FD4, 0x327ADF,
  0x377FE4, 0x3D85EA, 0xF2F1F0
];

const labelSet = [
  "fro", "GRe", "orIFG", "trIFG", "LOrG", "MOrG", "MFG-s", "MFG-i", "PCLa-i", "PrG-prc", "PrG-sl", "PrG-il", "SFG-m",
  "SFG-l", "LIG", "SIG", "CgGf-s", "CgGf-i", "CgGp-s", "DG", "CA1", "CA2", "CA3", "CA4", "S", "PHG-l", "PHG-cos",
  "Cun-pest", "Cun-str", "LiG-pest", "LiG-str", "OTG-s", "OTG-i", "SOG-s", "AnG-s", "AnG-i", "SMG-s", "SMG-i", "PoG-cs",
  "PoG-sl", "Pcu-s", "Pcu-i", "SPL-s", "SPL-i", "FuG-its", "FuG-cos", "HG", "ITG-its", "ITG-l", "ITG-mts", "MTG-s",
  "MTG-i", "PLP", "STG-l", "STG-i", "ATZ", "BLA", "BMA", "CeA", "COMA", "LA", "GPi", "BCd", "HCd", "TCd", "Acb", "Pu",
  "Cl", "LHM", "PHA", "PrOR", "Sb", "DTA", "ILc", "LGd", "DTLd", "DTLv", "DTM", "ILr", "R", "ZI", "RN", "SNC", "SNR",
  "VTA", "PV-V", "PV-VI", "He-VI", "PV-Crus I", "He-Crus I", "He-Crus II", "PV-VIIB", "He-VIIB", "Dt", "Pn", "PRF",
  "Arc", "Cu", "IO", "GiRt", "LMRt", "RaM", "Sp5", "8Ve", "cc"];

const imgSize = 512;
const plotDigitIndex = new Int32Array(imgSize*imgSize);
let stage;
let layer;
let nodes = [];
let tooltipLayer;
let tooltip;

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
  layer = new Konva.Layer();
  stage.add(layer);
}

function setupTooltip() {
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
      tooltip.getText().setText(node.getId());
      if (node.getId()) {
        tooltip.show();
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
                    coordData) {
  // first time create the points

  const kWidth = stage.width();
  const kHeight = stage.height();
  if (nodes.length === 0) {
    setupTooltip();
    for (let i = 0; i < numberPoints; i++) {
      const xcoord = Math.round(coordData[i*2] * (kWidth - 1));
      const ycoord = Math.round(coordData[i*2 + 1] * (kHeight - 1));
      const colStr = colors[i % 105].toString(16).padStart(6,'0');
      let node = new Konva.Circle({
        x: xcoord,
        y: ycoord,
        radius: 3.5,
        fill: '#' + colStr,
        id: labelSet[i % 105]
      });
      layer.add(node);
      nodes.push(node);
    }
  }
  else {
    for (let i = 0; i < numberPoints; i++) {
      const xcoord = Math.round(coordData[i*2] * (kWidth - 1));
      const ycoord = Math.round(coordData[i*2 + 1] * (kHeight - 1));
      nodes[i].position({x: xcoord, y: ycoord});
    }
  }
  layer.batchDraw();
}

async function loadIndices(fileBlob) {
  return new Promise((resolve, reject) => {
    let fr = new FileReader();

    fr.onload = () => {
      let data = fr.result;
      let array = new Uint32Array(data);
      resolve(array)
    };

    fr.onerror = reject;
    fr.readAsArrayBuffer(fileBlob);

  })
}

/**
 * Get a promise that loads the gene data.
 * @returns {Promise<*>}
 */
async function loadGenes(fileBlob) {
  return new Promise((resolve, reject) => {
    let fr = new FileReader();

    fr.onload = () => {
      let data = fr.result;
      let array = new Float32Array(data);
      resolve(array)
    };

    fr.onerror = reject;
    fr.readAsArrayBuffer(fileBlob);

  })
}

async function loadFile(filePath, blobAs) {
  return new Promise((resolve, reject) => {
    let xhr = new XMLHttpRequest();
    xhr.open('GET', filePath);
    xhr.responseType = 'blob';

    xhr.onload = async (e) => {
      let array = await blobAs(xhr.response);
      resolve(array);
    }

    xhr.onerror = () => reject(xhr.statusText);
    xhr.send();
  })
}

/**
 * A global to hold the MNIST data
 */
let dataSet;
let distSq;
let knnIndices;

let cancel = false;
let stepFlag= false;
/**
 * Run tf-tsne on the gene data and plot the data points
 * in a simple canvas.
 * @returns {Promise<void>}
 */
async function runTsne() {
  cancel = false;
  const rawTensor = tf.tensor(dataSet).
    reshape([105, 19992]);

  const allSampleTensor = rawTensor; //.transpose();
  allSampleTensor.print();
  console.log(allSampleTensor.shape);
  //const allSampleTensor = allGeneTensor.transpose();
  const numberData = allSampleTensor.shape[0];

  console.log(`calculating on: ${allSampleTensor.shape}`);
  const numIters = 1000;
  const geneOpt = tf_tsne.tsne(allSampleTensor, {
    perplexity : 12, // 10
    verbose : true,
    knnMode : 'bruteForce',
    exaggeration: 4, //4
    momentum: 0.8, //0.8
    //exaggerationIter: 300,
    //exaggerationDecayIter: 200,
  });

  await geneOpt.setKnnData(105, 28, distSq, knnIndices);

  const tsneIterElement = document.getElementById('tsneIterCount');
  // get the image data and access the data buffer to overwrite
  for(let i=0; i<numIters; i+=1) {
    await geneOpt.iterate(1);
    const coordData = await geneOpt.coordinates().data();
    displayTexture(
      geneOpt.optimizer.gpgpu.gl,
      geneOpt.optimizer.splatTexture,
      geneOpt.optimizer.splatTextureDiameter);
    plotCoordsKonva(numberData, coordData);
    tsneIterElement.innerHTML = 'tsne iteration: ' + (i + 1);
    // allow time for display
    if (cancel) {
      cancel = false;
      return;
    }
    /*while (true) {
      await sleep(1);
      if (stepFlag) {
        stepFlag = false;
        break;
      }
    }*/

  }
  console.log(`Tsne done`);
  tf.dispose(geneOpt);
}


function displayTexture(gl, texture, diameter) {
      var framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

      // Read the contents of the framebuffer
      var fdata = new Float32Array(diameter * diameter * 4);
      gl.readPixels(0, 0, diameter, diameter, gl.RGBA, gl.FLOAT, fdata);

      //gl.deleteFramebuffer(framebuffer);

      let color = 0;
      function getMax(prev, cur, ind) {
        if (ind%4 === color) {
          return (prev < cur) ? cur : prev;
        }
        return prev;
      }
      function getMin(prev, cur, ind) {
        if (ind%4 === color) {
          return (prev < cur) ? prev : cur;
        }
        return prev;
      }
      color = 0;
      const rangeMaxR = fdata.reduce(getMax, -1e6);
      const rangeMinR = fdata.reduce(getMin, 1e6);
      color = 1;
      const rangeMaxG = fdata.reduce(getMax, -1e6);
      const rangeMinG = fdata.reduce(getMin, 1e6);
      color = 2;
      const rangeMaxB = fdata.reduce(getMax, -1e6);
      const rangeMinB = fdata.reduce(getMin, 1e6);

      const rData = new Int32Array(diameter * diameter * 4);
      const bData = new Int32Array(diameter * diameter * 4);
      const gData = new Int32Array(diameter * diameter * 4);
      const rgbData = [rData, bData, gData];
      const rgbMins =[rangeMinR, rangeMinG, rangeMinB];
      const rgbMaxs =[rangeMaxR, rangeMaxG, rangeMaxB];

      // split and scale the first three components
      for (let i = 0; i < diameter * diameter * 4; i++) {
        const colIdx = i%4;
        const pixIdx = Math.floor(i/4) * 4;
        if (colIdx < 3) {
          let val = 0;
          if (colIdx > 0) {
            val = Math.floor(255 * fdata[i] / (rgbMaxs[colIdx] - rgbMins[colIdx]));
          } else {
            val = Math.floor(255 * fdata[i] / rgbMaxs[colIdx]);
          }
          if (colIdx === 0) {
            // svalues are 0 - 255 range
            rgbData[colIdx][pixIdx] = 255;
            rgbData[colIdx][pixIdx + 1] = 255 - val;
            rgbData[colIdx][pixIdx + 2] = 255 - val;
          }
          if (colIdx === 1 || colIdx == 2) {
            // Vx Vy val is in range -127 -> 127
            if (val >= 0) {
              rgbData[colIdx][pixIdx] = 255;
              rgbData[colIdx][pixIdx + 1] = 255 - 2 * val;
              rgbData[colIdx][pixIdx + 2] = 255 - 2 * val;
            }
            else{
              rgbData[colIdx][pixIdx + 2] = 255;
              rgbData[colIdx][pixIdx] = 255 - 2 * Math.abs(val);
              rgbData[colIdx][pixIdx + 1] = 255 - 2 * Math.abs(val);
            }
          }
          rgbData[colIdx][pixIdx + 3] = 255;

        }
      }
      let imgIds = ['canvasR', 'canvasG','canvasB'];
      for (let i = 0; i<3; i++) {
        displayImage(rgbData[i], diameter, imgIds[i]);
      }
    }

    function displayImage(data, diameter, id) {
      const imageTensor = tf.tensor3d(data, [diameter, diameter, 4], 'int32');
      const resizeImage = tf.image.resizeNearestNeighbor(imageTensor, [256, 256]);
      const resizeData = resizeImage.dataSync();
      // Create a 2D canvas to store the result
      var canvas = document.getElementById(id);
      canvas.width = 256;
      canvas.height = 256;
      var context = canvas.getContext('2d');

      // Copy the pixels to a 2D canvas
      var imageData = context.createImageData(256, 256);
      imageData.data.set(resizeData);
      context.putImageData(imageData, 0, 0);

      var img = new Image();
      img.src = canvas.toDataURL();
      return img;
    }

    /**
     * Handle the mousemove event to explore the points in the
     * plot canvas.
     * @param plotCanv
     * @param e
     */
    function plotExplore(plotCtx, e) {
      const x  = e.clientX - plotCtx.canvas.offsetLeft;
      const y  = e.clientY - plotCtx.canvas.offsetTop;
      const digitIndex = plotDigitIndex[y * plotCanv.width + x];
      if (digitIndex >= 1) {
        console.log(`digit idx: ${digitIndex}, label: ${labelSet[digitIndex]}`);
        const labelEl = document.getElementById('sampId');
        labelEl.innerText = labelSet[digitIndex];
      }
    }

    function restart() {
      cancel = true;
      setTimeout(async ()=> {
        initKonva();
        await runTsne()
  }, 1000)

}

function stop() {
  cancel = true;
}

function step() {
  stepFlag = true;
}

window.onload = async function() {
  initKonva();

  dataSet = await loadFile('../data/genescale_samplefirst.bin', loadGenes);
  distSq = await loadFile('../data/dist_sq_105.bin', loadGenes);
  knnIndices = await loadFile('../data/index_105.bin', loadIndices);

  // remove the self references in the loaded knn data
  const tempDist = new Float32Array(105 * 28);
  const tempInd = new Uint32Array(105 * 28);
  for (let i = 0; i<105; i++) {
    tempDist.set(distSq.slice(i*29 + 1, (i + 1)*29 + 1), i*28)
    tempInd.set(knnIndices.slice(i*29 + 1, (i + 1)*29 + 1), i*28)
  }
  distSq = tempDist;
  knnIndices = tempInd;



  document.getElementById('kNNSlider').oninput = () => {
    document.getElementById('sliderVal').innerHTML = 'max kNN iterations: ' + document.getElementById('kNNSlider').value;
  }
  document.getElementById('restartButton').addEventListener('click', restart);
  document.getElementById('stopButton').addEventListener('click', stop);
  document.getElementById('stepButton').addEventListener('click', step);
  await runTsne();
}
