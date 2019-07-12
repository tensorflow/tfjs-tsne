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

import Konva from 'konva';
const tsnejs = require('./tsne.js');

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
    setupTooltip();
    for (let i = 0; i < numberPoints; i++) {
      const xcoord = Math.round((coordData[i][0]/xRange) * (kWidth - 1) + 256);
      const ycoord = Math.round((coordData[i][1]/yRange) * (kHeight - 1) + 256);
      const colStr = colors[i].toString(16).padStart(6,'0');
      let node = new Konva.Circle({
        x: xcoord,
        y: ycoord,
        radius: 3.5,
        fill: '#' + colStr,
        id: labelSet[i]
      });
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


let cancel = false;
/**
 * Run tf-tsne on the gene data and plot the data points
 * in a simple canvas.
 * @returns {Promise<void>}
 */
async function runTsne() {
  cancel = false;
  let opt = {}
  opt.epsilon = 10; // epsilon is learning rate (10 = default)
  opt.perplexity = 10; // roughly how many neighbors each point influences (30 = default)
  opt.dim = 2; // dimensionality of the embedding (2 = default)

  let tsne = new tsnejs.tSNE(opt);
  let points = [];
  for (let p = 0; p<105; p++) {
    let row = [];
    for (let d = 0; d<19992; d++) {
      row.push(dataSet[p + d*105]);
    }
    points.push(row);
  }
  tsne.initDataRaw(points);
  const numIters = 1000;
  const numberData = 105;
  let coordData;
  const tsneIterElement = document.getElementById('tsneIterCount');
  for(let i=0; i<numIters; i+=1) {
    tsne.step();
    coordData = tsne.getSolution();
    plotCoordsKonva(numberData, coordData);
    tsneIterElement.innerHTML = 'tsne iteration: ' + (i + 1);
    await sleep(1);
  }
  console.log(`Tsne done`);
}

function restart() {
  cancel = true;
  setTimeout(async ()=> {
    initKonva();
    runTsne()
  }, 1000)

}

function stop() {
  cancel = true;
}

window.onload = async function() {
  initKonva();

  dataSet = await loadFile('../data/samplescale.bin', loadGenes);
  document.getElementById('restartButton').addEventListener('click', restart);
  document.getElementById('stopButton').addEventListener('click', stop);
  await runTsne();
}
