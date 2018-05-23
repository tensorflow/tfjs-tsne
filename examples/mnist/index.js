import * as tf from '@tensorflow/tfjs-core';
import * as d3 from 'd3';

import * as tf_tsne from 'tfjs-tsne';
import {MnistData} from './data';


let canvas;
let context;

const width = 800;
const height = 800;

const colors = d3.scaleOrdinal(d3.schemeCategory10);
console.log('colors', colors[0], colors(0))

async function loadData() {
  const data = new MnistData();
  await data.load();
  return data;
}

async function doEmbedding(data, labels, numIterations, onIteration) {
  console.log('do Embedding', data, numIterations);
  const tsne = tf_tsne.tsne(data, undefined, true);
  console.time("IterateKNN");
  await tsne.iterateKnn(1000); // i want to not have to give this a number
  console.timeEnd("IterateKNN");

  for(let i = 0; i < numIterations; i++) {
    console.log("--doing iteration-- " + i);
    console.time('Iterate');
    await tsne.iterate(5);
    console.time('Iterate');
    const coordsData = await tsne.coordinates().data();
    onIteration(coordsData, labels);
    await tf.nextFrame();
  }
}

function renderEmbedding(coordinates, labels) {
  // console.log("render embedding", coordinates, labels);
  
  const coords = [];
  for (let i = 0; i < coordinates.length; i += 2) {
    coords.push([coordinates[i], coordinates[i + 1]]);
  }

  // console.log('coords array', coords);

  const x = d3.scaleLinear()
    .range([0, width])
    .domain(d3.extent(coords, d => d[0]));

  const y = d3.scaleLinear()
    .range([0, height])
    .domain(d3.extent(coords, d => d[1]));


  context.clearRect(0, 0, width, height);
  coords.forEach(function(d, i) {
    context.font = '10px sans';
    context.fillStyle = colors(parseInt(labels[i], 10));
    context.fillText(labels[i], x(d[0]), y(d[1]));
  });

  console.log("done drawing")
}

async function initVisualization() {
  canvas = d3.select('#vis').append('canvas')
    .attr('width', width)
    .attr('height', height);

  context = canvas.node().getContext('2d');
}


async function start() {
  const IMAGE_SIZE = 28 * 28;
  // const NUM_IMAGES = 65000;
  const NUM_IMAGES = 10000;
  const NUM_CLASSES = 10;
  const IMG_WIDTH = 28;
  const IMG_HEIGHT = 28;

  const NEW_HEIGHT = 10;
  const NEW_WIDTH = 10;

  const numIterations = 600;
  const data = await loadData();
  console.log(data);

  const inputData = data.datasetImages;
  const labelsTensor = tf.tensor2d(data.datasetLabels, [NUM_IMAGES, NUM_CLASSES]);
  const labels = labelsTensor.argMax(1).dataSync();


  const images = tf.tensor4d(inputData, [NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1]);
  const resized = images.resizeBilinear([NEW_HEIGHT, NEW_WIDTH]);
  console.log(images);
  console.log(resized);
  const reshaped = resized.reshape([NUM_IMAGES, NEW_HEIGHT * NEW_WIDTH])
  console.log(reshaped);
  doEmbedding(reshaped, labels, numIterations, renderEmbedding);
}

initVisualization();
start();