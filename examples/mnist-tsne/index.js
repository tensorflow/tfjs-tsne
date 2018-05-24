import * as tf from '@tensorflow/tfjs-core';
import * as d3 from 'd3';

import * as tf_tsne from 'tfjs-tsne';
import {MnistData} from './data';


let canvas;
let context;
let data;

const width = 800;
const height = 800;


const colors = d3.scaleOrdinal(d3.schemeCategory10);


async function loadData() {
  data = new MnistData();
  return data.load();
}

async function doEmbedding(
    data, labels, numIterations, knnIter, perplexity, onIteration) {
  const tsne = tf_tsne.tsne(data, undefined, true);
  console.time('IterateKNN ' + knnIter);
  await tsne.iterateKnn(knnIter);  // i want to not have to give this a number
  console.timeEnd('IterateKNN ' + knnIter);

  console.time('T-SNE iterate ' + numIterations);
  for (let i = 0; i < numIterations; i++) {
    await tsne.iterate(1);
    const coordsData = await tsne.coordinates().data();
    onIteration(coordsData, labels);
    await tf.nextFrame();
  }
  console.timeEnd('T-SNE iterate ' + numIterations);
}

function renderEmbedding(coordinates, labels) {
  const coords = [];
  for (let i = 0; i < coordinates.length; i += 2) {
    coords.push([coordinates[i], coordinates[i + 1]]);
  }

  const x =
      d3.scaleLinear().range([0, width]).domain(d3.extent(coords, d => d[0]));

  const y =
      d3.scaleLinear().range([0, height]).domain(d3.extent(coords, d => d[1]));


  context.clearRect(0, 0, width, height);
  coords.forEach(function(d, i) {
    context.font = '10px sans';
    context.fillStyle = colors(parseInt(labels[i], 10));
    context.fillText(labels[i], x(d[0]), y(d[1]));
  });
}

async function initVisualization() {
  canvas = d3.select('#vis')
               .append('canvas')
               .attr('width', width)
               .attr('height', height);

  context = canvas.node().getContext('2d');
}


async function start(numPoints = 10000, tsneIter, knnIter, perplexity) {
  const IMAGE_SIZE = 28 * 28;
  const NUM_IMAGES = 10000;
  const NUM_CLASSES = 10;
  const IMG_WIDTH = 28;
  const IMG_HEIGHT = 28;

  const NEW_HEIGHT = 10;
  const NEW_WIDTH = 10;

  console.log('START', data, numPoints, tsneIter, knnIter, perplexity);

  const [reshaped, labels] = tf.tidy(() => {
    const labelsTensor =
        tf.tensor2d(data.testLabels, [NUM_IMAGES, NUM_CLASSES]);

    const labels = labelsTensor.argMax(1).dataSync();

    const images = tf.tensor4d(data.testImages, [
                       NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1
                     ]).slice([0], [numPoints]);

    console.log(images);
    const resized = images.resizeBilinear([NEW_HEIGHT, NEW_WIDTH]);

    console.log(resized);
    const reshaped = resized.reshape([numPoints, NEW_HEIGHT * NEW_WIDTH])
    // console.log(reshaped);
    return [reshaped, labels];
  });

  // Actually perform and render the T-SNE
  await doEmbedding(
      reshaped, labels, tsneIter, knnIter, perplexity, renderEmbedding);
  reshaped.dispose();
}

function initControls() {
  $('#start').prop('disabled', false);

  $('#start').click(() => {
    const numPoints =
        parseInt($('input:radio[name=\'numPoints\']:checked').val(), 10);
    const perplexity = parseInt($('#perplexity-input').val(), 10);
    const tsneIter = parseInt($('#tsne-input').val(), 10);
    const knnIter = parseInt($('#knn-input').val(), 10);

    start(numPoints, tsneIter, knnIter, perplexity);
  })

  $('input[type=range]').on('input', (e) => {
    const newVal = e.target.value;
    $(e.target).prev().text(newVal);
  })
}

loadData().then(() => {
  initControls();
  initVisualization();
})
