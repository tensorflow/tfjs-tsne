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
import * as tf_tsne from '../../src/index';
import * as d3 from 'd3';

/**
 * Run the example
 */
async function start() {
  const numDimensions = 100;
  const numPoints = 20000;

  const data = generateData(numDimensions, numPoints);
  const coordinates = await computeEmbedding(data, numPoints);
  showEmbedding(coordinates);
}

/*
 * Generate some synthetic data to demonstrate the T-SNE implementation.
 *
 * The data is drawn from a straight line in the high dimensional space to which
 * random noise is added. The data must be a rank 2 tensor.
 */
function generateData(numDimensions, numPoints) {
  const data = tfc.tidy(() => {
    return tfc.linspace(0, 1, numPoints * numDimensions)
      .reshape([numPoints, numDimensions])
      .add(tfc.randomUniform([numPoints, numDimensions]));
  });
  return data;
}

/*
 * Computes our embedding.
 * 
 * This runs the T-SNE algorithm over our data tensor 
 * and returns x,y coordinates in embedding space.
 * 
 */
async function computeEmbedding(data, numPoints) {
  const tsne = tf_tsne.tsne(data, { 
    perplexity: 30,
    verbose: true,
    knnMode: 'auto',
  });

  // This will run the TSNE computation for 1000 steps.
  // Note that this may take a while.
  await tsne.compute(1000);

  // Get the coordinates (in embedding space) of the data
  const coordinates = await tsne.coordinates().data();

  const coords = [];
  for(let p = 0; p < numPoints; ++p){
    // TODO reshape this to a 2d array.
    const x = coordinates[p * 2];
    const y = coordinates[p * 2 + 1];
    coords.push([x, y]);
  }
  return coords;
}

/**
 * This will add a new plot visualizing the embedding space on a scatterplot.
 */
function showEmbedding(data) {
  const margin = {top: 20, right: 15, bottom: 60, left: 60};
  const width = 800 - margin.left - margin.right;
  const height = 800 - margin.top - margin.bottom;

  const x = d3.scaleLinear()
    .domain(d3.extent(data, d => d[0]))
    .range([0, width]);

  const y = d3.scaleLinear()
    .domain(d3.extent(data, d => d[1]))
    .range([height, 0]);

  const chart = d3.select('body')
    .append('svg')
    .attr('width', width + margin.right + margin.left)
    .attr('height', height + margin.top + margin.bottom)
    .attr('class', 'chart');

  const main = chart.append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
    .attr('width', width)
    .attr('height', height)
    .attr('class', 'main');

  const xAxis = d3.axisBottom(x);
  main.append('g')
    .attr('transform', 'translate(0,' + height + ')')
    .attr('class', 'main axis date')
    .call(xAxis);

  const yAxis = d3.axisLeft(y);
  main.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'main axis date')
    .call(yAxis);

  const dots = main.append('g');

  dots.selectAll('scatter-dots').data(data)
    .enter()
    .append('svg:circle')
      .attr('cx', (d) => x(d[0]))
      .attr('cy', (d) => y(d[1]))
      .attr('stroke-width', 0.25)
      .attr('stroke', '#1f77b4')
      .attr('fill', 'none')
      .attr('r', 5);
}

start();