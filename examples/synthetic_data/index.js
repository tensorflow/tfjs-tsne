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
import * as d3 from 'd3';

import * as tsne from '../../src/index';

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
  const data = tf.tidy(() => {
    return tf.linspace(0, 1, numPoints * numDimensions)
        .reshape([numPoints, numDimensions])
        .add(tf.randomUniform([numPoints, numDimensions]));
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
  const embedder = tsne.tsne(data, {
    perplexity: 30,
    verbose: true,
    knnMode: 'auto',
  });

  // This will run the TSNE computation for 1000 steps.
  // Note that this may take a while.
  await embedder.compute(1000);

  // Get the normalized coordinates of the data
  return await embedder.coordsArray();
}

/**
 * This will add a new plot visualizing the embedding space on a scatterplot.
 */
function showEmbedding(data) {
  const margin = {top: 20, right: 15, bottom: 60, left: 60};
  const width = 800 - margin.left - margin.right;
  const height = 800 - margin.top - margin.bottom;

  const x = d3.scaleLinear().domain([0, 1]).range([0, width]);
  const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

  const chart = d3.select('body')
                    .append('svg')
                    .attr('width', width + margin.right + margin.left)
                    .attr('height', height + margin.top + margin.bottom)
                    .attr('class', 'chart');

  const main =
      chart.append('g')
          .attr(
              'transform', 'translate(' + margin.left + ',' + margin.top + ')')
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

  dots.selectAll('scatter-dots')
      .data(data)
      .enter()
      .append('svg:circle')
      .attr('cx', (d) => x(d[0]))
      .attr('cy', (d) => y(d[1]))
      .attr('stroke-width', 0.25)
      .attr('stroke', '#1f77b4')
      .attr('fill', 'none')
      .attr('r', 5);
}

document.addEventListener('DOMContentLoaded', () => start());
